import torch
from torch_geometric.data import Batch
from tqdm import trange
from omg.datamodule import OMGData
from omg.globals import SMALL_TIME, BIG_TIME
from omg.model.model import Model
from omg.utils import DataField
from omg_tf.abstracts import Combiner
from omg_tf.base_modules import base_modules


class SingleStepNoiseAnnealer(Combiner):
    """
    Combiner that scales base model velocities by a learned factor (1 + scale) at a single timestep.

    This combiner is similar to SingleStepNoiseCombiner, but instead of adding residual velocities,
    it scales the base model velocities by a factor of (1 + scale), where scale is predicted by the
    residual model via pos_s and cell_s outputs. This allows the model to learn to speed up or slow
    down the base model's velocities.

    At most timesteps, only the base model velocity is used. At a single randomly chosen timestep
    per structure, the residual model predicts scalar factors that are used to scale the base model
    velocities. Noise is added to the scalar factors for exploration.
    """

    def __init__(self, noise_scales: dict[str, float]) -> None:
        """Constructor of the SingleStepNoiseAnnealer class."""
        super().__init__(noise_scales=noise_scales)

    def training_integrate(
            self,
            residual_model: Model,
            x_0: OMGData
    ) -> tuple[OMGData, dict[DataField, torch.Tensor], dict[DataField, torch.Tensor]]:
        """
        Integrate the structures x_0 from time 0 to 1 with an Euler integration scheme, scaling
        base model velocities by learned factors at a single timestep.

        At most timesteps, only the base model velocity is used. At a single randomly chosen timestep
        per structure, the residual model predicts scalar factors (pos_s, cell_s) that scale the base
        model velocities by (1 + scale + noise). The log probability and mean squared values of the
        scales are returned for policy gradient updates and regularization.

        :param residual_model:
            The residual model predicting scalar factors (pos_s, cell_s).
        :type residual_model: Model
        :param x_0:
            Initial structures at time 0.
        :type x_0: OMGData

        :return:
            (Structures at final time 1 after integration,
             Batch-wise log probabilities of the applied scales for each integrated data field,
             Batch-wise mean squared scales for each integrated data field).
        :rtype: tuple[OMGData, dict[DataField, torch.tensor], dict[DataField, torch.tensor]]
        """
        base_model = base_modules["model"].model
        assert base_model is not None
        batch_size = len(x_0.n_atoms)
        times = torch.linspace(SMALL_TIME, BIG_TIME, self._integration_time_steps, device=x_0.pos.device)
        x_t = x_0.clone()
        log_probs = {key: torch.zeros(batch_size, device=x_0.pos.device)
                     for key in self._integrated_data_fields}
        mean_squared_scales = {key: torch.zeros(batch_size, device=x_0.pos.device)
                               for key in self._integrated_data_fields}
        noise_time_steps = torch.randint(low=1, high=len(times), size=(batch_size,), device=x_0.pos.device)
        for t_index in trange(1, len(times), desc="Integrating with scaling", position=1, leave=False):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            time = t.repeat(batch_size)
            with torch.no_grad():
                # Even though base model is frozen, we should call no_grad to avoid building computational graph.
                base_model_output = base_model(x_t, time)

            mask_per_structure = (noise_time_steps == t_index)
            structure_filter = mask_per_structure.nonzero(as_tuple=True)[0]
            non_noisy_filter = (~mask_per_structure).nonzero(as_tuple=True)[0]

            # For structures not at their noisy timestep, apply scaled velocities (no gradients).
            if len(non_noisy_filter) > 0:
                # noinspection PyUnresolvedReferences
                non_noisy_x_t = Batch.from_data_list(x_t.index_select(non_noisy_filter)).to(x_t.pos.device)
                non_noisy_time = time[non_noisy_filter]
                with torch.no_grad():
                    non_noisy_residual_output = residual_model(non_noisy_x_t, non_noisy_time)
                if self._integrate_pos:
                    non_noisy_mask_per_atom = (~mask_per_structure).repeat_interleave(x_0.n_atoms)
                    base_vel = base_model_output[DataField.pos.name + "_b"][non_noisy_mask_per_atom]
                    # pos_s is per-structure scalar factor [non_noisy_batch_size, 1].
                    scale = non_noisy_residual_output[DataField.pos.name + "_s"]
                    # Expand scale to match atom dimension.
                    scale_per_atom = scale.repeat_interleave(x_0.n_atoms[non_noisy_filter], dim=0)
                    # Apply scaled velocity to non-noisy structures.
                    x_t.pos[non_noisy_mask_per_atom] = (x_t.pos[non_noisy_mask_per_atom]
                                                        + base_vel * (1.0 + scale_per_atom) * dt)
                if self._integrate_cell:
                    base_vel = base_model_output[DataField.cell.name + "_b"][non_noisy_filter]
                    # cell_s is per-structure scalar factor [non_noisy_batch_size, 1].
                    scale = non_noisy_residual_output[DataField.cell.name + "_s"]
                    # Expand scale to match cell dimensions [non_noisy_batch_size, 3, 3].
                    scale_expanded = scale.unsqueeze(-1)  # [non_noisy_batch_size, 1, 1]
                    # Apply scaled velocity to non-noisy structures.
                    x_t.cell[non_noisy_filter] = (x_t.cell[non_noisy_filter]
                                                  + base_vel * (1.0 + scale_expanded) * dt)

            if len(structure_filter) == 0:
                continue

            # For structures at their noisy timestep, apply scaled velocities with noise (with gradients).
            # noinspection PyUnresolvedReferences
            filtered_x_t = Batch.from_data_list(x_t.index_select(structure_filter)).to(x_t.pos.device)
            filtered_time = time[structure_filter]
            filtered_residual_output = residual_model(filtered_x_t, filtered_time)

            if self._integrate_pos:
                # pos_s is per-structure scalar factor [filtered_batch_size, 1]
                scale_mean = filtered_residual_output[DataField.pos.name + "_s"]
                noise = torch.randn_like(scale_mean)
                noisy_scale = scale_mean + self._noise_scales[DataField.pos] * noise
                # Treat sampled action as constant for likelihood-ratio gradient.
                noisy_scale_detached = noisy_scale.detach()
                # Log probability: -0.5 * (log(2 π sigma^2) + ((noisy_scale_detached - scale_mean) / sigma)^2)
                # This maintains gradient connection to scale_mean
                sigma = torch.tensor(self._noise_scales[DataField.pos])
                log_probs_filtered = -0.5 * (
                        torch.log(2.0 * torch.pi * sigma**2)
                        + ((noisy_scale_detached - scale_mean) / sigma) ** 2
                ).squeeze(-1)
                log_probs[DataField.pos][structure_filter] += log_probs_filtered
                # Mean squared scale for regularization (already per-structure)
                mean_squared_scales_filtered = (scale_mean ** 2).squeeze(-1)
                mean_squared_scales[DataField.pos][structure_filter] += mean_squared_scales_filtered
                # Apply base + scaled velocity to filtered structures.
                mask_per_atom = mask_per_structure.repeat_interleave(x_0.n_atoms)
                base_vel_filtered = base_model_output[DataField.pos.name + "_b"][mask_per_atom]
                # Expand noisy_scale to match atom dimension.
                noisy_scale_per_atom = noisy_scale.repeat_interleave(x_0.n_atoms[structure_filter], dim=0)
                x_t.pos[mask_per_atom] = x_t.pos[mask_per_atom] + base_vel_filtered * (1.0 + noisy_scale_per_atom) * dt

            if self._integrate_cell:
                # cell_s is per-structure scalar factor [filtered_batch_size, 1]
                scale_mean = filtered_residual_output[DataField.cell.name + "_s"]  # [batch, 1]
                noise = torch.randn_like(scale_mean)
                noisy_scale = scale_mean + self._noise_scales[DataField.cell] * noise
                # Treat sampled action as constant for likelihood-ratio gradient.
                noisy_scale_detached = noisy_scale.detach()
                # Log probability: -0.5 * (log(2 π sigma^2) + ((noisy_scale_detached - scale_mean) / sigma)^2)
                # This maintains gradient connection to scale_mean
                sigma = torch.tensor(self._noise_scales[DataField.cell])
                log_probs_filtered = -0.5 * (
                        torch.log(2.0 * torch.pi * sigma**2)
                        + ((noisy_scale_detached - scale_mean) / sigma) ** 2
                ).squeeze(-1)
                log_probs[DataField.cell][structure_filter] += log_probs_filtered
                # Mean squared scale for regularization (already per-structure)
                mean_squared_scales_filtered = (scale_mean ** 2).squeeze(-1)
                mean_squared_scales[DataField.cell][structure_filter] += mean_squared_scales_filtered
                # Apply base + scaled velocity to filtered structures.
                base_vel_filtered = base_model_output[DataField.cell.name + "_b"][mask_per_structure]
                # Expand noisy_scale to match cell dimensions [filtered_batch_size, 3, 3]
                noisy_scale_expanded = noisy_scale.unsqueeze(-1)  # [filtered_batch_size, 1, 1]
                x_t.cell[mask_per_structure] = (x_t.cell[mask_per_structure]
                                                + base_vel_filtered * (1.0 + noisy_scale_expanded) * dt)

        # No mean over time necessary for mean_squared_scales since it consists only of a single timestep.
        return x_t, log_probs, mean_squared_scales

    def integrate(self, residual_model: Model, x_0: OMGData) -> OMGData:
        """
        Integrate the structures x_0 from time 0 to 1 with an Euler integration scheme, scaling
        base model velocities by learned factors (deterministic, no noise).

        :param residual_model:
            The residual model predicting scalar factors (pos_s, cell_s).
        :type residual_model: Model
        :param x_0:
            Initial structures at time 0.
        :type x_0: OMGData

        :return:
            Structures at final time 1 after integration.
        :rtype: OMGData
        """
        base_model = base_modules["model"].model
        assert base_model is not None
        batch_size = len(x_0.n_atoms)
        times = torch.linspace(SMALL_TIME, BIG_TIME, self._integration_time_steps, device=x_0.pos.device)
        x_t = x_0.clone()

        for t_index in trange(1, len(times), desc="Integrating with scaling", position=1, leave=False):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            time = t.repeat(batch_size)
            base_model_output = base_model(x_t, time)
            residual_output = residual_model(x_t, time)

            if self._integrate_pos:
                scale = residual_output[DataField.pos.name + "_s"]  # [batch, 1]
                scale_per_atom = scale.repeat_interleave(x_0.n_atoms, dim=0)
                base_vel = base_model_output[DataField.pos.name + "_b"]
                x_t.pos = x_t.pos + base_vel * (1.0 + scale_per_atom) * dt

            if self._integrate_cell:
                scale = residual_output[DataField.cell.name + "_s"]  # [batch, 1]
                scale_expanded = scale.unsqueeze(-1)  # [batch, 1, 1]
                base_vel = base_model_output[DataField.cell.name + "_b"]
                x_t.cell = x_t.cell + base_vel * (1.0 + scale_expanded) * dt

        return x_t
