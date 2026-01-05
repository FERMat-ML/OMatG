import torch
from torch_geometric.data import Batch
from torch_scatter import scatter_add, scatter_mean
from tqdm import trange
from omg.datamodule import OMGData
from omg.globals import SMALL_TIME, BIG_TIME
from omg.model.model import Model
from omg.utils import DataField
from omg_tf.abstracts import Combiner
from omg_tf.base_modules import base_modules


class SingleStepNoiseCombiner(Combiner):
    def __init__(self, noise_scales: dict[str, float], integrate_noisy: bool = False) -> None:
        """Constructor of the SingleStepNoiseCombiner class."""
        super().__init__(noise_scales=noise_scales, integrate_noisy=integrate_noisy)

    def training_integrate(
            self,
            residual_model: Model,
            x_0: OMGData
    ) -> tuple[OMGData, dict[DataField, torch.Tensor], dict[DataField, torch.Tensor]]:
        """
        Integrate the structures x_0 from time 0 to 1 with an Euler integration scheme relying on the added velocities
        of the base and residual models.

        This method performs an Euler integration similar to integrate_with_residual_means. It applies the mean residual
        at all timesteps (matching inference behavior), but adds noise at a single randomly chosen timestep per
        structure. At non-noisy timesteps, the residual model is called with torch.no_grad() to avoid building a
        computational graph through the entire trajectory. The returned log probability of the applied residuals for
        each integrated data field can then be used for policy gradient updates.

        In addition, this method also returns the mean squared residuals per integrated data field for regularization
        purposes.

        :param residual_model:
            The residual model predicting residual velocities.
        :type residual_model: Model
        :param x_0:
            Initial structures at time 0.
        :type x_0: OMGData

        :return:
            (Structures at final time 1 after integration with residuals,
             Batch-wise log probabilities of the applied residuals for each integrated data field,
             Batch-wise mean squared residuals for each integrated data field).
        :rtype: tuple[OMGData, dict[DataField, torch.tensor], dict[DataField, torch.tensor]]
        """
        base_model = base_modules["model"].model
        assert base_model is not None
        batch_size = len(x_0.n_atoms)
        times = torch.linspace(SMALL_TIME, BIG_TIME, self._integration_time_steps, device=x_0.pos.device)
        x_t = x_0.clone()
        log_probs = {key: torch.zeros(batch_size, device=x_0.pos.device)
                     for key in self._integrated_data_fields}
        mean_squared_residuals = {key: torch.zeros(batch_size, device=x_0.pos.device)
                                  for key in self._integrated_data_fields}
        noise_time_steps = torch.randint(low=1, high=len(times), size=(batch_size,), device=x_0.pos.device)
        for t_index in trange(1, len(times), desc="Integrating with residuals", position=1, leave=False):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            time = t.repeat(batch_size)
            with torch.no_grad():
                # Even though base model is frozen, we should call no_grad to avoid building computational graph.
                base_model_output = base_model(x_t, time)

            mask_per_structure = (noise_time_steps == t_index)
            structure_filter = mask_per_structure.nonzero(as_tuple=True)[0]
            non_noisy_filter = (~mask_per_structure).nonzero(as_tuple=True)[0]

            # For structures not at their noisy timestep, apply base + mean residual (no gradients).
            if len(non_noisy_filter) > 0:
                # noinspection PyUnresolvedReferences
                non_noisy_x_t = Batch.from_data_list(x_t.index_select(non_noisy_filter)).to(x_t.pos.device)
                non_noisy_time = time[non_noisy_filter]
                with torch.no_grad():
                    non_noisy_residual_output = residual_model(non_noisy_x_t, non_noisy_time)
                if self._integrate_pos:
                    non_noisy_mask_per_atom = (~mask_per_structure).repeat_interleave(x_0.n_atoms)
                    base_vel = base_model_output[DataField.pos.name + "_b"][non_noisy_mask_per_atom]
                    res_vel = non_noisy_residual_output[DataField.pos.name + "_b"]
                    x_t.pos[non_noisy_mask_per_atom] = x_t.pos[non_noisy_mask_per_atom] + (base_vel + res_vel) * dt
                if self._integrate_cell:
                    base_vel = base_model_output[DataField.cell.name + "_b"][non_noisy_filter]
                    res_vel = non_noisy_residual_output[DataField.cell.name + "_b"]
                    x_t.cell[non_noisy_filter] = x_t.cell[non_noisy_filter] + (base_vel + res_vel) * dt

            if len(structure_filter) == 0:
                continue

            # For structures at their noisy timestep, apply base + noisy residual (with gradients).
            # noinspection PyUnresolvedReferences
            filtered_x_t = Batch.from_data_list(x_t.index_select(structure_filter)).to(x_t.pos.device)
            filtered_time = time[structure_filter]
            filtered_residual_output = residual_model(filtered_x_t, filtered_time)

            if self._integrate_pos:
                res_b = filtered_residual_output[DataField.pos.name + "_b"]
                noise_b = torch.randn_like(res_b)
                noisy_res_b = res_b + self._noise_scales[DataField.pos] * noise_b
                # Treat sampled action as constant for likelihood-ratio gradient.
                noisy_res_b_detached = noisy_res_b.detach()
                # Log probability: -0.5 * (log(2 π sigma^2) + ((x - mean) / sigma)^2)
                # We write it as ((noisy_res_b_detached - res_b) / sigma)^2 to maintain gradient connection to res_b.
                # Sum log probs over all dimensions except atom dimension.
                sigma = torch.tensor(self._noise_scales[DataField.pos])
                log_probs_atoms = -0.5 * (
                        torch.log(2.0 * torch.pi * sigma**2)
                        + ((noisy_res_b_detached - res_b) / sigma) ** 2
                ).sum(dim=tuple(range(1, noise_b.ndim)))
                # Sum log probs over all atoms in each structure to get batch-wise log probs.
                log_probs_filtered = scatter_add(log_probs_atoms, filtered_x_t.batch)
                log_probs[DataField.pos][structure_filter] += log_probs_filtered
                # Sum squared residuals over x, y, z dimensions.
                squared_res_pos = (res_b ** 2).sum(dim=-1)
                # Get batch-wise mean squared residuals for regularization.
                mean_squared_residuals_filtered = scatter_mean(squared_res_pos, filtered_x_t.batch)
                mean_squared_residuals[DataField.pos][structure_filter] += mean_squared_residuals_filtered
                # Apply base + noisy residual to filtered structures.
                mask_per_atom = mask_per_structure.repeat_interleave(x_0.n_atoms)
                base_vel = base_model_output[DataField.pos.name + "_b"][mask_per_atom]
                x_t.pos[mask_per_atom] = x_t.pos[mask_per_atom] + (base_vel + noisy_res_b) * dt

            if self._integrate_cell:
                res_b = filtered_residual_output[DataField.cell.name + "_b"]
                noise_b = torch.randn_like(res_b)
                noisy_res_b = res_b + self._noise_scales[DataField.cell] * noise_b
                # Treat sampled action as constant for likelihood-ratio gradient.
                noisy_res_b_detached = noisy_res_b.detach()
                # Log probability: -0.5 * (log(2 π sigma^2) + ((x - mean) / sigma)^2)
                # We write it as ((noisy_res_b_detached - res_b) / sigma)^2 to maintain gradient connection to res_b.
                # Sum log probs over all dimensions except batch.
                sigma = torch.tensor(self._noise_scales[DataField.cell])
                log_probs_filtered = -0.5 * (
                        torch.log(2.0 * torch.pi * sigma**2)
                        + ((noisy_res_b_detached - res_b) / sigma) ** 2
                ).sum(dim=tuple(range(1, noise_b.ndim)))
                log_probs[DataField.cell][structure_filter] += log_probs_filtered
                # Get batch-wise mean squared residuals for regularization.
                mean_squared_residuals_filtered = (res_b ** 2).mean(dim=tuple(range(1, res_b.ndim)))
                mean_squared_residuals[DataField.cell][structure_filter] += mean_squared_residuals_filtered
                # Apply base + noisy residual to filtered structures.
                base_vel = base_model_output[DataField.cell.name + "_b"][mask_per_structure]
                x_t.cell[mask_per_structure] = x_t.cell[mask_per_structure] + (base_vel + noisy_res_b) * dt

        # No mean over time necessary for mean_squared_residuals since it consists only of a single timestep.
        return x_t, log_probs, mean_squared_residuals
