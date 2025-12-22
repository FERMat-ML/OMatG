import torch
from torch_geometric.data import Batch
from torch_scatter import scatter_add
from tqdm import trange
from omg.datamodule import OMGData
from omg.globals import SMALL_TIME, BIG_TIME
from omg.model.model import Model
from omg.utils import DataField
from omg_tf.abstracts import Combiner
from omg_tf.base_modules import base_modules


class SingleStepNoiseCombiner(Combiner):
    def __init__(self, noise_scales: dict[str, float]) -> None:
        """Constructor of the SingleStepNoiseCombiner class."""
        super().__init__(noise_scales=noise_scales)

    def training_integrate(self, residual_model: Model, x_0: OMGData) -> tuple[OMGData, torch.Tensor]:
        """
        Integrate the structures x_0 from time 0 to 1 with an Euler integration scheme relying on the added velocities
        of the base and residual models.

        This method performs an Euler integration similar to integrate_with_residual_means. During training, however, it
        uses only the velocity from the base model at most timesteps. Only at a single randomly chosen timestep per
        structure, it randomizes the residual velocities from the residual model by adding noise, and adds that to the
        velocity of the base model. The returned log probability of the applied residuals can then be used for policy
        gradient updates.

        TODO: IS THERE A WAY TO ALWAYS APPLY MEAN?

        :param residual_model:
            The residual model predicting residual velocities.
        :type residual_model: Model
        :param x_0:
            Initial structures at time 0.
        :type x_0: OMGData

        :return:
            (Structures at final time 1 after integration with residuals,
             Batch-wise log probabilities of the applied residuals).
        :rtype: tuple[OMGData, torch.tensor]
        """
        base_model = base_modules["model"].model
        assert base_model is not None
        batch_size = len(x_0.n_atoms)
        times = torch.linspace(SMALL_TIME, BIG_TIME, self._integration_time_steps, device=x_0.pos.device)
        x_t = x_0.clone()
        log_probs = torch.zeros(batch_size, device=x_0.pos.device)
        noise_time_steps = torch.randint(low=1, high=len(times), size=(batch_size,), device=x_0.pos.device)
        for t_index in trange(1, len(times), desc="Integrating with residuals"):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            time = t.repeat(batch_size)
            with torch.no_grad():
                # Even though base model is frozen, we should call no_grad to avoid building computational graph.
                base_model_output = base_model(x_t, time)

            mask_per_structure = (noise_time_steps == t_index)
            structure_filter = mask_per_structure.nonzero(as_tuple=True)[0]
            if len(structure_filter) == 0:
                # Only use base model velocities at this time step.
                if self._integrate_pos:
                    x_t.pos = x_t.pos + base_model_output[DataField.pos.name + "_b"] * dt
                if self._integrate_cell:
                    x_t.cell = x_t.cell + base_model_output[DataField.cell.name + "_b"] * dt
                continue

            # noinspection PyUnresolvedReferences
            filtered_x_t = Batch.from_data_list(x_t.index_select(structure_filter)).to(x_t.pos.device)
            filtered_time = time[structure_filter]
            # Base model was not applied to x_t yet.
            filtered_residual_output = residual_model(filtered_x_t, filtered_time)

            if self._integrate_pos:
                res_b = filtered_residual_output[DataField.pos.name + "_b"]
                noise_b = torch.randn_like(res_b)
                noisy_res_b = res_b + self._noise_scales[DataField.pos] * noise_b
                # Log probability of the specific sampled velocity x is -0.5 * ((x - mean) / sigma)^2.
                # Since we sample x = mean + sigma * noise, this becomes -0.5 * noise^2.
                # Sum log probs over all dimensions except atom dimension.
                log_probs_atoms = -0.5 * (noise_b ** 2).sum(dim=tuple(range(1, noise_b.ndim)))
                # Sum log probs over all atoms in each structure to get batch-wise log probs.
                log_probs_filtered = scatter_add(log_probs_atoms, filtered_x_t.batch)
                log_probs[structure_filter] += log_probs_filtered
                x_t.pos = x_t.pos + base_model_output[DataField.pos.name + "_b"] * dt
                mask_per_atom = mask_per_structure.repeat_interleave(x_0.n_atoms)
                x_t.pos[mask_per_atom] = x_t.pos[mask_per_atom] + noisy_res_b * dt

            if self._integrate_cell:
                res_b = filtered_residual_output[DataField.cell.name + "_b"]
                noise_b = torch.randn_like(res_b)
                noisy_res_b = res_b + self._noise_scales[DataField.cell] * noise_b
                # Log probability of the specific sampled velocity x is -0.5 * ((x - mean) / sigma)^2.
                # Since we sample x = mean + sigma * noise, this becomes -0.5 * noise^2.
                # Sum log probs over all dimensions except batch.
                log_probs_filtered = -0.5 * (noise_b ** 2).sum(dim=tuple(range(1, noise_b.ndim)))
                log_probs[structure_filter] += log_probs_filtered
                x_t.cell = x_t.cell + base_model_output[DataField.cell.name + "_b"] * dt
                x_t.cell[mask_per_structure] = x_t.cell[mask_per_structure] + noisy_res_b * dt

        return x_t, log_probs
