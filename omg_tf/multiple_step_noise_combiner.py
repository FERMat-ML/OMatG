import torch
from torch_scatter import scatter_add, scatter_mean
from tqdm import trange
from omg.datamodule import OMGData
from omg.globals import SMALL_TIME, BIG_TIME
from omg.model.model import Model
from omg.utils import DataField
from omg_tf.abstracts import Combiner
from omg_tf.base_modules import base_modules


class MultipleStepNoiseCombiner(Combiner):
    """
    TODO: Think about an annealer and a nudge window or so? Would be nice to apply residual only in some steps at
          the beginning and then gradually increase the number of steps with residuals.
    TODO: This also has memory problems most likely. Add activation checkpointing? Right now, it can be only solved with
          small batch sizes and possibly accumulate_grad_batches.
    """

    def __init__(self, noise_scales: dict[str, float], integrate_noisy: bool = False) -> None:
        """Constructor of the MultipleStepNoiseCombiner class."""
        super().__init__(noise_scales=noise_scales, integrate_noisy=integrate_noisy)

    def training_integrate(
            self,
            residual_model: Model,
            x_0: OMGData
    ) -> tuple[OMGData, dict[DataField,torch.Tensor], dict[DataField,torch.Tensor]]:
        """
        Integrate the structures x_0 from time 0 to 1 with an Euler integration scheme relying on the added velocities
        of the base and residual models.

        This method performs an Euler integration similar to integrate_with_residual_means. However, it randomizes the
        residual velocities from the residual model by adding noise at every timestep. The returned log probabilities of
        the applied residuals for every integrated data field can then be used for policy gradient updates.

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
        for t_index in trange(1, len(times), desc="Integrating with residuals", position=1, leave=False):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            time = t.repeat(batch_size)
            with torch.no_grad():
                # Even though base model is frozen, we should call no_grad to avoid building computational graph.
                base_model_output = base_model(x_t, time)
            residual_output = residual_model(x_t, time)

            if self._integrate_pos:
                res_b = residual_output[DataField.pos.name + "_b"]
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
                log_probs[DataField.pos] += scatter_add(log_probs_atoms, x_t.batch)
                # Sum squared residuals over x, y, z dimensions.
                squared_res_pos = (res_b ** 2).sum(dim=-1)
                # Get batch-wise mean squared residuals for regularization.
                mean_squared_residuals[DataField.pos] += scatter_mean(squared_res_pos, x_t.batch)
                pos_b = base_model_output[DataField.pos.name + "_b"] + noisy_res_b
                x_t.pos = x_t.pos + pos_b * dt

            if self._integrate_cell:
                res_b = residual_output[DataField.cell.name + "_b"]
                noise_b = torch.randn_like(res_b)
                noisy_res_b = res_b + self._noise_scales[DataField.cell] * noise_b
                noisy_res_b_detached = noisy_res_b.detach()
                # Treat sampled action as constant for likelihood-ratio gradient.
                # Log probability: -0.5 * (log(2 π sigma^2) + ((x - mean) / sigma)^2)
                # We write it as ((noisy_res_b_detached - res_b) / sigma)^2 to maintain gradient connection to res_b.
                # Sum log probs over all dimensions except batch.
                sigma = torch.tensor(self._noise_scales[DataField.cell])
                log_probs[DataField.cell] += -0.5 * (
                        torch.log(2.0 * torch.pi * sigma**2)
                        + ((noisy_res_b_detached - res_b) / sigma) ** 2
                ).sum(dim=tuple(range(1, noise_b.ndim)))
                # Get batch-wise mean squared residuals for regularization.
                mean_squared_residuals[DataField.cell] += (res_b ** 2).mean(dim=tuple(range(1, res_b.ndim)))
                cell_b = base_model_output[DataField.cell.name + "_b"] + noisy_res_b
                x_t.cell = x_t.cell + cell_b * dt

        # Average mean squared residuals over time steps for regularization.
        for key in mean_squared_residuals.keys():
            mean_squared_residuals[key] /= (len(times) - 1)

        return x_t, log_probs, mean_squared_residuals
