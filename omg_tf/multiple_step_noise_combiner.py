import torch
from torch_scatter import scatter_add
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

    def __init__(self, noise_scales: dict[str, float]) -> None:
        """Constructor of the MultipleStepNoiseResidualTransformer class."""
        super().__init__(noise_scales=noise_scales)

    def training_integrate(self, residual_model: Model, x_0: OMGData) -> tuple[OMGData, torch.Tensor]:
        """
        Integrate the structures x_0 from time 0 to 1 with an Euler integration scheme relying on the added velocities
        of the base and residual models.

        This method performs an Euler integration similar to integrate_with_residual_means. However, during training, it
        randomizes the residual velocities from the residual model by adding noise at every timestep. The returned log
        probabilities of the applied residuals can then be used for policy gradient updates.

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
        for t_index in trange(1, len(times), desc="Integrating with residuals"):
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
                # Log probability of the specific sampled velocity x is -0.5 * ((x - mean) / sigma)^2.
                # Since we sample x = mean + sigma * noise, this becomes -0.5 * noise^2.
                # Sum log probs over all dimensions except atom dimension.
                log_probs_atoms = -0.5 * (noise_b ** 2).sum(dim=tuple(range(1, noise_b.ndim)))
                # Sum log probs over all atoms in each structure to get batch-wise log probs.
                log_probs += scatter_add(log_probs_atoms, x_t.batch)
                pos_b = base_model_output[DataField.pos.name + "_b"] + noisy_res_b
                x_t.pos = x_t.pos + pos_b * dt

            if self._integrate_cell:
                res_b = residual_output[DataField.cell.name + "_b"]
                noise_b = torch.randn_like(res_b)
                noisy_res_b = res_b + self._noise_scales[DataField.cell] * noise_b
                # Log probability of the specific sampled velocity x is -0.5 * ((x - mean) / sigma)^2.
                # Since we sample x = mean + sigma * noise, this becomes -0.5 * noise^2.
                # Sum log probs over all dimensions except batch.
                log_probs += -0.5 * (noise_b ** 2).sum(dim=tuple(range(1, noise_b.ndim)))
                cell_b = base_model_output[DataField.cell.name + "_b"] + noisy_res_b
                x_t.cell = x_t.cell + cell_b * dt

        return x_t, log_probs
