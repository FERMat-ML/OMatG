from pathlib import Path
import time
from typing import Dict, Optional, Sequence
import lightning as L
import torch
from omg.datamodule import OMGData
from omg.model.model import Model
from omg.sampler.minimum_permutation_distance import correct_for_minimum_permutation_distance
from omg.sampler.sampler import Sampler
from omg.si.abstracts import StochasticInterpolantSpecies
from omg.si.stochastic_interpolants import StochasticInterpolants
from omg.utils import xyz_saver


class OMGLightning(L.LightningModule):
    """
    Main module which is fit and used to generate structures using Lightning CLI.
    """
    def __init__(self, si: StochasticInterpolants, sampler: Sampler, model: Model,
                 relative_si_costs: Dict[str, float], learning_rate: float = 0.001, use_min_perm_dist: bool = False,
                 generation_xyz_filename: Optional[str] = None, sobol_time: bool = False,
                 float_32_matmul_precision: str = "medium") -> None:
        super().__init__()
        self.si = si
        self.sampler = sampler
        self.lr = learning_rate  # Learning rate must be stored in this class for learning rate finder.
        self.use_min_perm_dist = use_min_perm_dist
        if self.use_min_perm_dist:
            self._pos_corrector = self.si.get_stochastic_interpolant("pos").get_corrector()
        else:
            self._pos_corrector = None
        species_stochastic_interpolant = self.si.get_stochastic_interpolant("species")
        if not isinstance(species_stochastic_interpolant, StochasticInterpolantSpecies):
            raise ValueError("Species stochastic interpolant must be of type StochasticInterpolantSpecies.")
        if species_stochastic_interpolant.uses_masked_species():
            model.enable_masked_species()
        self.model = model

        if not all(cost >= 0.0 for cost in relative_si_costs.values()):
            raise ValueError("All cost factors must be non-negative.")
        if not abs(sum(cost for cost in relative_si_costs.values()) - 1.0) < 1e-10:
            raise ValueError("The sum of all cost factors should be equal to 1.")
        si_loss_keys = self.si.loss_keys()
        for key in relative_si_costs.keys():
            if key not in si_loss_keys:
                raise ValueError(f"Loss key {key} not found in the stochastic interpolants.")
        for key in si_loss_keys:
            if key not in relative_si_costs.keys():
                raise ValueError(f"Loss key {key} not found in the costs.")
        self._relative_si_costs = relative_si_costs

        if not sobol_time:
            self.time_sampler = torch.rand
        else:
            self.time_sampler = lambda n: torch.reshape(
                torch.quasirandom.SobolEngine(dimension=1, scramble=True).draw(n), (-1, ))
        self.generation_xyz_filename = generation_xyz_filename

        # Possible values are "medium", "high", and "highest".
        torch.set_float32_matmul_precision(float_32_matmul_precision)

    def forward(self, x_t: Sequence[torch.Tensor], t: torch.Tensor) -> Sequence[Sequence[torch.Tensor]]:
        """
        Calls encoder + head stack

        :param x_t:
            Sequence of torch.tensors corresponding to batched species, fractional coordinates and lattices.
        :type x_t: Sequence[torch.Tensor]

        :param t:
            Sampled times
        :type t: torch.Tensor

        :return:
            Predicted b and etas for species, coordinates and lattices, respectively.
        :rtype: Sequence[Sequence[torch.Tensor]]
        """
        x = self.model(x_t, t)
        return x

    def on_fit_start(self) -> None:
        """
        Set the learning rate of the optimizers to the learning rate of this class.
        """
        for optimizer in self.trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr

    # TODO: specify argument types
    def training_step(self, x_1: OMGData) -> torch.Tensor:
        """
        Performs one training step given a batch of x_1

        :return:
            Loss from training step
        :rtype: torch.Tensor
        """
        x_0 = self.sampler.sample_p_0(x_1).to(self.device)

        # Minimize permutational distance between clusters.
        if self.use_min_perm_dist:
            # Don't switch species to allow for crystal-structure prediction.
            correct_for_minimum_permutation_distance(x_0, x_1, self._pos_corrector, switch_species=False)

        # Sample t for each structure.
        t = self.time_sampler(len(x_1.n_atoms)).to(self.device)

        losses = self.si.losses(self.model, t, x_0, x_1)

        total_loss = torch.tensor(0.0, device=self.device)

        for loss_key in losses:
            losses[loss_key] = self._relative_si_costs[loss_key] * losses[loss_key]
            total_loss += losses[loss_key]

        assert "loss_total" not in losses
        losses["loss_total"] = total_loss

        self.log_dict(
            losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size
        )

        return total_loss

    def validation_step(self, x_1: OMGData) -> torch.Tensor:
        """
        Performs one validation step given a batch of x_1
        """

        x_0 = self.sampler.sample_p_0(x_1).to(self.device)

        # Sample t for each structure.
        t = self.time_sampler(len(x_1.n_atoms)).to(self.device)

        losses = self.si.losses(self.model, t, x_0, x_1)

        total_loss = torch.tensor(0.0, device=self.device)

        for loss_key in losses:
            losses[f"val_{loss_key}"] = self._relative_si_costs[loss_key] * losses[loss_key]
            total_loss += losses[f"val_{loss_key}"]
            losses.pop(loss_key)

        assert "loss_total" not in losses
        losses["val_loss_total"] = total_loss

        self.log_dict(
            losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size
        )

        return total_loss

    # TODO: what do we want to return
    def predict_step(self, x: OMGData) -> OMGData:
        """
        Performs generation
        """
        x_0 = self.sampler.sample_p_0(x).to(self.device)
        gen, inter = self.si.integrate(x_0, self.model, save_intermediate=True)
        # probably want to turn structure back into some other object that's easier to work with
        filename = (Path(self.generation_xyz_filename) if self.generation_xyz_filename is not None
                    else Path(f"{time.strftime("%Y%m%d-%H%M%S")}.xyz"))
        init_filename = filename.with_stem(filename.stem + "_init")
        xyz_saver(x_0.to("cpu"), init_filename)
        xyz_saver(gen.to("cpu"), filename)
        return gen
