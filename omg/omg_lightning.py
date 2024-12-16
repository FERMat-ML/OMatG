from pathlib import Path
import time
from typing import Optional, Sequence
import lightning as L
import torch
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
                 relative_si_costs: Sequence[float], learning_rate: float = 0.001,
                 load_checkpoint: Optional[str] = None, use_min_perm_dist: bool = False,
                 generation_xyz_filename: Optional[str] = None, sobol_time: bool = False) -> None:
        super().__init__()
        if load_checkpoint:
            checkpoint = torch.load(load_checkpoint, map_location=self.device)
            self.load_state_dict(checkpoint['state_dict'])
        else:
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
            model = model.double()  # TODO: Should this be an option?
            self.model = model

            if not len(relative_si_costs) == len(self.si):
                raise ValueError("The number of stochastic interpolants and costs must be equal.")
            if not all(cost >= 0.0 for cost in relative_si_costs):
                raise ValueError("All cost factors must be non-negative.")
            if not abs(sum(relative_si_costs) - 1.0) < 1e-10:
                raise ValueError("The sum of all cost factors should be equal to 1.")
            self._relative_si_costs = relative_si_costs

            if not sobol_time:
                self.time_sampler = torch.rand
            else:
                self.time_sampler = lambda n: torch.reshape(
                    torch.quasirandom.SobolEngine(dimension=1, scramble=True).draw(n), (-1, ))
            self.generation_xyz_filename = generation_xyz_filename

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
    def training_step(self, x_1) -> torch.Tensor:
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

        for cost, loss_key in zip(self._relative_si_costs, losses):
            losses[loss_key] = cost * losses[loss_key]
            total_loss += losses[loss_key]
        # TODO: Look at how SDE losses are combined

        assert "loss_total" not in losses
        losses["loss_total"] = total_loss

        self.log_dict(
            losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def validation_step(self, x_1) -> torch.Tensor:
        """
        Performs one validation step given a batch of x_1
        """

        x_0 = self.sampler.sample_p_0(x_1).to(self.device)

        # Sample t for each structure.
        t = self.time_sampler(len(x_1.n_atoms)).to(self.device)

        losses = self.si.losses(self.model, t, x_0, x_1)

        total_loss = torch.tensor(0.0, device=self.device)

        for cost, loss_key in zip(self._relative_si_costs, losses):
            losses[f"val_{loss_key}"] = cost * losses[loss_key]
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
        )

        return total_loss

    # TODO: what do we want to return
    def predict_step(self, x):
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
