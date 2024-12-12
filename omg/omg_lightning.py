from pathlib import Path
import time
from typing import Optional, Sequence
from ase.io import read
import lightning as L
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
                 relative_si_costs: Sequence[float], load_checkpoint: Optional[str] = None,
                 learning_rate: Optional[float] = 1.e-3, lr_scheduler: Optional[bool] = False,
                 use_min_perm_dist: bool = False, generation_xyz_filename: Optional[str] = None,
                 overfitting_test: bool = False) -> None:
        super().__init__()
        self.si = si
        self.sampler = sampler
        self.learning_rate = learning_rate
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
        if load_checkpoint:
            checkpoint = torch.load(load_checkpoint, map_location=self.device)
            self.load_state_dict(checkpoint['state_dict'])
        self.lr_scheduler = lr_scheduler
        self.generation_xyz_filename = generation_xyz_filename
        self.overfitting_test = overfitting_test

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

    def on_fit_start(self):
        if self.learning_rate:
            # Overwrite learning rate after running LearningRateFinder
            for optimizer in self.trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["learning_rate"] = self.learning_rate

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

        # sample t uniformly for each structure
        t = torch.rand(len(x_1.n_atoms)).to(self.device)

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

        # sample t uniformly for each structure
        t = torch.rand(len(x_1.n_atoms)).to(self.device)

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

    @staticmethod
    def _structure_matcher(s1, s2, ltol=0.2, stol=0.3, angle_tol=5.0):
        """ Checks if structures s1 and s2 of ase type Atoms are the same."""
        sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
        # conversion to pymatgen type
        a1 = AseAtomsAdaptor.get_structure(s1)
        a2 = AseAtomsAdaptor.get_structure(s2)
        return sm.fit(a1, a2)

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

        if self.overfitting_test:
            base_filename = filename.with_stem(filename.stem + "_base")
            xyz_saver(x.to("cpu"), base_filename)
            atoms_one = read(base_filename, index=":")
            atoms_two = read(filename, index=":")
            assert len(atoms_one) == len(atoms_two)
            successes = 0
            for a_one, a_two in zip(atoms_one, atoms_two):
                assert len(a_one) == len(a_two)
                if self._structure_matcher(a_one, a_two):
                    successes += 1
            print(f"Overfitting test: {successes}/{len(atoms_one)} structures are the same "
                  f"({successes / len(atoms_one) * 100.0} percent success rate).")

        return gen

    # TODO allow for YAML config
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.lr_scheduler:
            lr_scheduler = ReduceLROnPlateau(optimizer, patience=40)
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss_total",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
            }
            return {"optimizer": optimizer,
                    "lr_scheduler": lr_scheduler_config
                    }
        else:
            return optimizer
