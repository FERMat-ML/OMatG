from enum import Enum, auto
from pathlib import Path
import time
from typing import Dict, Optional, Sequence
from ase import Atoms
import lightning as L
import torch
from omg.analysis import match_rate_and_rmsd, ValidAtoms
from omg.datamodule import OMGData
from omg.globals import SMALL_TIME, BIG_TIME
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

    class ValidationMetric(Enum):
        """
        Enum for the possible types of reported validation metrics.
        """

        LOSS = auto()
        """
        Ordinary loss.
        """
        MATCH_RATE = auto()
        """
        Match rate for the CSP task.
        """

    def __init__(self, si: StochasticInterpolants, sampler: Sampler, model: Model,
                 relative_si_costs: Dict[str, float], learning_rate: float = 0.001, use_min_perm_dist: bool = False,
                 generation_xyz_filename: Optional[str] = None, sobol_time: bool = False,
                 float_32_matmul_precision: str = "medium", validation_mode: str = "loss") -> None:
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
            # Don't sample between 0 and 1 because the gamma function may diverge as t -> 0 or t -> 1, which
            # may result in NaN losses during training if t was to close to 0 or 1 (especially at 32-true precision).
            self.time_sampler = lambda n: torch.rand(n) * (BIG_TIME - SMALL_TIME) + SMALL_TIME
        else:
            # Don't sample between 0 and 1 because the gamma function may diverge as t -> 0 or t -> 1, which
            # may result in NaN losses during training if t was to close to 0 or 1 (especially at 32-true precision).
            self.time_sampler = (
                lambda n: torch.reshape(torch.quasirandom.SobolEngine(dimension=1, scramble=True).draw(n), (-1,))
                          * (BIG_TIME - SMALL_TIME) + SMALL_TIME)
        self.generation_xyz_filename = generation_xyz_filename

        try:
            self._validation_metric = self.ValidationMetric[validation_mode.upper()]
        except AttributeError:
            raise ValueError(f"Unknown validation metric f{validation_mode}.")
        self.matches = 0
        self.rmsd = 0
        self.counts = 0

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
            batch_size=len(x_1.n_atoms)
        )

        return total_loss

    def validation_step(self, x_1: OMGData) -> torch.Tensor:
        batch_size = len(x_1.n_atoms)
        x_0 = self.sampler.sample_p_0(x_1).to(self.device)

        if self._validation_metric == self.ValidationMetric.MATCH_RATE:
            gen = self.si.integrate(x_0, self.model, save_intermediate=False)
            gen.to('cpu')
            # Prevent moving x_1 to cpu because it's needed below.
            x_1_cpu = x_1.clone().to('cpu')
            x_1_atoms = []
            gen_atoms = []
            assert torch.equal(gen.n_atoms, x_1_cpu.n_atoms)
            assert torch.equal(gen.ptr, x_1_cpu.ptr)
            assert torch.equal(gen.species, x_1_cpu.species)
            for i in range(batch_size):
                lower, upper = x_1_cpu.ptr[i], x_1_cpu.ptr[i + 1]
                x_1_atoms.append(Atoms(numbers=x_1_cpu.species[lower:upper],
                                       scaled_positions=x_1_cpu.pos[lower:upper, :],
                                       cell=x_1_cpu.cell[i, :, :], pbc=(1, 1, 1)))
                gen_atoms.append(Atoms(numbers=gen.species[lower:upper], scaled_positions=gen.pos[lower:upper, :],
                                       cell=gen.cell[i, :, :], pbc=(1, 1, 1)))
            gen_valid_atoms = ValidAtoms.get_valid_atoms(gen_atoms, desc="Validating generated structures",
                                                         skip_validation=True, number_cpus=1)
            ref_valid_atoms = ValidAtoms.get_valid_atoms(x_1_atoms, desc="Validating reference structures",
                                                         skip_validation=True, number_cpus=1)

            match1, rmsd1, _, _ = match_rate_and_rmsd(gen_valid_atoms, ref_valid_atoms, ltol=0.3, stol=0.5,
                                                      angle_tol=10.0, number_cpus=1)

            self.matches += match1 * batch_size
            self.rmsd += rmsd1 * batch_size
            self.counts += batch_size
            gen_losses = {
                "matches": match1 * batch_size,
                "total_rmsd": rmsd1 * batch_size,
            }
        else:
            gen_losses = {}

        # Minimize permutational distance between clusters. Should be done after integrating.
        if self.use_min_perm_dist:
            # Don't switch species to allow for crystal-structure prediction.
            correct_for_minimum_permutation_distance(x_0, x_1, self._pos_corrector, switch_species=False)

        # Sample t for each structure.
        t = self.time_sampler(len(x_1.n_atoms)).to(self.device)

        losses = self.si.losses(self.model, t, x_0, x_1)

        total_loss = torch.tensor(0.0, device=self.device)

        # Force creation of copy of keys because dictionary will be changed in iteration.
        for loss_key in list(losses):
            losses[f"val_{loss_key}"] = self._relative_si_costs[loss_key] * losses[loss_key]
            total_loss += losses[f"val_{loss_key}"]
            losses.pop(loss_key)

        assert "loss_total" not in losses
        losses["val_loss_total"] = total_loss

        self.log_dict(
            losses | gen_losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size
        )

        return total_loss

    def on_validation_epoch_start(self):
        self.matches = 0
        self.rmsd = 0
        self.counts = 0

    def on_validation_epoch_end(self):
        if self._validation_metric == self.ValidationMetric.MATCH_RATE:
            self.log("match_rate", self.matches / self.counts, sync_dist=True)
            self.log("mean_rmsd", self.rmsd / self.counts, sync_dist=True)

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
