from enum import Enum, auto
from pathlib import Path
import time
from typing import Dict, Optional
from ase import Atoms
import lightning
import numpy as np
from scipy.stats import wasserstein_distance
import torch
from omg.analysis import get_coordination_numbers, get_cov, match_rmsds, ValidAtoms
from omg.datamodule import OMGData
from omg.globals import SMALL_TIME, BIG_TIME
from omg.model.model import Model
from omg.sampler.minimum_permutation_distance import correct_for_minimum_permutation_distance
from omg.sampler.sampler import Sampler
from omg.si.abstracts import StochasticInterpolantSpecies
from omg.si.stochastic_interpolants import StochasticInterpolants
from omg.si.single_stochastic_interpolant_identity import SingleStochasticInterpolantIdentity
from omg.utils import xyz_saver


class OMGLightning(lightning.LightningModule):
    """
    Lightning module of OMatG which defines the model and training logic.

    This class contains the stochastic interpolants for the atomic positions ("pos" data field in
    torch_geometric.data.Data instances), the cell vectors ("cell" data field), and the atomic numbers ("species" data
    field) of the materials structures.

    This class contains the base distribution for each data field by using a sampling method for each data
    field of a sampler class.

    The underlying model is trained to predict the stochastic interpolants for these data fields at a sampled time t,
    which is sampled (uniformly or with a Sobol sequence) from the interval [SMALL_TIME, BIG_TIME]. During training,
    the stochastic interpolants interpolate between a data sample and a sample from the base distribution. Optionally,
    the sample from the base distribution can be permuted to minimize the fractional-coordinate distance between the
    data sample and the sample from the base distribution.

    The stochastic interpolants compute a set of losses that all have different loss keys. This class combines these
    losses by weighting them with relative costs that are provided as a dictionary mapping the loss keys to the
    relative costs. The sum of the relative costs must be 1.

    This class allows for three different validation modes:
    - "loss": Only the validation loss on the validation dataset is computed as the sum of the weighted losses and
              logged as "val_loss_total".
    - "match_rate": A structure is generated for each composition in the validation dataset, and the match rate and
                    average root-mean-square distance between matched structures are logged. This validation mode
                    requires that the stochastic interpolant for the species keeps them unchanged so that the model
                    works in crystal-structure prediction mode. This validation mode is slow so it should not be run on
                    every epoch.
    - "dng_eval": A structure is generated for each number of atoms in the validation dataset, and the de-novo
                  generation metrics are logged. Additionally, an averaged metric is logged as "dng_eval". This
                  validation mode is slow so it should not be run on every epoch.

    :param si:
        Collection of stochastic interpolants used for training and generation/prediction.
    :type si: StochasticInterpolants
    :param sampler:
        Sampler used to sample initial structures from the base distribution based on data samples.
    :type sampler: Sampler
    :param model:
        Model architecture used for training and generation/prediction.
    :type model: Model
    :param relative_si_costs:
        Relative costs for the stochastic interpolants used in the loss function.
    :type relative_si_costs: Dict[str, float]
    :param use_min_perm_dist:
        Whether to use the minimum permutation distance coupling during training.
        Defaults to False.
    :type use_min_perm_dist: bool
    :param generation_xyz_filename:
        If not None, the filename to which the generated structures will be saved in XYZ format during prediction.
        If None, the filename will be generated based on the current time.
        Defaults to None.
    :type generation_xyz_filename: Optional[str]
    :param sobol_time:
        Whether to use Sobol sampling for the time variable t.
        If False, uniform sampling is used.
        Defaults to False.
    :type sobol_time: bool
    :param float_32_matmul_precision:
        Precision for float32 matrix multiplication in torch (see
        https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html).
        Possible values are "medium", "high", and "highest".
        Defaults to "medium".
    :type float_32_matmul_precision: str
    :param validation_mode:
        The mode for validation metrics. Possible values are "loss", "match_rate", and "dng_eval".
        Defaults to "loss".
    :type validation_mode: str
    :param dataset_name:
        The name of the dataset used for training and validation.
        This is only relevant if validation_mode is "dng_eval".
        Possible values are "mp_20", "carbon_24", "perov_5", "mpts_52", and "alex_mp_20".
        Defaults to "mp_20".
    :type dataset_name: str
    :param number_cpus:
        The number of CPUs to use for parallel processing during validation and matching.
        This is only relevant if validation_mode is "match_rate" or "dng_eval".
        Defaults to 12.
    :type number_cpus: int

    :raises ValueError:
        If the interpolant for the species is not of type StochasticInterpolantSpecies.
        If the relative costs do not sum to 1.
        If any cost is negative.
        If the relative costs do not contain all loss keys from the stochastic interpolants or vice versa.
        If the validation mode is not one of "loss", "match_rate", or "dng_eval".
        If the species stochastic interpolant is not of type SingleStochasticInterpolantIdentity when
        validation_mode is "match_rate".
        If the dataset_name is not one of "mp_20", "carbon_24", "perov_5", "mpts_52", or "alex_mp_20".
        If the number of CPUs is less than 1.
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
        DNG_EVAL = auto()
        """
        Evaluation for the DNG task.
        """

    def __init__(self, si: StochasticInterpolants, sampler: Sampler, model: Model,
                 relative_si_costs: Dict[str, float], use_min_perm_dist: bool = False,
                 generation_xyz_filename: Optional[str] = None, sobol_time: bool = False,
                 float_32_matmul_precision: str = "medium", validation_mode: str = "loss",
                 dataset_name: str = "mp_20", number_cpus: int = 12) -> None:
        """Constructor of the OMGLightning class."""
        super().__init__()
        self.si = si
        self.sampler = sampler
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
        if self._validation_metric == self.ValidationMetric.MATCH_RATE:
            if not isinstance(species_stochastic_interpolant, SingleStochasticInterpolantIdentity):
                raise ValueError("Species stochastic interpolant must be of type SingleStochasticInterpolantIdentity "
                                 "for match rate validation.")
        self.dataset_name = dataset_name
        if not self.dataset_name in ["mp_20", "carbon_24", "perov_5", "mpts_52", "alex_mp_20"]:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")
        self.number_cpus = number_cpus
        if not self.number_cpus >= 1:
            raise ValueError(f"Number of CPUs {self.number_cpus} must be >= 1.")

        # Possible values are "medium", "high", and "highest".
        torch.set_float32_matmul_precision(float_32_matmul_precision)

        self.reference_atoms = []
        self.generated_atoms = []

    def training_step(self, x_1: OMGData) -> torch.Tensor:
        """
        Performs one training step given a batch of structures x_1 from the training dataset.

        This method samples the initial structures x_0 from the base distribution using the sampler, possibly
        permutes the initial structures to minimize the permutational distance, samples times t for each structure,
        computes the stochastic interpolants, and computes the losses based on the model predictions for the velocity
        fields and denoisers at sampled times t. The total loss is computed as the weighted sum of the individual
        losses using the provided relative costs.

        :return:
            Loss from training step.
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

    def on_validation_epoch_start(self) -> None:
        """
        Called at the start of the validation epoch to reset the lists of reference and generated atoms.

        These lists are used to store the reference structures from the validation dataset and the generated
        structures during validation, which are later used for computing the match rate or de-novo generation
        metrics at the end of the validation epoch.

        This is only relevant if the validation metric is set to "match_rate" or "dng_eval".
        """
        self.reference_atoms = []
        self.generated_atoms = []

    def validation_step(self, x_1: OMGData) -> torch.Tensor:
        """
        Performs one validation step given a batch of structures x_1 from the validation dataset.

        This method samples the initial structures x_0 from the base distribution using the sampler, possibly
        permutes the initial structures to minimize the permutational distance, samples times t for each
        structure, computes the stochastic interpolants, and computes the losses based on the model predictions for the
        velocity fields and denoisers at sampled times t. The total loss is computed as the weighted sum of the
        individual losses using the provided relative costs.

        If the validation metric is set to "match_rate" or "dng_eval", it also generates structures and stores them
        in the `self.generated_atoms` list, and stores the reference structures from the validation dataset
        in the `self.reference_atoms` list for later evaluation at the end of the validation epoch.

        :param x_1:
            Batch of structures from the validation dataset.
        :type x_1: OMGData
        """
        batch_size = len(x_1.n_atoms)
        x_0 = self.sampler.sample_p_0(x_1).to(self.device)

        if (self._validation_metric == self.ValidationMetric.MATCH_RATE
                or self._validation_metric == self.ValidationMetric.DNG_EVAL):
            # Prevent moving x_1 to cpu because it's needed below.
            x_1_cpu = x_1.clone().to('cpu')
            for i in range(batch_size):
                lower, upper = x_1_cpu.ptr[i], x_1_cpu.ptr[i + 1]
                self.reference_atoms.append(
                    Atoms(numbers=x_1_cpu.species[lower:upper], scaled_positions=x_1_cpu.pos[lower:upper, :],
                          cell=x_1_cpu.cell[i, :, :], pbc=(1, 1, 1)))

            gen = self.si.integrate(x_0, self.model, save_intermediate=False)
            gen.to('cpu')
            assert len(gen.n_atoms) == batch_size
            assert torch.equal(gen.n_atoms, x_1_cpu.n_atoms)
            assert torch.equal(gen.ptr, x_1_cpu.ptr)
            for i in range(batch_size):
                lower, upper = gen.ptr[i], gen.ptr[i + 1]
                self.generated_atoms.append(
                    Atoms(numbers=gen.species[lower:upper], scaled_positions=gen.pos[lower:upper, :],
                          cell=gen.cell[i, :, :], pbc=(1, 1, 1)))
        else:
            assert self._validation_metric == self.ValidationMetric.LOSS

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
            losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size
        )

        return total_loss

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch to compute and log the validation metrics.

        This method computes the match rate or de-novo generation metrics based on the generated structures
        and the reference structures from the validation dataset, depending on the validation metric set during
        initialization. It logs the computed metrics, such as match rate, mean RMSD, valid rate, Wasserstein
        distances for density, n-arity, and coordination numbers, and coverage (for mp_20, carbon_24, and perov_5
        datasets), as well as the average of these metrics.

        This is only relevant if the validation metric is set to "match_rate" or "dng_eval".
        """
        if self._validation_metric == self.ValidationMetric.MATCH_RATE:
            assert len(self.generated_atoms) == len(self.reference_atoms)
            gen_valid_atoms = ValidAtoms.get_valid_atoms(self.generated_atoms, desc="Validating generated structures",
                                                         skip_validation=True, number_cpus=1)
            ref_valid_atoms = ValidAtoms.get_valid_atoms(self.reference_atoms, desc="Validating reference structures",
                                                         skip_validation=True, number_cpus=1)

            match_rate, mean_rmsd, _, _, _, _, corr_rmsd, _ = match_rmsds(
                gen_valid_atoms, ref_valid_atoms, ltol=0.3, stol=0.5, angle_tol=10.0, number_cpus=self.number_cpus,
                enable_progress_bar=True)

            self.log("match_rate", float(match_rate), sync_dist=True)
            self.log("mean_rmsd", float(mean_rmsd), sync_dist=True)
            self.log("corr_rmsd", float(corr_rmsd), sync_dist=True)
        elif self._validation_metric == self.ValidationMetric.DNG_EVAL:
            assert len(self.generated_atoms) == len(self.reference_atoms)
            gen_valid_atoms = ValidAtoms.get_valid_atoms(self.generated_atoms, desc="Validating generated structures",
                                                         number_cpus=self.number_cpus)
            ref_valid_atoms = ValidAtoms.get_valid_atoms(self.reference_atoms, desc="Validating reference structures",
                                                         number_cpus=self.number_cpus)
            valid_rate = sum(va.valid for va in gen_valid_atoms) / len(gen_valid_atoms)

            gen_densities = [struc.structure.density for struc in gen_valid_atoms]
            ref_densities = [struc.structure.density for struc in ref_valid_atoms]
            wdist_density = wasserstein_distance(gen_densities, ref_densities)
            gen_narity = [len(set(struc.structure.species)) for struc in gen_valid_atoms]
            ref_narity = [len(set(struc.structure.species)) for struc in ref_valid_atoms]
            wdist_narity = wasserstein_distance(gen_narity, ref_narity)
            gen_coordination_numbers = [np.mean(get_coordination_numbers(struc.atoms)) for struc in gen_valid_atoms]
            ref_coordination_numbers = [np.mean(get_coordination_numbers(struc.atoms)) for struc in ref_valid_atoms]
            wdist_coordination_numbers = wasserstein_distance(gen_coordination_numbers, ref_coordination_numbers)
            wdist_avg = np.average([wdist_density, wdist_narity, wdist_coordination_numbers])

            if self.dataset_name in ("mp_20", "carbon_24", "perov_5"):
                # Taken from https://github.com/jiaor17/DiffCSP/blob/7121d159826efa2ba9500bf299250d96da37f146/scripts/compute_metrics.py
                COV_Cutoffs = {
                    "mp_20": {"struc": 0.4, "comp": 10.0},
                    "carbon_24": {"struc": 0.2, "comp": 4.0},
                    "perov_5": {"struc": 0.2, "comp": 4}
                }
                struc_cutoff = COV_Cutoffs[self.dataset_name]["struc"]
                comp_cutoff = COV_Cutoffs[self.dataset_name]["comp"]
                cov_precision, cov_recall = get_cov(gen_valid_atoms, ref_valid_atoms, struc_cutoff, comp_cutoff, None)
                cov_avg = np.average([1.0 - cov_precision, 1.0 - cov_recall])
                metrics = {
                    "valid_rate": valid_rate,
                    "wdist_density": wdist_density,
                    "wdist_narity": wdist_narity,
                    "wdist_coordination_numbers": wdist_coordination_numbers,
                    "cov_precision": cov_precision,
                    "cov_recall": cov_recall,
                }
                dng_eval = float(np.average([cov_avg, wdist_avg, 1.0 - valid_rate]))
            else:
                assert self.dataset_name in ["mpts_52", "alex_mp_20"]
                metrics = {
                    "valid_rate": valid_rate,
                    "wdist_density": wdist_density,
                    "wdist_narity": wdist_narity,
                    "wdist_coordination_numbers": wdist_coordination_numbers,
                }
                dng_eval = float(np.average([wdist_avg, 1.0 - valid_rate]))

            self.log("dng_eval", dng_eval, sync_dist=True)
            self.log_dict(metrics, sync_dist=True)
        else:
            assert self._validation_metric == self.ValidationMetric.LOSS

    def predict_step(self, x: OMGData) -> OMGData:
        """
        Performs one prediction step given a batch of structures x_1 from the prediction dataset.

        This method samples the initial structures x_0 from the base distribution using the sampler, integrates
        the stochastic interpolants to generate new structures, and saves the generated structures in XYZ format.
        The initial structures are saved in a separate file with "_init" appended to the filename stem.
        If the `generation_xyz_filename` is not set, the filename is generated based on the current time.

        :param x:
            Batch of structures from the prediction dataset.
        :type x: OMGData

        :return:
            Generated structures after integrating the stochastic interpolants.
        :rtype: OMGData
        """
        x_0 = self.sampler.sample_p_0(x).to(self.device)
        gen, inter = self.si.integrate(x_0, self.model, save_intermediate=True)
        filename = (Path(self.generation_xyz_filename) if self.generation_xyz_filename is not None
                    else Path(f"{time.strftime('%Y%m%d-%H%M%S')}.xyz"))
        init_filename = filename.with_stem(filename.stem + "_init")
        xyz_saver(x_0.to("cpu"), init_filename)
        xyz_saver(gen.to("cpu"), filename)
        return gen
