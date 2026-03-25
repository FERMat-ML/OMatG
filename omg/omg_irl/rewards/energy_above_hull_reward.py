import logging
from typing import Literal, Optional, Sequence
import numpy as np
import tqdm
import warnings
import torch
from pymatgen.core import Composition
from omg.datamodule import Structure
from omg.utils import prefixed_stdout
from .abstracts import ComputeStage, Reward

with warnings.catch_warnings(), prefixed_stdout(prefix="[MACE] "):
    warnings.simplefilter("ignore", category=UserWarning)
    from mace.calculators import mace_mp
with prefixed_stdout(prefix="[TorchSim] "):
    from torch_sim import static
    from torch_sim.autobatching import BinningAutoBatcher
    from torch_sim.models.mace import MaceModel

from ._energy_above_hull import get_energy_above_hull


class EnergyAboveHullReward(Reward):
    """
    Reward function that encourages thermodynamic stability by minimizing the energy above the convex hull.

    This reward is designed for de-novo generation (DNG) where different structures in a GRPO group may have different
    compositions. Unlike raw energy per atom, energy above hull is a composition-aware stability metric that measures
    the distance to the convex hull of known stable phases (in eV/atom), making it comparable across compositions.

    The reward pipeline:
    1. Compute total energy of each structure using MACE-MPA-0.
    2. Extract the composition from the generated species.
    3. Compute the energy above the convex hull using reference phase diagrams from the LeMat-GenBench dataset.
    4. Return the negative energy above hull as the reward (lower e_above_hull = more stable = higher reward).

    Invalid structures (based on volume, interatomic distance, and polar sine criteria) are assigned a configurable
    penalty value.

    The convex hull phase diagrams are cached per unique set of elements to avoid expensive recomputation.

    :param scale:
        Scaling factor for the reward.
        Must be positive.
        Defaults to 1.0.
    :type scale: float
    :param hull_type:
        Type of convex hull to use for the reference phase diagram.
        "mace_omat" uses the OMatG hull, "mace_mp" uses the Materials Project hull.
        Defaults to "mace_omat".
    :type hull_type: Literal["mace_mp", "mace_omat"]
    :param hull_threshold:
        Energy above hull threshold in eV/atom for filtering the reference dataset.
        Defaults to 0.001.
    :type hull_threshold: float
    :param device:
        The device to run MACE on. Can be "cpu" or "cuda".
        Defaults to "cpu".
    :type device: Literal["cpu", "cuda"]
    :param default_dtype:
        The default dtype to use for MACE calculations. Can be "float32" or "float64".
        Defaults to "float64".
    :type default_dtype: Literal["float32", "float64"]
    :param enable_cueq:
        Whether to enable CuEq in MACE.
        Defaults to False.
    :type enable_cueq: bool
    :param max_memory_scaler:
        The max_memory_scaler parameter for TorchSim's BinningAutoBatcher when using CUDA.
        Defaults to 500000.0.
    :type max_memory_scaler: float
    :param invalid_penalty:
        Penalty energy above hull in eV/atom to assign to invalid structures.
        If None, no penalty is applied and energies for invalid structures are computed normally.
        Defaults to None.
    :type invalid_penalty: Optional[float]
    :param volume_check_cutoff:
        Minimum volume for which to compute the energy normally.
        Must be non-negative.
        Defaults to 0.1.
    :type volume_check_cutoff: float
    :param structure_check_cutoff:
        Minimum interatomic distance threshold.
        Must be non-negative.
        Defaults to 0.5.
    :type structure_check_cutoff: float
    :param polar_sine_cutoff:
        Minimum polar sine threshold.
        Must be non-negative.
        Defaults to 1.0e-3.
    :type polar_sine_cutoff: float

    :raises ValueError:
        If scale is not positive.
        If volume_check_cutoff is negative.
        If structure_check_cutoff is negative.
        If polar_sine_cutoff is negative.
    """

    def __init__(self, scale: float = 1.0, hull_type: Literal["mace_mp", "mace_omat"] = "mace_omat",
                 hull_threshold: float = 0.001, device: Literal["cpu", "cuda"] = "cpu",
                 default_dtype: Literal["float32", "float64"] = "float64", enable_cueq: bool = False,
                 max_memory_scaler: float = 500000.0, invalid_penalty: Optional[float] = None,
                 volume_check_cutoff: float = 0.1, structure_check_cutoff: float = 0.5,
                 polar_sine_cutoff: float = 1.0e-3) -> None:
        """Constructor for EnergyAboveHullReward."""
        super().__init__()
        if not scale > 0.0:
            raise ValueError("Scale must be positive.")
        if not volume_check_cutoff >= 0.0:
            raise ValueError("Volume check cutoff must be non-negative.")
        if not structure_check_cutoff >= 0.0:
            raise ValueError("Structure check cutoff must be non-negative.")
        if not polar_sine_cutoff >= 0.0:
            raise ValueError("Polar sine cutoff must be non-negative.")
        self._scale = scale
        self._hull_type = hull_type
        self._hull_threshold = hull_threshold
        self._device = device
        self._default_dtype = default_dtype
        # Catch warnings from MACE and prefix stdout.
        with prefixed_stdout("[MACE] "), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if self._device == "cpu":
                logging.disable(logging.WARNING)  # MACE prints a warning when using CPU, disable it.
                # Calling mace_mp with return_raw_model=False changes the default dtype of torch.
                self._mace_model = mace_mp(model="medium-mpa-0", device=device, default_dtype=default_dtype,
                                           enable_cueq=enable_cueq)
                logging.disable(logging.NOTSET)  # Undo the disabling of logging.
                torch.set_default_dtype(torch.float32)  # Undo what happened in mace_mp construction.
                self._batcher = None
                if max_memory_scaler != 500000.0:
                    warnings.warn("max_memory_scaler is only used when device is 'cuda', the specified value will be "
                                  "ignored.")
            else:
                # TorchSim's batching does not work on CPU.
                assert self._device == "cuda"
                mace = mace_mp(model="medium-mpa-0", device=device, default_dtype=default_dtype,
                               enable_cueq=enable_cueq, return_raw_model=True)
                # noinspection PyTypeChecker
                self._mace_model = MaceModel(model=mace, device=device, compute_forces=False, compute_stress=False,
                                             dtype=torch.float64 if default_dtype == "float64" else torch.float32,
                                             enable_cueq=enable_cueq)
                self._batcher = BinningAutoBatcher(model=self._mace_model, max_memory_scaler=max_memory_scaler,
                                                   memory_scales_with="n_atoms_x_density")
        self._invalid_penalty = invalid_penalty
        self._volume_check_cutoff = volume_check_cutoff
        self._structure_check_cutoff = structure_check_cutoff
        self._polar_sine_cutoff = polar_sine_cutoff

    def _is_valid(self, structure: Structure) -> bool:
        """Check if the structure is valid based on volume, interatomic distances, and polar sine criteria."""
        py_structure = structure.get_pymatgen_structure()
        volume_valid = (py_structure.volume >= self._volume_check_cutoff)
        dist_mat = py_structure.distance_matrix
        dist_mat = dist_mat + np.diag(
            np.ones(dist_mat.shape[0]) * (self._structure_check_cutoff + 10.0)
        )
        min_dist = dist_mat.min()
        structure_valid = (min_dist >= self._structure_check_cutoff)
        polar_sine = py_structure.lattice.volume / np.prod(py_structure.lattice.lengths)
        polar_sine_valid = (polar_sine >= self._polar_sine_cutoff)
        return volume_valid and structure_valid and polar_sine_valid

    def _compute_energy_above_hull(self, total_energy: float, composition: Composition) -> float:
        """
        Compute the energy above the convex hull for a structure with the given total energy and composition.

        :param total_energy:
            Total energy in eV.
        :type total_energy: float
        :param composition:
            Pymatgen Composition object.
        :type composition: Composition

        :return:
            Energy above hull in eV/atom.
        :rtype: float
        """
        return get_energy_above_hull(total_energy, composition, hull_type=self._hull_type,
                                     threshold=self._hull_threshold)

    def compute(self, structures: Sequence[Structure], stage: ComputeStage,
                enable_progress_bar: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Compute energy above hull rewards for a batch of structures.

        First computes total energies using MACE, then computes energy above hull for each structure using its
        composition and the reference convex hull.

        The stage parameter is included for compatibility but is not used in this reward function.

        :param structures:
            Sequence of Structure objects representing generated structures.
        :type structures: Sequence[Structure]
        :param stage:
            Stage of the reward computation. Not used in this reward function.
        :type stage: ComputeStage
        :param enable_progress_bar:
            Whether to enable the progress bar for this computation.
        :type enable_progress_bar: bool

        :return:
            (Rewards per structure, info dictionary with 'e_above_hull', 'energy_per_atom', and 'ehull_invalid' keys).
        :rtype: tuple[np.ndarray, dict[str, np.ndarray]]
        """
        e_above_hull = np.empty(len(structures), dtype=float)
        energies_per_atom = np.empty(len(structures), dtype=float)
        invalid_flags = np.zeros(len(structures), dtype=float)

        # Step 1: Compute total energies using MACE (same as EnergyReward).
        total_energies = np.empty(len(structures), dtype=float)
        if self._device == "cpu":
            with torch.set_grad_enabled(True):  # MACE needs gradients.
                torch.set_default_dtype(torch.float64 if self._default_dtype == "float64" else torch.float32)
                for idx, structure in enumerate(tqdm.tqdm(structures, desc="Computing energies with MACE",
                                                          disable=not enable_progress_bar, unit="structures",
                                                          position=1)):
                    if self._invalid_penalty is not None and not self._is_valid(structure):
                        total_energies[idx] = np.nan
                        energies_per_atom[idx] = np.nan
                        e_above_hull[idx] = self._invalid_penalty
                        invalid_flags[idx] = 1.0
                        continue
                    total_energies[idx] = self._mace_model.get_potential_energy(structure.get_ase_atoms())
                    energies_per_atom[idx] = total_energies[idx] / len(structure.atomic_numbers)
                torch.set_default_dtype(torch.float32)
        else:
            assert self._device == "cuda"
            if self._invalid_penalty is not None:
                valid_mask = np.array([self._is_valid(structure) for structure in structures])
            else:
                valid_mask = np.ones(len(structures), dtype=bool)

            total_energies[~valid_mask] = np.nan
            energies_per_atom[~valid_mask] = np.nan
            e_above_hull[~valid_mask] = self._invalid_penalty
            invalid_flags[~valid_mask] = 1.0

            valid_atoms = [structures[i].get_ase_atoms() for i in range(len(structures)) if valid_mask[i]]
            res = static(system=valid_atoms, model=self._mace_model, autobatcher=self._batcher,
                         pbar={"desc": "Computing energies with MACE",
                               "postfix": f"{len(structures) - len(valid_atoms)} invalid structures",
                               "unit": "structures", "position": 1, "leave": False} if enable_progress_bar else False)
            assert len(res) == len(valid_atoms)
            valid_idx = 0
            for i in range(len(structures)):
                if valid_mask[i]:
                    total_energies[i] = float(res[valid_idx]["potential_energy"][0])
                    energies_per_atom[i] = total_energies[i] / len(structures[i].atomic_numbers)
                    valid_idx += 1

        # Step 2: Compute energy above hull for valid structures.
        for idx in tqdm.tqdm(range(len(structures)), desc="Computing energy above hull",
                             disable=not enable_progress_bar, unit="structures", position=1):
            if invalid_flags[idx] > 0.5:
                continue  # Already assigned penalty.
            try:
                composition = structures[idx].get_pymatgen_structure().composition
                e_above_hull[idx] = self._compute_energy_above_hull(total_energies[idx], composition)
            except ValueError as e:
                logging.warning(f"Failed to compute energy above hull for structure {idx}: {e}")
                if self._invalid_penalty is not None:
                    e_above_hull[idx] = self._invalid_penalty
                    invalid_flags[idx] = 1.0
                else:
                    # Fall back to a large positive value (very unstable).
                    e_above_hull[idx] = 10.0

        rewards = -self._scale * e_above_hull  # Minimize energy above hull.
        info_dict = {
            "e_above_hull": e_above_hull,
            "energy_per_atom": energies_per_atom,
            "ehull_invalid": invalid_flags,
        }
        return rewards, info_dict
