from typing import Optional, Sequence
import numpy as np
import tqdm
import warnings
import torch
from omg.datamodule import Structure
from omg.utils import prefixed_stdout
from .abstracts import ComputeStage, Reward


class EnergyReward(Reward):
    """
    Reward function that encourages lower energies per atom based on the MACE-MPA-0 model.

    This reward first identifies invalid structures based on volume, interatomic distances, and polar sine criteria.
    If specified, these invalid structures are assigned a penalty energy instead of their computed (likely diverging)
    energies. The energy per atom of valid structures is then calculated using the MACE-MPA-0 model. Finally, the
    energies within a GRPO group can be clipped based on a specified number of standard deviations from the mean.
    The reward for each structure is the negative of its (possibly penalized or clipped) energy per atom, scaled by the
    provided scaling factor.

    TODO: We should batch the MACE calculations for better performance.
    TODO: Change clipping to clipping within GRPO groups and compare results.

    :param scale:
        Scaling factor for the reward.
        Must be positive.
        Defaults to 1.0.
    :type scale: float
    :param device:
        The device to run MACE on. Can be "cpu" or "cuda".
        Defaults to "cpu".
    :type device: Literal["cpu", "cuda"]
    :param default_dtype:
        The default dtype to use for MACE calculations. Can be "float32" or "float64".
        Defaults to "float64".
    :type default_dtype: Literal["float32", "float64"]
    :param enable_cueq:
        Whether to enable the CuEq in MACE.
        Defaults to False.
    :type enable_cueq: bool
    :param invalid_penalty:
        Penalty energy in eV to assign to invalid structures.
        If None, no penalty is applied and energies for invalid structures are computed normally.
        Defaults to None.
    :type invalid_penalty: Optional[float]
    :param volume_check_cutoff:
        Minimum volume for which to compute the energy normally.
        If the generated structure has a volume below this cutoff, the energy is potentially penalized.
        Must be non-negative.
        Defaults to 0.1.
    :type volume_check_cutoff: float
    :param structure_check_cutoff:
        If the minimum interatomic distance in the generated structure is below this cutoff, the energy is potentially
        penalized.
        Must be non-negative.
        Defaults to 0.5.
    :type structure_check_cutoff: float
    :param polar_sine_cutoff:
        If the polar sine (volume / product of lattice lengths) of the generated structure is below this cutoff,
        the energy is potentially penalized.
        Must be non-negative.
        Defaults to 1.0e-3.
    :type polar_sine_cutoff: float
    :param clip_std:
        If specified, energies within a GRPO group are clipped to be within mean +/- clip_std
        standard deviations.
        Must be positive if specified.
        Defaults to None.
    :type clip_std: Optional[float]

    :raises ValueError:
        If scale is not positive.
        If volume_check_cutoff is negative.
        If structure_check_cutoff is negative.
        If polar_sine_cutoff is negative.
        If clip_std is specified and not positive.
    """

    def __init__(self, scale: float = 1.0, device: Literal["cpu", "cuda"] = "cpu",
                 default_dtype: Literal["float32", "float64"] = "float64", enable_cueq: bool = False,
                 invalid_penalty: Optional[float] = None, volume_check_cutoff: float = 0.1,
                 structure_check_cutoff: float = 0.5, polar_sine_cutoff: float = 1.0e-3,
                 clip_std: Optional[float] = None, ) -> None:
        """Constructor for EnergyReward."""
        super().__init__()
        if not scale > 0.0:
            raise ValueError("Scale must be positive.")
        if not volume_check_cutoff >= 0.0:
            raise ValueError("Volume check cutoff must be non-negative.")
        if not structure_check_cutoff >= 0.0:
            raise ValueError("Structure check cutoff must be non-negative.")
        if not polar_sine_cutoff >= 0.0:
            raise ValueError("Polar sine cutoff must be non-negative.")
        if clip_std is not None and not clip_std > 0.0:
            raise ValueError("clip_std must be positive.")
        self._scale = scale
        # Catch warnings from MACE and prefix stdout.
        with prefixed_stdout("[MACE] "), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            from mace.calculators import mace_mp
            self._mace_calculator = mace_mp(model="medium-mpa-0", device=device, default_dtype=default_dtype,
                                            enable_cueq=enable_cueq)
        self._invalid_penalty = invalid_penalty
        self._volume_check_cutoff = volume_check_cutoff
        self._structure_check_cutoff = structure_check_cutoff
        self._polar_sine_cutoff = polar_sine_cutoff
        self._clip_std = clip_std

    def compute(self, structures: Sequence[Structure], stage: ComputeStage,
                enable_progress_bar: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Compute energy rewards for a batch of structures.

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
            (List of rewards per structure, info dictionary).
        :rtype: tuple[np.ndarray, dict[str, np.ndarray]]
        """
        energies = np.empty(len(structures), dtype=float)
        invalid_flags = np.zeros(len(structures), dtype=float)

        with torch.set_grad_enabled(True):  # Mace needs gradients.
            for idx, structure in enumerate(tqdm.tqdm(structures, desc="Computing energy rewards",
                                                      disable=not enable_progress_bar)):
                if self._invalid_penalty is not None:
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
                    if not (volume_valid and structure_valid and polar_sine_valid):
                        energies[idx] = self._invalid_penalty
                        invalid_flags[idx] = 1.0
                        continue
                energies[idx] = (
                    self._mace_calculator.get_potential_energy(structure.get_ase_atoms())
                    / len(structure.atomic_numbers))

        raw_energies = energies.copy()
        if self._clip_std is not None:
            valid_mask = (invalid_flags == 0.0)
            valid_energies = energies[valid_mask]
            if valid_energies.size > 1:
                mean = valid_energies.mean()
                std = valid_energies.std()
                if std > 0.0:
                    low = mean - self._clip_std * std
                    high = mean + self._clip_std * std
                    energies[valid_mask] = np.clip(energies[valid_mask], low, high)

        rewards = -self._scale * energies  # Minimize energy.
        info_dict = {"energy_per_atom": energies, "energy_invalid": invalid_flags}
        if self._clip_std is not None:
            info_dict["energy_per_atom_raw"] = raw_energies
        return rewards, info_dict
