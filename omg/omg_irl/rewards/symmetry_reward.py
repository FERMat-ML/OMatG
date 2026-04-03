from contextlib import contextmanager
import os
from typing import Sequence
import numpy as np
import spglib
from omg.datamodule import Structure
from .abstracts import ComputeStage, Reward


@contextmanager
def _suppress_stderr():
    """Suppress stderr at the OS file-descriptor level (catches C library output)."""
    old_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)


class NonTriclinicReward(Reward):
    """
    Binary reward: 1.0 if the structure has space group number > 2 (non-triclinic), 0.0 otherwise.

    Non-differentiable reward computed via spglib. Invalid structures (degenerate lattices, unphysical
    bond lengths) receive a reward of 0.0.

    :param symprec:
        Symmetry tolerance for spglib in Angstroms.
        Defaults to 0.1.
    :type symprec: float
    :param volume_check_cutoff:
        Minimum volume (in A^3) for a structure to be considered valid.
        Defaults to 0.1.
    :type volume_check_cutoff: float
    :param structure_check_cutoff:
        Minimum interatomic distance (in A) for a structure to be considered valid.
        Defaults to 0.5.
    :type structure_check_cutoff: float
    :param polar_sine_cutoff:
        Minimum polar sine (volume / product of lattice lengths) for a structure to be considered valid.
        Defaults to 1.0e-3.
    :type polar_sine_cutoff: float
    """

    def __init__(self, symprec: float = 0.1, volume_check_cutoff: float = 0.1,
                 structure_check_cutoff: float = 0.5, polar_sine_cutoff: float = 1.0e-3) -> None:
        super().__init__()
        if not symprec > 0.0:
            raise ValueError("symprec must be positive.")
        if not volume_check_cutoff >= 0.0:
            raise ValueError("Volume check cutoff must be non-negative.")
        if not structure_check_cutoff >= 0.0:
            raise ValueError("Structure check cutoff must be non-negative.")
        if not polar_sine_cutoff >= 0.0:
            raise ValueError("Polar sine cutoff must be non-negative.")
        self._symprec = symprec
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

    def _get_spacegroup_number(self, structure: Structure) -> int:
        """Return the space group number for a structure, or 0 if spglib fails."""
        atoms = structure.get_ase_atoms()
        cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
        with _suppress_stderr():
            sym_data = spglib.get_symmetry_dataset(cell, symprec=self._symprec)
        if sym_data is None:
            return 0
        return sym_data.number

    def compute(self, structures: Sequence[Structure], stage: ComputeStage,
                enable_progress_bar: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Compute non-triclinic rewards for a batch of structures.

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
            (Rewards array, info dictionary with space group numbers and invalid flags).
        :rtype: tuple[np.ndarray, dict[str, np.ndarray]]
        """
        rewards = np.zeros(len(structures), dtype=float)

        for idx, structure in enumerate(structures):
            if not self._is_valid(structure):
                continue
            sg_number = self._get_spacegroup_number(structure)
            if sg_number > 2:
                rewards[idx] = 1.0

        info_dict = {
            "symmetry_non_triclinic": rewards.copy(),
        }
        return rewards, info_dict


class NonCentrosymmetricReward(Reward):
    """
    Binary reward: 1.0 if the structure is both non-triclinic (space group > 2) AND non-centrosymmetric
    (no inversion symmetry), 0.0 otherwise.

    Targets materials relevant for piezoelectricity, ferroelectricity, and nonlinear optics.
    Non-differentiable reward computed via spglib. Invalid structures receive a reward of 0.0.

    :param symprec:
        Symmetry tolerance for spglib in Angstroms.
        Defaults to 0.1.
    :type symprec: float
    :param volume_check_cutoff:
        Minimum volume (in A^3) for a structure to be considered valid.
        Defaults to 0.1.
    :type volume_check_cutoff: float
    :param structure_check_cutoff:
        Minimum interatomic distance (in A) for a structure to be considered valid.
        Defaults to 0.5.
    :type structure_check_cutoff: float
    :param polar_sine_cutoff:
        Minimum polar sine (volume / product of lattice lengths) for a structure to be considered valid.
        Defaults to 1.0e-3.
    :type polar_sine_cutoff: float
    """

    def __init__(self, symprec: float = 0.1, volume_check_cutoff: float = 0.1,
                 structure_check_cutoff: float = 0.5, polar_sine_cutoff: float = 1.0e-3) -> None:
        super().__init__()
        if not symprec > 0.0:
            raise ValueError("symprec must be positive.")
        if not volume_check_cutoff >= 0.0:
            raise ValueError("Volume check cutoff must be non-negative.")
        if not structure_check_cutoff >= 0.0:
            raise ValueError("Structure check cutoff must be non-negative.")
        if not polar_sine_cutoff >= 0.0:
            raise ValueError("Polar sine cutoff must be non-negative.")
        self._symprec = symprec
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

    def _compute_symmetry(self, structure: Structure) -> tuple[int, bool]:
        """Return (space group number, has_inversion) for a structure. Returns (0, False) if spglib fails."""
        atoms = structure.get_ase_atoms()
        cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
        with _suppress_stderr():
            sym_data = spglib.get_symmetry_dataset(cell, symprec=self._symprec)
        if sym_data is None:
            return 0, False
        has_inversion = any(
            np.array_equal(rot, -np.eye(3, dtype=int))
            for rot in sym_data.rotations
        )
        return sym_data.number, has_inversion

    def compute(self, structures: Sequence[Structure], stage: ComputeStage,
                enable_progress_bar: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Compute non-centrosymmetric rewards for a batch of structures.

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
            (Rewards array, info dictionary with space group numbers, inversion flags, and invalid flags).
        :rtype: tuple[np.ndarray, dict[str, np.ndarray]]
        """
        rewards = np.zeros(len(structures), dtype=float)
        is_not_triclinic = np.zeros(len(structures), dtype=float)
        has_inversion_flags = np.zeros(len(structures), dtype=float)

        for idx, structure in enumerate(structures):
            if not self._is_valid(structure):
                continue
            sg_number, has_inversion = self._compute_symmetry(structure)
            is_not_triclinic[idx] = float(sg_number > 2)
            has_inversion_flags[idx] = float(has_inversion)
            if sg_number > 2 and not has_inversion:
                rewards[idx] = 1.0

        info_dict = {
            "symmetry_non_triclinic": is_not_triclinic,
            "symmetry_has_inversion": has_inversion_flags,
            "symmetry_non_centrosymmetric": rewards.copy(),
        }
        return rewards, info_dict
