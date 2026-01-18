from functools import partial
import os
from typing import Optional, Sequence
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure as PymatgenStructure
import torch
import tqdm
from tqdm.contrib.concurrent import process_map
import warnings
from omg.datamodule import OMGDataset, Structure
from .abstracts import Reward


class VolumeReward(Reward):
    """
    Simple reward function that encourages larger unit cell volumes.
    """

    def __init__(self) -> None:
        """Constructor for VolumeReward."""
        super().__init__()

    def compute(self, structures: Sequence[Structure], stage: Reward.ComputeStage,
                enable_progress_bar: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Compute rewards for a batch of structures.

        This reward is simply the volume of each structure's unit cell. The stage parameter is included
        for compatibility but is not used in this reward function.

        :param structures:
            Sequence of Structure objects representing generated structures.
        :type structures: Sequence[Structure]
        :param stage:
            Stage of the reward computation. Not used in this reward function.
        :type stage: Reward.ComputeStage
        :param enable_progress_bar:
            Whether to enable the progress bar for this computation.
        :type enable_progress_bar: bool

        :return:
            (List of rewards per structure, info dictionary).
        :rtype: tuple[np.ndarray, dict[str, np.ndarray]]
        """
        volumes = np.array([structure.get_ase_atoms().get_volume() for structure in structures])
        return volumes, {"volume": volumes}


class ValidityPenaltyReward(Reward):
    """
    Reward function that penalizes invalid structures based on simple geometric checks.

    A structure is considered invalid if any of the following hold:
      - Cell volume is below volume_check_cutoff
      - Minimum interatomic distance is below structure_check_cutoff
      - Polar sine is below polar_sine_cutoff (near-collinear cell)

    The reward is -scale for invalid structures and 0 for valid ones.
    """

    def __init__(self, scale: float = 1.0, volume_check_cutoff: float = 0.1,
                 structure_check_cutoff: float = 0.5, polar_sine_cutoff: float = 1.0e-3) -> None:
        """Constructor for ValidityPenaltyReward."""
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
        self._volume_check_cutoff = volume_check_cutoff
        self._structure_check_cutoff = structure_check_cutoff
        self._polar_sine_cutoff = polar_sine_cutoff

    def compute(self, structures: Sequence[Structure], stage: Reward.ComputeStage,
                enable_progress_bar: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Compute validity penalties for a batch of structures.

        :param structures:
            Sequence of Structure objects representing generated structures.
        :type structures: Sequence[Structure]
        :param stage:
            Stage of the reward computation. Not used in this reward function.
        :type stage: Reward.ComputeStage
        :param enable_progress_bar:
            Whether to enable the progress bar for this computation.
        :type enable_progress_bar: bool

        :return:
            (List of rewards per structure, info dictionary).
        :rtype: tuple[np.ndarray, dict[str, np.ndarray]]
        """
        invalid_flags = []
        invalid_volume = []
        invalid_min_dist = []
        invalid_polar_sine = []

        for structure in tqdm.tqdm(structures, desc="Computing validity penalties",
                                   disable=not enable_progress_bar):
            py_structure = structure.get_pymatgen_structure()

            volume_valid = (py_structure.volume >= self._volume_check_cutoff)
            dist_mat = py_structure.distance_matrix
            # Pad diagonal with a large number to remove self-matches.
            dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (self._structure_check_cutoff + 10.0))
            min_dist = dist_mat.min()
            structure_valid = (min_dist >= self._structure_check_cutoff)

            polar_sine = py_structure.lattice.volume / np.prod(py_structure.lattice.lengths)
            polar_sine_valid = (polar_sine >= self._polar_sine_cutoff)

            invalid = not (volume_valid and structure_valid and polar_sine_valid)
            invalid_flags.append(1.0 if invalid else 0.0)
            invalid_volume.append(0.0 if volume_valid else 1.0)
            invalid_min_dist.append(0.0 if structure_valid else 1.0)
            invalid_polar_sine.append(0.0 if polar_sine_valid else 1.0)

        invalid_flags = np.array(invalid_flags, dtype=float)
        rewards = -self._scale * invalid_flags
        info_dict = {
            "invalid_structure": invalid_flags,
            "invalid_volume": np.array(invalid_volume, dtype=float),
            "invalid_min_dist": np.array(invalid_min_dist, dtype=float),
            "invalid_polar_sine": np.array(invalid_polar_sine, dtype=float),
        }
        return rewards, info_dict


class CRMSEReward(Reward):
    """
    Reward function that reduces the corrected root-mean-square error (cRMSE) between generated structures and known
    stable structures in a reference dataset.

    During the reward computation, each generated structure is compared to all structures in the reference dataset with
    the same reduced composition using Pymatgen's StructureMatcher. The cRMSE is calculated as the minimum RMSD
    among all matched structures. If no match is found within the specified tolerances, a penalty equal to the structure
    tolerance (stol) is assigned. The RMSD is computed using Pymatgen's get_rms_dist method and is normalized by
    (Vol / nsites) ** (1/3).  Hence, the cRMSE is similarly normalized.

    The training and validation dataset structures are indexed by their reduced composition for efficient lookup. The
    indices are built once when set_datasets is called.

    :param ltol:
        Fractional length tolerance for Pymatgen's StructureMatcher.
        Defaults to 0.3 (Pymatgen's default is 0.2).
    :type ltol: float
    :param stol:
        Site tolerance for Pymatgen's StructureMatcher.
        Defaults to 0.5 (Pymatgen's default is 0.3).
    :type stol: float
    :param angle_tol:
        Angle tolerance in degrees for Pymatgen's StructureMatcher.
        Defaults to 10.0 (Pymatgen's default is 10.0).
    :type angle_tol: float
    :param volume_check_cutoff:
        Minimum volume for which to compute the RMSD normally.
        If the generated structure has a volume below this cutoff, the RMSD is set to stol directly.
        This avoids issues with very small volumes leading to Pymatgen's StructureMatcher hanging indefinitely.
        Must be non-negative.
        Defaults to 0.1.
    :type volume_check_cutoff: float
    :param structure_check_cutoff:
        If the minimum interatomic distance in the generated structure is below this cutoff, the RMSD is set to stol
        directly.
        This avoids issues with unphysical structures causing problems in Pymatgen's StructureMatcher.
        Must be non-negative.
        Defaults to 0.5.
    :type structure_check_cutoff: float
    :param polar_sine_cutoff:
        If the polar sine (volume / product of lattice lengths) of the generated structure is below this cutoff,
        the RMSD is set to stol directly. This avoids issues with collinear or near-collinear structures causing
        problems in Pymatgen's StructureMatcher.
        Must be non-negative.
        Defaults to 1.0e-3.
    :type polar_sine_cutoff: float
    :param scale:
        Scaling factor for the reward.
        Must be positive.
        Defaults to 1.0.
    :type scale: float
    :param number_cpus:
        Number of CPUs to use for parallelization. If None, use os.cpu_count().
        Defaults to None.
    :type number_cpus: Optional[int]
    :raises ValueError:
        If scale is not positive.
    """
    def __init__(self, ltol: float = 0.3, stol: float = 0.5, angle_tol: float = 10.0, scale: float = 1.0,
                 volume_check_cutoff: float = 0.1, structure_check_cutoff: float = 0.5,
                 polar_sine_cutoff: float = 1.0e-3, number_cpus: Optional[int] = None) -> None:
        """Constructor for CRMSEReward."""
        super().__init__()
        self._ltol = ltol
        self._stol = stol
        self._angle_tol = angle_tol
        if not scale > 0.0:
            raise ValueError("Scale must be positive.")
        self._scale = scale
        if not volume_check_cutoff >= 0.0:
            raise ValueError("Volume check cutoff must be non-negative.")
        self._volume_check_cutoff = volume_check_cutoff
        if not structure_check_cutoff >= 0.0:
            raise ValueError("Structure check cutoff must be non-negative.")
        self._structure_check_cutoff = structure_check_cutoff
        if not polar_sine_cutoff >= 0.0:
            raise ValueError("Polar sine cutoff must be non-negative.")
        self._polar_sine_cutoff = polar_sine_cutoff
        if number_cpus is not None and number_cpus < 1:
            raise ValueError("The number of CPUs must be at least 1.")
        self._cpu_count = number_cpus if number_cpus is not None else os.cpu_count()

        self._train_reduced_composition_map = None
        self._val_reduced_composition_map = None
        self._pred_reduced_composition_map = None

    def set_train_dataset(self, train_dataset: OMGDataset) -> None:
        """
        Set the training dataset for reward computation.

        This method is called in the OMGTFLightning class before compute is called. It builds a dictionary that maps
        reduced compositions to lists of Pymatgen Structures with that composition in the training dataset for
        efficient lookup during reward computation.

        # TODO: This is not optimal for large datasets.

        :param train_dataset:
            Training dataset.
        :type train_dataset: OMGDataset
        """
        self._train_reduced_composition_map = self._build_reduced_composition_map(
            train_dataset, desc="Building reduced composition map for training dataset")

    def set_val_dataset(self, val_dataset: OMGDataset) -> None:
        """
        Set the validation dataset for reward computation.

        This method is called in the OMGTFLightning class before compute is called. It builds a dictionary that maps
        reduced compositions to lists of Pymatgen Structures with that composition in the validation dataset for
        efficient lookup during reward computation.

        :param val_dataset:
            Validation dataset.
        :type val_dataset: OMGDataset
        """
        self._val_reduced_composition_map = self._build_reduced_composition_map(
            val_dataset, desc="Building reduced composition map for validation dataset")

    def set_pred_dataset(self, pred_dataset: OMGDataset) -> None:
        """
        Set the prediction dataset for reward computation.

        This method is called in the OMGTFLightning class before compute is called. It builds a dictionary that maps
        reduced compositions to lists of Pymatgen Structures with that composition in the prediction dataset for
        efficient lookup during reward computation.

        :param pred_dataset:
            Prediction dataset.
        :type pred_dataset: OMGDataset
        """
        self._pred_reduced_composition_map = self._build_reduced_composition_map(
            pred_dataset, desc="Building reduced composition map for prediction dataset")

    @staticmethod
    def _build_reduced_composition_map(dataset: OMGDataset,
                                       desc: Optional[str] = None) -> dict[tuple[int, ...], list[PymatgenStructure]]:
        """
        Build a reduced composition map for the given dataset.

        The returned map allows efficient lookup of structures of the given dataset by their reduced composition during
        reward computation.

        :param dataset:
            Dataset for which to build the reduced composition map.
        :type dataset: OMGDataset
        :param desc:
            Optional description for the progress bar.
        :type desc: Optional[str]

        :return:
            Dictionary mapping reduced composition keys to lists of structures.
        :rtype: Dict[tuple[int, ...], List[PymatgenStructure]]
        """
        reduced_composition_map = {}
        # noinspection PyTypeChecker
        for structure in tqdm.tqdm(dataset.get_structure_dataset(), desc=desc, total=len(dataset)):
            py_structure = structure.get_pymatgen_structure()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                reduced_composition = py_structure.composition.reduced_composition
            if reduced_composition not in reduced_composition_map:
                reduced_composition_map[reduced_composition] = []
            reduced_composition_map[reduced_composition].append(py_structure)
        return reduced_composition_map

    @staticmethod
    def _compute_rmse(py_structure: PymatgenStructure, reference_py_structures: Sequence[PymatgenStructure],
                      ltol: float, stol: float, angle_tol: float, volume_check_cutoff: float,
                      structure_check_cutoff: float, polar_sine_cutoff: float) -> float:
        """
        Compute the cRMSE between a generated structure and the closest matching structure of the reference structures.

        This method assumes that all reference structures have the same reduced composition as the generated structure.

        :param py_structure:
            Generated structure as a Pymatgen Structure.
        :type py_structure: PymatgenStructure
        :param reference_py_structures:
            Sequence of reference structures as Pymatgen Structures with the same reduced composition.
        :type reference_py_structures: Sequence[PymatgenStructure]
        :param ltol:
            Fractional length tolerance for Pymatgen's StructureMatcher.
        :type ltol: float
        :param stol:
            Site tolerance for Pymatgen's StructureMatcher.
        :type stol: float
        :param angle_tol:
            Angle tolerance in degrees for Pymatgen's StructureMatcher.
        :type angle_tol: float
        :param volume_check_cutoff:
            Minimum volume for which to compute the RMSD normally.
            If the generated structure has a volume below this cutoff, the RMSD is set to stol directly.
            This avoids issues with very small volumes leading to Pymatgen's StructureMatcher hanging indefinitely.
        :type volume_check_cutoff: float
        :param structure_check_cutoff:
            If the minimum interatomic distance in the generated structure is below this cutoff, the RMSD is set to stol
            directly.
            This avoids issues with unphysical structures causing problems in Pymatgen's StructureMatcher.
        :type structure_check_cutoff: float
        :param polar_sine_cutoff:
            If the polar sine (volume / product of lattice lengths) of the generated structure is below this cutoff,
            the RMSD is set to stol directly. This avoids issues with collinear or near-collinear structures causing
            problems in Pymatgen's StructureMatcher.
        :type polar_sine_cutoff: float

        :return:
            The cRMSE value.
        :rtype: float
        """
        sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
        # Match structures and take smallest RMSE.
        # Use stol for non-matching structures.
        assert len(reference_py_structures) > 0
        rmses = []
        for ref_py_structure in reference_py_structures:
            volume_valid = (py_structure.volume >= volume_check_cutoff)

            dist_mat = py_structure.distance_matrix
            # Pad diagonal with a large number to remove self-matches.
            dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (structure_check_cutoff + 10.0))
            structure_valid = (dist_mat.min() >= structure_check_cutoff)

            # Polar sine can be measured to detect collinearity.
            polar_sine = py_structure.lattice.volume / np.prod(py_structure.lattice.lengths)
            polar_sine_valid = (polar_sine >= polar_sine_cutoff)

            if volume_valid and structure_valid and polar_sine_valid:
                res = sm.get_rms_dist(py_structure, ref_py_structure)  # res is None if no match is found.
            else:
                res = None

            assert res is None or res[0] <= stol
            rmses.append(stol if res is None else res[0])

        return min(rmses)

    def compute(self, structures: Sequence[Structure], stage: Reward.ComputeStage,
                enable_progress_bar: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Compute rewards for a batch of structures.

        The reward is computed as scale * (stol - cRMSE) for each structure, where cRMSE is the corrected
        root-mean-square error between the generated structure and the closest matching structure in the relevant
        reference dataset with the same reduced composition. If no match is found, cRMSE is set to stol. Thus, higher
        rewards correspond to lower cRMSE values.

        :param structures:
            Sequence of Structure objects representing generated structures.
        :type structures: Sequence[Structure]
        :param stage:
            Stage of the reward computation.
        :type stage: Reward.ComputeStage
        :param enable_progress_bar:
            Whether to enable the progress bar for this computation.
        :type enable_progress_bar: bool

        :return:
            (List of rewards per structure, info dictionary).
        :rtype: tuple[np.ndarray, dict[str, np.ndarray]]
        """
        if stage == Reward.ComputeStage.TRAIN:
            reduced_composition_map = self._train_reduced_composition_map
            if reduced_composition_map is None:
                raise RuntimeError("Training dataset not set. Call set_train_dataset before compute.")
        elif stage == Reward.ComputeStage.VAL:
            reduced_composition_map = self._val_reduced_composition_map
            if reduced_composition_map is None:
                raise RuntimeError("Validation dataset not set. Call set_val_dataset before compute.")
        else:
            assert stage == Reward.ComputeStage.PRED
            reduced_composition_map = self._pred_reduced_composition_map
            if reduced_composition_map is None:
                raise RuntimeError("Prediction dataset not set. Call set_pred_dataset before compute.")

        # Be careful to convert structures to pymatgen structures before parallel processing to avoid pickling issues
        # with torch tensors.
        py_structures = []
        relevant_structures_list = []
        for structure in structures:
            py_structure = structure.get_pymatgen_structure()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                reduced_composition = py_structure.composition.reduced_composition
            py_structures.append(py_structure)
            assert (reduced_composition in reduced_composition_map
                    and len(reduced_composition_map[reduced_composition]) > 0)
            relevant_structures_list.append(reduced_composition_map[reduced_composition])

        crmse_function = partial(self._compute_rmse, ltol=self._ltol, stol=self._stol, angle_tol=self._angle_tol,
                                 volume_check_cutoff=self._volume_check_cutoff,
                                 structure_check_cutoff=self._structure_check_cutoff,
                                 polar_sine_cutoff=self._polar_sine_cutoff)

        if self._cpu_count > 1:
            crmse_values = process_map(crmse_function, py_structures, relevant_structures_list,
                                       desc="Computing cRMSE rewards",
                                       chunksize=max(min(len(py_structures) // self._cpu_count, 100), 1),
                                       max_workers=self._cpu_count, position=1, leave=False,
                                       disable=not enable_progress_bar)
        else:
            crmse_values = list(map(crmse_function,
                                    tqdm.tqdm(py_structures, desc="Computing cRMSE rewards", total=len(py_structures),
                                              position=1, leave=False, disable=not enable_progress_bar),
                                    relevant_structures_list))

        return self._scale * (self._stol - np.array(crmse_values)), {"cRMSE": np.array(crmse_values)}


class ProbabilityMatchingCRMSEReward(CRMSEReward):
    """
    Reward function that combines cRMSE matching with probability matching across GRPO groups.

    This reward extends CRMSEReward by adding a per-structure adjustment that encourages the empirical distribution
    of matched polymorphs within a GRPO group to match the reference distribution in the dataset.

    For each GRPO group:
    1. Each generated structure is matched to reference structures with the same composition.
    2. The best matching reference structure (lowest RMSE) is identified for each generated structure.
    3. The empirical distribution of matched polymorphs within the group is computed.
    4. Each structure's reward is adjusted based on whether its matched polymorph is over- or under-represented
       relative to the reference distribution.

    The adjustment formula for a structure matching polymorph i is:
        adjustment = adjustment_scale * (reference_prob[i] - empirical_prob[i])

    This gives positive adjustments to structures matching under-represented polymorphs and negative adjustments
    to structures matching over-represented polymorphs.

    :param grpo_group_size:
        Number of structures per GRPO group.
    :type grpo_group_size: int
    :param adjustment_scale:
        Scaling factor for the probability matching adjustment.
        Defaults to 1.0.
    :type adjustment_scale: float
    :param ltol:
        Fractional length tolerance for Pymatgen's StructureMatcher.
        Defaults to 0.3.
    :type ltol: float
    :param stol:
        Site tolerance for Pymatgen's StructureMatcher.
        Defaults to 0.5.
    :type stol: float
    :param angle_tol:
        Angle tolerance in degrees for Pymatgen's StructureMatcher.
        Defaults to 10.0.
    :type angle_tol: float
    :param volume_check_cutoff:
        Minimum volume for which to compute the RMSD normally.
        Defaults to 0.1.
    :type volume_check_cutoff: float
    :param structure_check_cutoff:
        Minimum interatomic distance cutoff.
        Defaults to 0.5.
    :type structure_check_cutoff: float
    :param polar_sine_cutoff:
        Minimum polar sine cutoff.
        Defaults to 1.0e-3.
    :type polar_sine_cutoff: float
    :param scale:
        Scaling factor for the base cRMSE reward.
        Defaults to 1.0.
    :type scale: float
    :param number_cpus:
        Number of CPUs to use for parallelization. If None, use os.cpu_count().
        Defaults to None.
    :type number_cpus: Optional[int]
    """

    def __init__(self, grpo_group_size: int, adjustment_scale: float = 1.0, ltol: float = 0.3, stol: float = 0.5,
                 angle_tol: float = 10.0, scale: float = 1.0, volume_check_cutoff: float = 0.1,
                 structure_check_cutoff: float = 0.5, polar_sine_cutoff: float = 1.0e-3,
                 number_cpus: Optional[int] = None) -> None:
        """Constructor for ProbabilityMatchingCRMSEReward."""
        super().__init__(ltol=ltol, stol=stol, angle_tol=angle_tol, scale=scale,
                         volume_check_cutoff=volume_check_cutoff, structure_check_cutoff=structure_check_cutoff,
                         polar_sine_cutoff=polar_sine_cutoff, number_cpus=number_cpus)
        if not grpo_group_size > 1:
            raise ValueError("GRPO group size must be greater than 1.")
        self._grpo_group_size = grpo_group_size
        self._adjustment_scale = adjustment_scale

    @staticmethod
    def _compute_rmse_with_match_index(py_structure: PymatgenStructure,
                                       reference_py_structures: Sequence[PymatgenStructure],
                                       ltol: float, stol: float, angle_tol: float, volume_check_cutoff: float,
                                       structure_check_cutoff: float, polar_sine_cutoff: float) -> tuple[float, int]:
        """
        Compute the cRMSE and return the index of the best matching reference structure.

        :param py_structure:
            Generated structure as a Pymatgen Structure.
        :type py_structure: PymatgenStructure
        :param reference_py_structures:
            Sequence of reference structures with the same reduced composition.
        :type reference_py_structures: Sequence[PymatgenStructure]
        :param ltol:
            Fractional length tolerance.
        :type ltol: float
        :param stol:
            Site tolerance.
        :type stol: float
        :param angle_tol:
            Angle tolerance in degrees.
        :type angle_tol: float
        :param volume_check_cutoff:
            Minimum volume cutoff.
        :type volume_check_cutoff: float
        :param structure_check_cutoff:
            Minimum interatomic distance cutoff.
        :type structure_check_cutoff: float
        :param polar_sine_cutoff:
            Minimum polar sine cutoff.
        :type polar_sine_cutoff: float

        :return:
            (cRMSE value, index of best matching reference structure or -1 if no match).
        :rtype: tuple[float, int]
        """
        sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
        # Match structures and take smallest RMSE.
        # Use stol for non-matching structures.
        assert len(reference_py_structures) > 0
        rmses = []
        for ref_index, ref_py_structure in enumerate(reference_py_structures):
            volume_valid = (py_structure.volume >= volume_check_cutoff)

            dist_mat = py_structure.distance_matrix
            # Pad diagonal with a large number to remove self-matches.
            dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (structure_check_cutoff + 10.0))
            structure_valid = (dist_mat.min() >= structure_check_cutoff)

            polar_sine = py_structure.lattice.volume / np.prod(py_structure.lattice.lengths)
            polar_sine_valid = (polar_sine >= polar_sine_cutoff)

            if volume_valid and structure_valid and polar_sine_valid:
                res = sm.get_rms_dist(py_structure, ref_py_structure)
            else:
                res = None

            assert res is None or res[0] <= stol
            # Use -1 as match index if no match found.
            rmses.append((stol, -1) if res is None else (res[0], ref_index))

        return min(rmses)

    def compute(self, structures: Sequence[Structure], stage: Reward.ComputeStage,
                enable_progress_bar: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Compute rewards with probability matching adjustment for GRPO groups.

        The reward is computed as:
            reward = scale * (stol - cRMSE) + adjustment_scale * (reference_prob - empirical_prob)

        where reference_prob and empirical_prob refer to the matched polymorph within each GRPO group.

        :param structures:
            Sequence of Structure objects representing generated structures.
        :type structures: Sequence[Structure]
        :param stage:
            Stage of the reward computation.
        :type stage: Reward.ComputeStage
        :param enable_progress_bar:
            Whether to enable the progress bar for this computation.
        :type enable_progress_bar: bool

        :return:
            (List of rewards per structure, info dictionary).
        :rtype: tuple[np.ndarray, dict[str, np.ndarray]]
        """
        if stage == Reward.ComputeStage.TRAIN:
            reduced_composition_map = self._train_reduced_composition_map
            if reduced_composition_map is None:
                raise RuntimeError("Training dataset not set. Call set_train_dataset before compute.")
        elif stage == Reward.ComputeStage.VAL:
            reduced_composition_map = self._val_reduced_composition_map
            if reduced_composition_map is None:
                raise RuntimeError("Validation dataset not set. Call set_val_dataset before compute.")
        else:
            assert stage == Reward.ComputeStage.PRED
            reduced_composition_map = self._pred_reduced_composition_map
            if reduced_composition_map is None:
                raise RuntimeError("Prediction dataset not set. Call set_pred_dataset before compute.")

        # Verify batch size is compatible with GRPO group size.
        if len(structures) % self._grpo_group_size != 0:
            raise ValueError(f"Number of structures ({len(structures)}) must be divisible by "
                             f"GRPO group size ({self._grpo_group_size}).")

        # Check matching compositions.
        for grpo_group_index in range(len(structures) // self._grpo_group_size):
            start_idx = grpo_group_index * self._grpo_group_size
            end_idx = (grpo_group_index + 1) * self._grpo_group_size
            group_slice = slice(start_idx, end_idx)
            group_structures = structures[group_slice]
            # Atomic numbers should match exactly
            first_numbers = group_structures[0].atomic_numbers
            for structure in group_structures[1:]:
                assert structure.atomic_numbers == first_numbers

        # Convert structures to pymatgen structures.
        py_structures = []
        relevant_structures_list = []
        for structure in structures:
            py_structure = structure.get_pymatgen_structure()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                reduced_composition = py_structure.composition.reduced_composition
            py_structures.append(py_structure)
            assert (reduced_composition in reduced_composition_map
                    and len(reduced_composition_map[reduced_composition]) > 0)
            relevant_structures_list.append(reduced_composition_map[reduced_composition])

        # Compute cRMSE and match indices.
        crmse_function = partial(self._compute_rmse_with_match_index, ltol=self._ltol, stol=self._stol,
                                 angle_tol=self._angle_tol, volume_check_cutoff=self._volume_check_cutoff,
                                 structure_check_cutoff=self._structure_check_cutoff,
                                 polar_sine_cutoff=self._polar_sine_cutoff)

        if self._cpu_count > 1:
            results = process_map(crmse_function, py_structures, relevant_structures_list,
                                  desc="Computing cRMSE rewards",
                                  chunksize=max(min(len(py_structures) // self._cpu_count, 100), 1),
                                  max_workers=self._cpu_count, position=1, leave=False,
                                  disable=not enable_progress_bar)
        else:
            results = list(map(crmse_function,
                               tqdm.tqdm(py_structures, desc="Computing cRMSE rewards",
                                         total=len(py_structures), position=1, leave=False,
                                         disable=not enable_progress_bar),
                               relevant_structures_list))

        crmse_values = np.array([r[0] for r in results])
        match_indices = [r[1] for r in results]

        # Compute base cRMSE rewards.
        base_rewards = self._scale * (self._stol - crmse_values)

        # Compute probability matching adjustments per GRPO group.
        adjustments = np.zeros(len(structures))
        coverage_rates = np.zeros(len(structures) // self._grpo_group_size)
        num_groups = len(structures) // self._grpo_group_size

        for group_idx in range(num_groups):
            start_idx = group_idx * self._grpo_group_size
            end_idx = (group_idx + 1) * self._grpo_group_size
            group_slice = slice(start_idx, end_idx)

            group_match_indices = match_indices[group_slice]
            group_relevant_structures = relevant_structures_list[start_idx]
            # Same for all in group.
            assert all(group_relevant_structures is relevant_structures_list[i] for i in range(start_idx, end_idx))
            num_polymorphs = len(group_relevant_structures)

            # Reference distribution: uniform over polymorphs (each polymorph counted once in reference).
            reference_probs = np.ones(num_polymorphs) / num_polymorphs

            # Empirical distribution: count how many times each polymorph was matched.
            empirical_counts = np.zeros(num_polymorphs)
            unmatched_count = 0
            for idx in group_match_indices:
                if idx >= 0:
                    empirical_counts[idx] += 1
                else:
                    unmatched_count += 1

            # Coverage rate: fraction of unique polymorphs matched by at least one structure in the group.
            polymorphs_covered = np.sum(empirical_counts > 0)
            coverage_rate = polymorphs_covered / num_polymorphs
            coverage_rates[group_idx] = coverage_rate

            # Normalize empirical counts (only over matched structures).
            matched_count = self._grpo_group_size - unmatched_count
            if matched_count > 0:
                empirical_probs = empirical_counts / matched_count
            else:
                # No matches at all - no adjustment possible.
                empirical_probs = np.zeros(num_polymorphs)

            # Compute per-structure adjustments.
            # Normalize by maximum possible deviation (1 - 1/N) so that adjustments are in [-1, 1] * adjustment_scale
            # regardless of the number of polymorphs.
            max_deviation = 1.0 - 1.0 / num_polymorphs
            for i, idx in enumerate(group_match_indices):
                struct_idx = start_idx + i
                if idx >= 0 and matched_count > 0 and max_deviation > 0:
                    # Positive adjustment if under-represented, negative if over-represented.
                    adjustments[struct_idx] = (self._adjustment_scale
                                               * (reference_probs[idx] - empirical_probs[idx]) / max_deviation)
                else:
                    # No match or only one polymorph - no adjustment.
                    adjustments[struct_idx] = 0.0

        total_rewards = base_rewards + adjustments

        return total_rewards, {
            "cRMSE": crmse_values,
            "prob_adjustment": adjustments,
            "match_rate": np.array([1.0 if idx >= 0 else 0.0 for idx in match_indices]),
            "coverage_rate": coverage_rates
        }


class EnergyReward(Reward):
    """
    Reward function that encourages lower potential energy per atom.

    :param scale:
        Scaling factor for the reward.
        Must be positive.
        Defaults to 1.0.
    :type scale: float
    :raises ValueError:
        If scale is not positive.
    """

    def __init__(self, scale: float = 1.0, device: Optional[str] = "cpu") -> None:
        """Constructor for EnergyReward.

        CPU device seems to be quicker for small batches (tested up to 1024 structures).
        """
        super().__init__()
        if not scale > 0.0:
            raise ValueError("Scale must be positive.")
        from mace.calculators import mace_mp
        self._scale = scale
        self._mace_calculator = mace_mp(model="medium-mpa-0", device=device)

    def compute(self, structures: Sequence[Structure], stage: Reward.ComputeStage,
                enable_progress_bar: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Compute rewards for a batch of structures.

        This reward is simply the negative potential energy per atom of each structure, scaled by the specified factor.
        The stage parameter is included for compatibility but is not used in this reward function.

        :param structures:
            Sequence of Structure objects representing generated structures.
        :type structures: Sequence[Structure]
        :param stage:
            Stage of the reward computation. Not used in this reward function.
        :type stage: Reward.ComputeStage
        :param enable_progress_bar:
            Whether to enable the progress bar for this computation.
        :type enable_progress_bar: bool

        :return:
            (List of rewards per structure, info dictionary).
        :rtype: tuple[np.ndarray, dict[str, np.ndarray]]
        """
        with torch.set_grad_enabled(True):  # Mace needs gradients.
            energies = np.array([self._mace_calculator.get_potential_energy(structure.get_ase_atoms()) / len(structure.atomic_numbers)
                                 for structure in tqdm.tqdm(structures, desc="Computing energy rewards",
                                                            disable=not enable_progress_bar)])
        rewards = -self._scale * energies  # Minimize energy.
        for structure in structures:
            print(structure.a)
        return rewards, {"energy_per_atom": energies}


class CompositeRewards(Reward):
    """
    Composite reward that combines multiple reward functions.

    :param rewards: List of reward functions to combine
    :type rewards: List[RewardFunction]
    :param weights: Weights for each reward function (must sum to 1)
    :type weights: List[float]
    """

    def __init__(self, rewards: Sequence[Reward], weights: Sequence[float]) -> None:
        """Constructor for CompositeReward."""
        super().__init__()
        if len(rewards) != len(weights):
            raise ValueError("Number of reward functions must match number of weights.")
        if not all(w > 0.0 for w in weights):
            raise ValueError("Weights must be positive.")
        if not abs(sum(weights) - 1.0) < 1e-6:
            raise ValueError("Weights must sum to 1.0")
        self._reward_functions = rewards
        self._weights = weights

    def set_train_dataset(self, train_dataset: OMGDataset) -> None:
        """
        Set the training dataset for all reward functions.

        :param train_dataset:
            Training dataset.
        :type train_dataset: OMGDataset
        """
        for reward_function in self._reward_functions:
            reward_function.set_train_dataset(train_dataset)

    def set_val_dataset(self, val_dataset: OMGDataset) -> None:
        """
        Set the validation dataset for all reward functions.

        :param val_dataset:
            Validation dataset.
        :type val_dataset: OMGDataset
        """
        for reward_function in self._reward_functions:
            reward_function.set_val_dataset(val_dataset)

    def set_pred_dataset(self, pred_dataset: OMGDataset) -> None:
        """
        Set the prediction dataset for all reward functions.

        :param pred_dataset:
            Prediction dataset.
        :type pred_dataset: OMGDataset
        """
        for reward_function in self._reward_functions:
            reward_function.set_pred_dataset(pred_dataset)

    def compute(self, structures: Sequence[Structure], stage: Reward.ComputeStage,
                enable_progress_bar: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Compute rewards for a batch of structures.

        Some reward functions may require access to a reference dataset for computing rewards (e.g., to compute
        similarity to known stable structures). The stage parameter indicates whether the reward is being computed
        for training or validation, allowing the reward function to use the appropriate precomputed data from the
        set_datasets method.

        :param structures:
            Sequence of Structure objects representing generated structures.
        :type structures: Sequence[Structure]
        :param stage:
            Stage of the reward computation.
        :type stage: Reward.ComputeStage
        :param enable_progress_bar:
            Whether to enable the progress bar for this computation.
        :type enable_progress_bar: bool

        :return:
            (List of rewards per structure, info dictionary).
        :rtype: tuple[np.ndarray, dict[str, np.ndarray]]
        """
        total_rewards = np.zeros(len(structures))
        total_dict = {}
        for reward_function, weight in zip(self._reward_functions, self._weights):
            rewards, info_dict = reward_function.compute(structures, stage, enable_progress_bar)
            total_rewards += weight * rewards
            for key, value in info_dict.items():
                assert key not in total_dict
                total_dict[key] = value
        return total_rewards, total_dict
