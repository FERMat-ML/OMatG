from functools import partial
import os
from typing import Optional, Sequence
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure as PymatgenStructure
import tqdm
from tqdm.contrib.concurrent import process_map
from omg.datamodule import OMGDataset, Structure
from omg.globals import MAX_ATOM_NUM
from .abstracts import Reward


class VolumeReward(Reward):
    """
    Simple reward function that encourages larger unit cell volumes.
    """

    def __init__(self) -> None:
        """Constructor for VolumeReward."""
        super().__init__()

    def compute(self, structures: Sequence[Structure],
                stage: Reward.ComputeStage) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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

        :return:
            (List of rewards per structure, info dictionary).
        :rtype: tuple[np.ndarray, dict[str, np.ndarray]]
        """
        volumes = np.array([structure.get_ase_atoms().get_volume() for structure in structures])
        return volumes, {"volume": volumes}


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
                 number_cpus: Optional[int] = None) -> None:
        """Constructor for CRMSEReward."""
        super().__init__()
        self._ltol = ltol
        self._stol = stol
        self._angle_tol = angle_tol
        if not scale > 0.0:
            raise ValueError("Scale must be positive.")
        self._scale = scale
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
    def _get_reduced_composition_key(atomic_numbers: np.ndarray) -> tuple[int, ...]:
        """
        Compute a hashable key representing the reduced composition from atomic numbers.

        This method uses the same approach as omg.analysis.analysis._element_check. It counts occurrences of each
        element and divides by the minimum count to obtain a reduced count of elements. It the returns a tuple of the
        reduced counts.

        :param atomic_numbers:
            Array of atomic numbers for the structure.
        :type atomic_numbers: np.ndarray

        :return:
            Hashable tuple of length MAX_ATOM_NUM representing the reduced composition.
        :rtype: tuple[int, ...]
        """
        counts = np.bincount(atomic_numbers, minlength=MAX_ATOM_NUM)
        # Find the element with the minimum number of occurrences.
        min_count = np.min(counts[counts > 0])
        reduced_counts = counts // min_count
        return tuple(int(rc) for rc in reduced_counts)

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
        for structure in tqdm.tqdm(dataset.get_structure_dataset(), desc=desc, total=len(dataset)):
            key = CRMSEReward._get_reduced_composition_key(structure.atomic_numbers.numpy(force=True))
            py_structure = structure.get_pymatgen_structure()
            if key not in reduced_composition_map:
                reduced_composition_map[key] = []
            reduced_composition_map[key].append(py_structure)
        return reduced_composition_map

    @staticmethod
    def _compute_rmse(py_structure: PymatgenStructure, reference_py_structures: Sequence[PymatgenStructure],
                      ltol: float, stol: float, angle_tol: float) -> float:
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
            res = sm.get_rms_dist(py_structure, ref_py_structure)
            assert res is None or res[0] <= stol
            rmses.append(stol if res is None else res[0])
        return min(rmses)

    def compute(self, structures: Sequence[Structure],
                stage: Reward.ComputeStage) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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
            key = self._get_reduced_composition_key(structure.atomic_numbers.numpy(force=True))
            py_structures.append(structure.get_pymatgen_structure())
            assert key in reduced_composition_map and len(reduced_composition_map[key]) > 0
            relevant_structures_list.append(reduced_composition_map[key])

        crmse_function = partial(self._compute_rmse, ltol=self._ltol, stol=self._stol, angle_tol=self._angle_tol)

        if self._cpu_count > 1:
            crmse_values = process_map(crmse_function, py_structures, relevant_structures_list,
                                       desc="Computing cRMSE rewards",
                                       chunksize=max(min(len(py_structures) // self._cpu_count, 100), 1),
                                       max_workers=self._cpu_count)
        else:
            crmse_values = list(map(crmse_function,
                                    tqdm.tqdm(py_structures, desc="Computing cRMSE rewards", total=len(py_structures)),
                                    relevant_structures_list))

        return self._scale * (self._stol - np.array(crmse_values)), {"cRMSE": np.array(crmse_values)}


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

    def compute(self, structures: Sequence[Structure],
                stage: Reward.ComputeStage) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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

        :return:
            (List of rewards per structure, info dictionary).
        :rtype: tuple[np.ndarray, dict[str, np.ndarray]]
        """
        total_rewards = np.zeros(len(structures))
        total_dict = {}
        for reward_function, weight in zip(self._reward_functions, self._weights):
            rewards, info_dict = reward_function.compute(structures, stage)
            total_rewards += weight * rewards
            for key, value in info_dict.items():
                assert key not in total_dict
                total_dict[key] = value
        return total_rewards, total_dict
