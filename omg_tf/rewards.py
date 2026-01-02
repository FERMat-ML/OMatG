from functools import partial
import os
from typing import Optional, Sequence
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure as PymatgenStructure
import tqdm
from tqdm.contrib.concurrent import process_map
from omg.datamodule import OMGDataset, Structure
from .abstracts import Reward


class VolumeReward(Reward):
    """
    Simple reward function that encourages larger unit cell volumes.
    """

    def __init__(self) -> None:
        """Constructor for VolumeReward."""
        super().__init__()

    def compute(self, structures: Sequence[Structure], reference_dataset: OMGDataset) -> np.ndarray:
        """
        Compute rewards for a batch of structures.

        This reward is simply the volume of each structure's unit cell. The reference_dataset parameter is included
        for compatibility but is not used in this reward function.

        :param structures:
            Sequence of Structure objects representing generated structures.
        :type structures: Sequence[Structure]
        :param reference_dataset:
            Reference dataset for computing rewards.
        :type reference_dataset: OMGDataset

        :return:
            List of rewards, one per structure.
        :rtype: np.ndarray
        """
        return np.array([structure.get_ase_atoms().get_volume() for structure in structures])


class CRMSEReward(Reward):
    """
    Reward function that reduces the corrected root-mean-square error (cRMSE) between generated structures and known
    stable structures in a reference dataset.

    During the reward computation, each generated structure is compared to all structures in the reference dataset with
    the same reduced composition using Pymatgen's StructureMatcher. The cRMSE is calculated as the minimum RMSD
    among all matched structures. If no match is found within the specified tolerances, a penalty equal to the structure
    tolerance (stol) is assigned. The RMSD is computed using Pymatgen's get_rms_dist method and is normalized by
    (Vol / nsites) ** (1/3).  Hence, the cRMSE is similarly normalized.

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

    @staticmethod
    def _compute_rmse(py_structure: PymatgenStructure, reference_dataset: OMGDataset, ltol: float, stol: float,
                      angle_tol: float) -> float:
        """
        Compute the cRMSE between a generated structure and the closest matching structure in the reference dataset.

        :param py_structure:
            Generated structure as a Pymatgen Structure.
        :type py_structure: PymatgenStructure
        :param reference_dataset:
            Reference dataset for computing cRMSE.
        :type reference_dataset: OMGDataset
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
        py_composition = py_structure.composition.reduced_composition
        relevant_py_structures = []
        for reference_structure in reference_dataset.get_structure_dataset():
            ref_py_structure = reference_structure.get_pymatgen_structure()
            if ref_py_structure.composition.reduced_composition == py_composition:
                relevant_py_structures.append(ref_py_structure)
        # Match found structures and take smallest RMSE.
        # Use stol for non-matching structures.
        assert len(relevant_py_structures) > 0
        rmses = []
        for ref_py_structure in relevant_py_structures:
            res = sm.get_rms_dist(py_structure, ref_py_structure)
            assert res is None or res[0] <= stol
            rmses.append(stol if res is None else res[0])
        return min(rmses)

    def compute(self, structures: Sequence[Structure], reference_dataset: OMGDataset) -> np.ndarray:
        """
        Compute rewards for a batch of structures.

        The reward is computed as scale * (stol - cRMSE) for each structure, where cRMSE is the corrected
        root-mean-square error between the generated structure and the closest matching structure in the reference
        dataset with the same reduced composition. If no match is found, cRMSE is set to stol. Thus, higher rewards
        correspond to lower cRMSE values.

        :param structures:
            Sequence of Structure objects representing generated structures.
        :type structures: Sequence[Structure]
        :param reference_dataset:
            Reference dataset for computing rewards.
        :type reference_dataset: OMGDataset

        :return:
            List of rewards, one per structure.
        :rtype: np.ndarray
        """
        crmse_function = partial(self._compute_rmse, reference_dataset=reference_dataset, ltol=self._ltol,
                                 stol=self._stol, angle_tol=self._angle_tol)
        # Be careful to convert structures to pymatgen structures before parallel processing to avoid pickling issues
        # with torch tensors.
        py_structures = [structure.get_pymatgen_structure() for structure in structures]

        if self._cpu_count > 1:
            crmse_values = process_map(crmse_function, py_structures, desc="Computing cRMSE rewards",
                                       chunksize=max(min(len(py_structures) // self._cpu_count, 100), 1),
                                       max_workers=self._cpu_count)
        else:
            crmse_values = list(map(crmse_function, tqdm.tqdm(py_structures, desc="Computing cRMSE rewards",
                                                              total=len(py_structures))))

        return self._scale * (self._stol - np.array(crmse_values))


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

    def compute(self, structures: Sequence[Structure], reference_dataset: OMGDataset) -> np.ndarray:
        """
        Compute rewards for a batch of structures.

        Some reward functions may require access to a reference dataset for computing rewards (e.g., to compute
        similarity to known stable structures). This dataset is the training or validation dataset, depending on the
        context in which the reward is computed.

        :param structures:
            Sequence of Structure objects representing generated structures.
        :type structures: Sequence[Structure]
        :param reference_dataset:
            Reference dataset for computing rewards.
        :type reference_dataset: OMGDataset

        :return:
            List of rewards, one per structure.
        :rtype: np.ndarray
        """
        total_rewards = np.zeros(len(structures))
        for reward_function, weight in zip(self._reward_functions, self._weights):
            rewards = reward_function.compute(structures, reference_dataset)
            total_rewards += weight * rewards
        return total_rewards
