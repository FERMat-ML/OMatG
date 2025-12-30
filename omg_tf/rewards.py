from typing import Sequence
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
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
    """
    def __init__(self, ltol: float = 0.3, stol: float = 0.5, angle_tol: float = 10.0) -> None:
        """Constructor for CRMSEReward."""
        super().__init__()
        self._ltol = ltol
        self._stol = stol
        self._angle_tol = angle_tol

    def compute(self, structures: Sequence[Structure], reference_dataset: OMGDataset) -> np.ndarray:
        """
        Compute rewards for a batch of structures.

        The reward is computed as (stol - cRMSE) for each structure, where cRMSE is the corrected root-mean-square
        error between the generated structure and the closest matching structure in the reference dataset with the same
        reduced composition. If no match is found, cRMSE is set to stol. Thus, higher rewards correspond to lower cRMSE
        values.

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
        crmse_values = np.zeros(len(structures))
        sm = StructureMatcher(ltol=self._ltol, stol=self._stol, angle_tol=self._angle_tol)
        # TODO: This might have to be parallelized.
        for structure_index, structure in enumerate(structures):
            py_structure = structure.get_pymatgen_structure()
            py_composition = py_structure.composition.reduced_composition
            relevant_py_structures = []
            for reference_structure in reference_dataset.get_structure_dataset():
                ref_py_structure = reference_structure.get_pymatgen_structure()
                if ref_py_structure.composition.reduced_composition == py_composition:
                    relevant_py_structures.append(ref_py_structure)
            # Match found structures and take smallest RMSE.
            # Use stol for non-matching structures.
            if len(relevant_py_structures) > 0:
                rmses = []
                for ref_py_structure in relevant_py_structures:
                    res = sm.get_rms_dist(py_structure, ref_py_structure)
                    assert res is None or res[0] <= self._stol
                    rmses.append(self._stol if res is None else res[0])
                crmse_values[structure_index] = min(rmses)
        return self._stol - crmse_values


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
