from typing import Sequence
from ase import Atoms
import numpy as np
from .abstracts import Reward


class VolumeReward(Reward):
    """
    Simple reward function that encourages larger unit cell volumes.

    This is a toy example for testing the RL framework.
    """

    def __init__(self) -> None:
        """Constructor for VolumeReward."""
        super().__init__()

    def compute(self, structures: Sequence[Atoms]) -> np.ndarray:
        """
        Compute rewards for a batch of structures.

        :param structures:
            Sequence of ASE Atoms objects representing generated structures
        :type structures: Sequence[Atoms]

        :return:
            List of rewards, one per structure.
        :rtype: np.ndarray
        """
        return np.array([atoms.get_volume() for atoms in structures])


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

    def compute(self, structures: Sequence[Atoms]) -> np.ndarray:
        """
        Compute rewards for a batch of structures.

        :param structures:
            Sequence of ASE Atoms objects representing generated structures
        :type structures: Sequence[Atoms]

        :return:
            List of rewards, one per structure.
        :rtype: np.ndarray
        """
        total_rewards = np.zeros(len(structures))
        for reward_function, weight in zip(self._reward_functions, self._weights):
            rewards = reward_function.compute(structures)
            total_rewards += weight * rewards
        return total_rewards
