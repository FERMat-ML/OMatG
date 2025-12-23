from typing import Sequence
from ase import Atoms
from .abstracts import Reward


class VolumeReward(Reward):
    """
    Simple reward function that encourages larger unit cell volumes.

    This is a toy example for testing the RL framework.
    """

    def __init__(self) -> None:
        """Constructor for VolumeReward."""
        super().__init__()

    def compute(self, structures: Sequence[Atoms]) -> list[float]:
        """
        Compute rewards for a batch of structures.

        :param structures:
            Sequence of ASE Atoms objects representing generated structures
        :type structures: Sequence[Atoms]

        :return:
            List of rewards, one per structure.
        :rtype: list[float]
        """
        return [atoms.get_volume() for atoms in structures]


class DensityReward(Reward):
    """
    Reward function based on density, encouraging structures near a target density.

    :param target_density: Target density in g/cm^3
    :type target_density: float
    :param tolerance: Tolerance for density matching
    :type tolerance: float
    """

    def __init__(self, target_density: float = 3.0, tolerance: float = 0.5) -> None:
        """Constructor for DensityReward."""
        super().__init__()
        self._target_density = target_density
        self._tolerance = tolerance

    def compute(self, structures: Sequence[Atoms]) -> list[float]:
        """
        Compute rewards for a batch of structures.

        :param structures:
            Sequence of ASE Atoms objects representing generated structures
        :type structures: Sequence[Atoms]

        :return:
            List of rewards, one per structure.
        :rtype: list[float]
        """
        densities = [atoms.get_density() for atoms in structures]
        density_diffs = [abs(d - self._target_density) for d in densities]

        # Reward is higher when closer to target
        return [max(0, 1.0 - diff / self._tolerance) for diff in density_diffs]


class CompositeReward(Reward):
    """
    Composite reward that combines multiple reward functions.

    :param reward_functions: List of reward functions to combine
    :type reward_functions: List[RewardFunction]
    :param weights: Weights for each reward function (must sum to 1)
    :type weights: List[float]
    """

    def __init__(self, reward_functions: Sequence[Reward], weights: Sequence[float]) -> None:
        """Constructor for CompositeReward."""
        super().__init__()
        if len(reward_functions) != len(weights):
            raise ValueError("Number of reward functions must match number of weights.")
        if not all(w > 0.0 for w in weights):
            raise ValueError("Weights must be positive.")
        if not abs(sum(weights) - 1.0) < 1e-6:
            raise ValueError("Weights must sum to 1.0")
        self._reward_functions = reward_functions
        self._weights = weights

    def compute(self, structures: Sequence[Atoms]) -> list[float]:
        """
        Compute rewards for a batch of structures.

        :param structures:
            Sequence of ASE Atoms objects representing generated structures
        :type structures: Sequence[Atoms]

        :return:
            List of rewards, one per structure.
        :rtype: list[float]
        """
        total_rewards = [0.0 for _ in structures]

        for reward, weight in zip(self._reward_functions, self._weights):
            rewards = reward.compute(structures)
            total_rewards = [tr + weight * r for tr, r in zip(total_rewards, rewards)]

        return total_rewards
