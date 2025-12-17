"""
Reward functions for RL training of residual models.

This module provides a flexible interface for defining custom reward functions,
along with several example implementations.
"""

from abc import ABC, abstractmethod
from typing import List
import torch
from ase import Atoms
import numpy as np


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.

    All reward functions should inherit from this class and implement the compute() method.
    """

    @abstractmethod
    def compute(self, structures: List[Atoms]) -> torch.Tensor:
        """
        Compute rewards for a batch of structures.

        :param structures: List of ASE Atoms objects representing generated structures
        :type structures: List[Atoms]

        :return: Tensor of rewards, one per structure
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def is_differentiable(self) -> bool:
        """
        Indicate whether this reward function is differentiable.

        :return: True if reward can be backpropagated through, False otherwise
        :rtype: bool
        """
        raise NotImplementedError


class VolumeReward(RewardFunction):
    """
    Simple reward function that encourages larger unit cell volumes.

    This is a toy example for testing the RL framework.

    :param scale: Scaling factor for the reward
    :type scale: float
    """

    def __init__(self, scale: float = 1.0):
        """Constructor for VolumeReward."""
        self.scale = scale

    def compute(self, structures: List[Atoms]) -> torch.Tensor:
        """
        Compute volume-based rewards.

        :param structures: List of ASE Atoms objects
        :type structures: List[Atoms]

        :return: Rewards based on volume
        :rtype: torch.Tensor
        """
        volumes = [atoms.get_volume() for atoms in structures]
        return torch.tensor(volumes, dtype=torch.float32) * self.scale

    def is_differentiable(self) -> bool:
        """Volume is differentiable w.r.t. cell parameters."""
        return True


class DensityReward(RewardFunction):
    """
    Reward function based on density, encouraging structures near a target density.

    :param target_density: Target density in g/cm^3
    :type target_density: float
    :param tolerance: Tolerance for density matching
    :type tolerance: float
    """

    def __init__(self, target_density: float = 3.0, tolerance: float = 0.5):
        """Constructor for DensityReward."""
        self.target_density = target_density
        self.tolerance = tolerance

    def compute(self, structures: List[Atoms]) -> torch.Tensor:
        """
        Compute density-based rewards.

        :param structures: List of ASE Atoms objects
        :type structures: List[Atoms]

        :return: Rewards based on proximity to target density
        :rtype: torch.Tensor
        """
        densities = [atoms.get_density() for atoms in structures]
        density_diffs = [abs(d - self.target_density) for d in densities]

        # Reward is higher when closer to target
        rewards = [max(0, 1.0 - diff / self.tolerance) for diff in density_diffs]

        return torch.tensor(rewards, dtype=torch.float32)

    def is_differentiable(self) -> bool:
        """Density is differentiable w.r.t. structure parameters."""
        return True


class StabilityReward(RewardFunction):
    """
    Reward function based on structure stability from an external relaxation process.

    This is a placeholder for interfacing with expensive energy calculations (DFT, force fields, etc.).
    The actual implementation would call external codes.

    :param energy_calculator: Optional callable that computes energy given an Atoms object
    :type energy_calculator: callable
    :param energy_scale: Scale factor to convert energies to rewards
    :type energy_scale: float
    """

    def __init__(self, energy_calculator=None, energy_scale: float = 1.0):
        """Constructor for StabilityReward."""
        self.energy_calculator = energy_calculator
        self.energy_scale = energy_scale

    def compute(self, structures: List[Atoms]) -> torch.Tensor:
        """
        Compute stability-based rewards.

        :param structures: List of ASE Atoms objects
        :type structures: List[Atoms]

        :return: Rewards based on stability (lower energy = higher reward)
        :rtype: torch.Tensor
        """
        if self.energy_calculator is None:
            # Placeholder: return dummy rewards
            # In practice, this would call DFT, ML potential, or other energy calculator
            return torch.ones(len(structures), dtype=torch.float32)

        energies = []
        for atoms in structures:
            # This is where you'd interface with your energy calculator
            # Example: atoms.calc = self.energy_calculator
            #          energy = atoms.get_potential_energy()
            energy = self.energy_calculator(atoms)
            energies.append(energy)

        # Convert energies to rewards (lower energy = higher reward)
        # Normalize by number of atoms
        energies = torch.tensor(energies, dtype=torch.float32)
        n_atoms = torch.tensor([len(atoms) for atoms in structures], dtype=torch.float32)
        energies_per_atom = energies / n_atoms

        # Negative energy as reward (more stable = more negative = higher reward when negated)
        rewards = -energies_per_atom * self.energy_scale

        return rewards

    def is_differentiable(self) -> bool:
        """Energy calculations are typically not differentiable."""
        return False


class CompositeReward(RewardFunction):
    """
    Composite reward that combines multiple reward functions.

    :param reward_functions: List of reward functions to combine
    :type reward_functions: List[RewardFunction]
    :param weights: Weights for each reward function (must sum to 1)
    :type weights: List[float]
    """

    def __init__(self, reward_functions: List[RewardFunction], weights: List[float]):
        """Constructor for CompositeReward."""
        if len(reward_functions) != len(weights):
            raise ValueError("Number of reward functions must match number of weights")

        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0")

        self.reward_functions = reward_functions
        self.weights = weights

    def compute(self, structures: List[Atoms]) -> torch.Tensor:
        """
        Compute weighted combination of multiple rewards.

        :param structures: List of ASE Atoms objects
        :type structures: List[Atoms]

        :return: Combined rewards
        :rtype: torch.Tensor
        """
        total_reward = torch.zeros(len(structures), dtype=torch.float32)

        for reward_fn, weight in zip(self.reward_functions, self.weights):
            rewards = reward_fn.compute(structures)
            total_reward += weight * rewards

        return total_reward

    def is_differentiable(self) -> bool:
        """Composite is differentiable only if all components are."""
        return all(rf.is_differentiable() for rf in self.reward_functions)
