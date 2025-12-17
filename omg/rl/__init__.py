"""
Reinforcement Learning module for residual policy learning in OMatG.

This module provides a flexible framework for training residual velocity models
using various RL algorithms (REINFORCE, PPO, GRPO) with custom reward functions.

Recommended: Use OMGRLLightning for training (PyTorch Lightning interface).

For discrete flow matching (species), see residual_model_discrete module.
"""

from .residual_model import ResidualModel
from .residual_model_discrete import ResidualRateMatrix, combine_rate_matrices
from .reward_functions import RewardFunction, VolumeReward, DensityReward, StabilityReward, CompositeReward
from .rl_config import RLConfig
from .omg_rl_lightning import OMGRLLightning

# Legacy standalone trainer (kept for compatibility)
from .rl_trainer import RLTrainer

__all__ = [
    # Core models
    "ResidualModel",  # For continuous (pos, cell)
    "ResidualRateMatrix",  # For discrete (species)
    "combine_rate_matrices",  # Utility for discrete
    # Reward functions
    "RewardFunction",
    "VolumeReward",
    "DensityReward",
    "StabilityReward",
    "CompositeReward",
    # Configuration and training
    "RLConfig",
    "OMGRLLightning",  # Recommended
    "RLTrainer",  # Legacy
]
