"""
Configuration class for RL training.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import torch


@dataclass
class RLConfig:
    """
    Configuration for RL training of residual models.

    :param algorithm: RL algorithm to use ('reinforce', 'ppo', 'grpo')
    :type algorithm: Literal['reinforce', 'ppo', 'grpo']
    :param batch_size: Number of trajectories to sample per update
    :type batch_size: int
    :param learning_rate: Learning rate for optimizer
    :type learning_rate: float
    :param num_iterations: Number of RL training iterations
    :type num_iterations: int
    :param noise_scale: Initial noise scale for stochastic policy
    :type noise_scale: float
    :param noise_anneal: Whether to anneal noise over training
    :type noise_anneal: bool
    :param noise_anneal_factor: Factor to multiply noise by each iteration (if annealing)
    :type noise_anneal_factor: float
    :param regularization_weight: Weight for L2 regularization on residuals
    :type regularization_weight: float
    :param gamma: Discount factor (if using episodic RL)
    :type gamma: float
    :param clip_epsilon: Clipping parameter for PPO
    :type clip_epsilon: float
    :param ppo_epochs: Number of epochs to run PPO updates
    :type ppo_epochs: int
    :param grpo_group_size: Group size for GRPO
    :type grpo_group_size: int
    :param value_coef: Coefficient for value loss (if using value network)
    :type value_coef: float
    :param entropy_coef: Coefficient for entropy bonus
    :type entropy_coef: float
    :param max_grad_norm: Maximum gradient norm for clipping
    :type max_grad_norm: float
    :param log_interval: How often to log metrics
    :type log_interval: int
    :param save_interval: How often to save checkpoints
    :type save_interval: int
    :param device: Device to run on ('cpu', 'cuda', or specific device)
    :type device: str
    """

    # Algorithm selection
    algorithm: Literal['reinforce', 'ppo', 'grpo'] = 'grpo'

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_iterations: int = 1000

    # Noise and exploration
    noise_scale: float = 0.1
    noise_anneal: bool = True
    noise_anneal_factor: float = 0.999

    # Regularization
    regularization_weight: float = 0.01

    # RL-specific parameters
    gamma: float = 1.0  # No discounting for single-step generation

    # PPO-specific
    clip_epsilon: float = 0.2
    ppo_epochs: int = 4

    # GRPO-specific
    grpo_group_size: int = 8

    # Loss coefficients
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # Optimization
    max_grad_norm: float = 1.0

    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 100

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        """Validate configuration."""
        if self.algorithm not in ['reinforce', 'ppo', 'grpo']:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.grpo_group_size > self.batch_size:
            raise ValueError("grpo_group_size must be <= batch_size")

        if self.batch_size % self.grpo_group_size != 0:
            raise ValueError("batch_size must be divisible by grpo_group_size")


# Need to import torch for default device check
import torch
