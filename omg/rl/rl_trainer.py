"""
RL Trainer for residual policy learning.

This module provides the main training loop for RL-based residual models using TorchRL.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from typing import List, Dict, Optional, Callable
from pathlib import Path
from tqdm import trange
import numpy as np
from ase import Atoms

from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.modules import ProbabilisticActor, ValueOperator

from omg.rl.residual_model import ResidualModel
from omg.rl.reward_functions import RewardFunction
from omg.rl.rl_config import RLConfig
from omg.datamodule import OMGData
from omg.model.model import Model
from omg.si.stochastic_interpolants import StochasticInterpolants
from omg.sampler.sampler import Sampler


class RLTrainer:
    """
    Trainer for residual policy learning using various RL algorithms.

    :param base_model: Frozen base flow matching model
    :type base_model: Model
    :param residual_model: Residual model to train
    :type residual_model: ResidualModel
    :param stochastic_interpolants: Stochastic interpolants for integration
    :type stochastic_interpolants: StochasticInterpolants
    :param sampler: Sampler for initial structures
    :type sampler: Sampler
    :param reward_function: Reward function to optimize
    :type reward_function: RewardFunction
    :param config: RL training configuration
    :type config: RLConfig
    :param checkpoint_dir: Directory to save checkpoints
    :type checkpoint_dir: Optional[Path]
    """

    def __init__(
        self,
        base_model: Model,
        residual_model: ResidualModel,
        stochastic_interpolants: StochasticInterpolants,
        sampler: Sampler,
        reward_function: RewardFunction,
        config: RLConfig,
        checkpoint_dir: Optional[Path] = None,
    ):
        """Constructor for RLTrainer."""
        self.base_model = base_model
        self.residual_model = residual_model
        self.si = stochastic_interpolants
        self.sampler = sampler
        self.reward_function = reward_function
        self.config = config

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        # Setup optimizer
        self.optimizer = Adam(
            self.residual_model.parameters(),
            lr=config.learning_rate,
        )

        # Setup checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metrics tracking
        self.metrics_history = {
            'iteration': [],
            'mean_reward': [],
            'std_reward': [],
            'mean_loss': [],
            'regularization_loss': [],
            'policy_loss': [],
        }

        # Move models to device
        self.device = torch.device(config.device)
        self.base_model.to(self.device)
        self.residual_model.to(self.device)

    def generate_trajectories(
        self,
        batch_data: OMGData,
    ) -> tuple[List[Atoms], torch.Tensor, List[torch.Tensor]]:
        """
        Generate structures using base model + residual model and collect log probabilities.

        :param batch_data: Batch of data for conditioning (e.g., compositions)
        :type batch_data: OMGData

        :return: Tuple of (generated structures, total log probs, mean residuals per step)
        :rtype: tuple[List[Atoms], torch.Tensor, List[torch.Tensor]]
        """
        self.residual_model.train()

        # Sample initial structures
        x_0 = self.sampler.sample_p_0(batch_data).to(self.device)

        # Track log probabilities throughout trajectory
        log_probs_per_step = []
        mean_residuals_per_step = []

        # Integrate with residual model
        batch_size = len(x_0.n_atoms)
        times = torch.linspace(0.0, 1.0, self.si._integration_time_steps, device=self.device)

        x_t = x_0.clone()

        for t_index in range(1, len(times)):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]

            # Get base velocity
            with torch.no_grad():
                base_output = self.base_model(x_t, t.repeat(batch_size))

            # Get residual velocity (with noise during training)
            residual_output, log_prob, mean_residual = self.residual_model(
                x_t, t.repeat(batch_size), return_log_prob=True
            )

            # Combine velocities
            combined_output = base_output.clone()
            for key in base_output.keys():
                if key.endswith('_b'):  # Velocity fields
                    combined_output[key] = base_output[key] + residual_output[key]

            # Update x_t (simplified Euler step)
            # In practice, this would use the full integration logic from StochasticInterpolants
            for key in x_t.keys():
                if key in combined_output and key.endswith('_b'):
                    # Assuming the data field is the prefix
                    data_field = key[:-2]  # Remove '_b'
                    if hasattr(x_t, data_field):
                        x_t[data_field] = x_t[data_field] + combined_output[key] * dt

            # Store log probs and mean residuals
            if log_prob is not None:
                log_probs_per_step.append(log_prob)
            mean_residuals_per_step.append(mean_residual)

        # Convert final structures to ASE Atoms
        x_t.to('cpu')
        structures = []
        for i in range(batch_size):
            lower, upper = x_t.ptr[i], x_t.ptr[i + 1]
            atoms = Atoms(
                numbers=x_t.species[lower:upper],
                scaled_positions=x_t.pos[lower:upper, :],
                cell=x_t.cell[i, :, :],
                pbc=(1, 1, 1)
            )
            structures.append(atoms)

        # Sum log probs over all time steps
        total_log_probs = sum(log_probs_per_step)

        return structures, total_log_probs, mean_residuals_per_step

    def compute_reinforce_loss(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        mean_residuals_per_step: List[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute REINFORCE loss.

        :param rewards: Rewards for each trajectory
        :type rewards: torch.Tensor
        :param log_probs: Log probabilities for each trajectory
        :type log_probs: torch.Tensor
        :param mean_residuals_per_step: Mean residuals for regularization
        :type mean_residuals_per_step: List[torch.Tensor]

        :return: Total loss and metrics dict
        :rtype: tuple[torch.Tensor, Dict[str, float]]
        """
        # Normalize rewards (baseline)
        normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Policy loss: -E[reward * log_prob]
        policy_loss = -(normalized_rewards * log_probs).mean()

        # Regularization: encourage small residuals
        reg_loss = 0.0
        for mean_res in mean_residuals_per_step:
            reg_loss += self.residual_model.compute_regularization_loss(mean_res)
        reg_loss = reg_loss / len(mean_residuals_per_step)

        # Total loss
        total_loss = policy_loss + reg_loss

        metrics = {
            'policy_loss': policy_loss.item(),
            'regularization_loss': reg_loss.item(),
        }

        return total_loss, metrics

    def compute_grpo_loss(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        mean_residuals_per_step: List[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO (Group Relative Policy Optimization) loss.

        :param rewards: Rewards for each trajectory
        :type rewards: torch.Tensor
        :param log_probs: Log probabilities for each trajectory
        :type log_probs: torch.Tensor
        :param mean_residuals_per_step: Mean residuals for regularization
        :type mean_residuals_per_step: List[torch.Tensor]

        :return: Total loss and metrics dict
        :rtype: tuple[torch.Tensor, Dict[str, float]]
        """
        # Split into groups and normalize within each group
        group_size = self.config.grpo_group_size
        num_groups = len(rewards) // group_size

        advantages = torch.zeros_like(rewards)

        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size

            group_rewards = rewards[start_idx:end_idx]

            # Normalize within group
            group_mean = group_rewards.mean()
            group_std = group_rewards.std()
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / (group_std + 1e-8)

        # Policy loss
        policy_loss = -(advantages * log_probs).mean()

        # Regularization
        reg_loss = 0.0
        for mean_res in mean_residuals_per_step:
            reg_loss += self.residual_model.compute_regularization_loss(mean_res)
        reg_loss = reg_loss / len(mean_residuals_per_step)

        # Total loss
        total_loss = policy_loss + reg_loss

        metrics = {
            'policy_loss': policy_loss.item(),
            'regularization_loss': reg_loss.item(),
        }

        return total_loss, metrics

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
    ):
        """
        Main training loop.

        :param dataloader: DataLoader providing batches of structures/compositions
        :type dataloader: torch.utils.data.DataLoader
        """
        iteration = 0

        for epoch in trange(self.config.num_iterations, desc="RL Training"):
            for batch in dataloader:
                batch = batch.to(self.device)

                # Generate trajectories
                structures, log_probs, mean_residuals = self.generate_trajectories(batch)

                # Compute rewards
                rewards = self.reward_function.compute(structures).to(self.device)

                # Compute loss based on algorithm
                if self.config.algorithm == 'reinforce':
                    loss, metrics = self.compute_reinforce_loss(
                        rewards, log_probs, mean_residuals
                    )
                elif self.config.algorithm == 'grpo':
                    loss, metrics = self.compute_grpo_loss(
                        rewards, log_probs, mean_residuals
                    )
                elif self.config.algorithm == 'ppo':
                    # PPO would require storing old log probs and doing multiple epochs
                    # For now, fall back to REINFORCE
                    loss, metrics = self.compute_reinforce_loss(
                        rewards, log_probs, mean_residuals
                    )
                else:
                    raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.residual_model.parameters(),
                        self.config.max_grad_norm
                    )
                self.optimizer.step()

                # Update noise scale if annealing
                if self.config.noise_anneal:
                    new_noise = self.config.noise_scale * (self.config.noise_anneal_factor ** iteration)
                    self.residual_model.set_noise_scale(new_noise)

                # Log metrics
                self.metrics_history['iteration'].append(iteration)
                self.metrics_history['mean_reward'].append(rewards.mean().item())
                self.metrics_history['std_reward'].append(rewards.std().item())
                self.metrics_history['mean_loss'].append(loss.item())
                self.metrics_history['regularization_loss'].append(metrics['regularization_loss'])
                self.metrics_history['policy_loss'].append(metrics['policy_loss'])

                if iteration % self.config.log_interval == 0:
                    print(f"\nIteration {iteration}:")
                    print(f"  Mean Reward: {rewards.mean().item():.4f} ± {rewards.std().item():.4f}")
                    print(f"  Loss: {loss.item():.4f}")
                    print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                    print(f"  Reg Loss: {metrics['regularization_loss']:.4f}")

                # Save checkpoint
                if self.checkpoint_dir is not None and iteration % self.config.save_interval == 0:
                    self.save_checkpoint(iteration)

                iteration += 1

    def save_checkpoint(self, iteration: int):
        """
        Save checkpoint.

        :param iteration: Current iteration number
        :type iteration: int
        """
        checkpoint = {
            'iteration': iteration,
            'residual_model_state_dict': self.residual_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics_history': self.metrics_history,
        }

        path = self.checkpoint_dir / f"checkpoint_{iteration}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path):
        """
        Load checkpoint.

        :param path: Path to checkpoint file
        :type path: Path
        """
        checkpoint = torch.load(path)
        self.residual_model.load_state_dict(checkpoint['residual_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics_history = checkpoint['metrics_history']
        print(f"Loaded checkpoint from {path}")
