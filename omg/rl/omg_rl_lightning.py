"""
Lightning module for RL-based residual policy learning.

This module provides a PyTorch Lightning interface for training residual models,
consistent with the existing OMGLightning interface.
"""

import lightning
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
from ase import Atoms
from tqdm import trange

from omg.rl.residual_model import ResidualModel
from omg.rl.reward_functions import RewardFunction
from omg.rl.rl_config import RLConfig
from omg.datamodule import OMGData
from omg.model.model import Model
from omg.si.stochastic_interpolants import StochasticInterpolants
from omg.sampler.sampler import Sampler
from omg.globals import SMALL_TIME, BIG_TIME


class OMGRLLightning(lightning.LightningModule):
    """
    Lightning module for RL training of residual models.

    This class integrates with the existing OMatG Lightning infrastructure,
    providing a consistent interface for RL-based residual policy learning.

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
    :param save_structures: Whether to save generated structures during validation
    :type save_structures: bool
    :param structure_save_dir: Directory to save structures
    :type structure_save_dir: Optional[Path]
    """

    def __init__(
        self,
        base_model: Model,
        residual_model: ResidualModel,
        stochastic_interpolants: StochasticInterpolants,
        sampler: Sampler,
        reward_function: RewardFunction,
        config: RLConfig,
        save_structures: bool = False,
        structure_save_dir: Optional[Path] = None,
    ):
        """Constructor for OMGRLLightning."""
        super().__init__()

        # Store components
        self.base_model = base_model
        self.residual_model = residual_model
        self.si = stochastic_interpolants
        self.sampler = sampler
        self.reward_function = reward_function
        self.rl_config = config

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        # Structure saving
        self.save_structures = save_structures
        self.structure_save_dir = structure_save_dir
        if save_structures and structure_save_dir is not None:
            structure_save_dir.mkdir(parents=True, exist_ok=True)

        # Save hyperparameters (Lightning feature)
        self.save_hyperparameters(ignore=['base_model', 'residual_model',
                                          'stochastic_interpolants', 'sampler',
                                          'reward_function'])

        # Metrics tracking
        self.validation_structures = []
        self.validation_rewards = []

    def generate_trajectories(
        self,
        batch: OMGData,
    ) -> tuple[List[Atoms], torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Generate structures using base + residual model.

        :param batch: Batch of data for conditioning
        :type batch: OMGData

        :return: Tuple of (structures, log_probs, mean_residuals_per_step)
        :rtype: tuple[List[Atoms], torch.Tensor, List[Dict[str, torch.Tensor]]]
        """
        self.residual_model.train()

        # Sample initial structures
        x_0 = self.sampler.sample_p_0(batch).to(self.device)

        # Track log probabilities and mean residuals
        log_probs_per_step = []
        mean_residuals_per_step = []

        # Integration setup
        batch_size = len(x_0.n_atoms)
        times = torch.linspace(
            SMALL_TIME,
            BIG_TIME,
            self.si.integration_time_steps,
            device=self.device
        )

        # Clone for integration
        data_fields = [df.name for df in self.si._data_fields]
        x_t = x_0.clone(*data_fields)
        x_t_dict = x_t.to_dict()

        # Integration loop
        for t_index in range(1, len(times)):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]

            t_batch = t.repeat(batch_size)

            # Get base velocity (frozen)
            with torch.no_grad():
                base_output = self.base_model(x_t, t_batch)

            # Get residual velocity (with noise during training)
            residual_output, log_prob, mean_residual = self.residual_model(
                x_t, t_batch, return_log_prob=True
            )

            # Combine velocities and integrate each field
            for stochastic_interpolant, data_field in zip(
                self.si._stochastic_interpolants,
                self.si._data_fields
            ):
                b_key = data_field.name + "_b"

                if b_key in base_output:
                    # Combine base + residual
                    base_vel = base_output[b_key]
                    if b_key in residual_output:
                        total_vel = base_vel + residual_output[b_key]
                    else:
                        total_vel = base_vel

                    # Simple Euler step (could use more sophisticated integration)
                    x_t_dict[data_field.name] = x_t_dict[data_field.name] + total_vel * dt

            # Store log probs and mean residuals
            if log_prob is not None:
                log_probs_per_step.append(log_prob)
            mean_residuals_per_step.append(mean_residual)

        # Convert to ASE Atoms
        x_t = x_t.to('cpu')
        structures = []
        for i in range(batch_size):
            lower, upper = x_t.ptr[i], x_t.ptr[i + 1]
            atoms = Atoms(
                numbers=x_t.species[lower:upper].numpy(),
                scaled_positions=x_t.pos[lower:upper, :].numpy(),
                cell=x_t.cell[i, :, :].numpy(),
                pbc=(1, 1, 1)
            )
            structures.append(atoms)

        # Sum log probs over trajectory
        total_log_probs = sum(log_probs_per_step) if log_probs_per_step else torch.zeros(batch_size)

        return structures, total_log_probs, mean_residuals_per_step

    def compute_loss(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        mean_residuals_per_step: List[Dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute RL loss based on configured algorithm.

        :param rewards: Rewards for each trajectory
        :type rewards: torch.Tensor
        :param log_probs: Log probabilities for each trajectory
        :type log_probs: torch.Tensor
        :param mean_residuals_per_step: Mean residuals for regularization
        :type mean_residuals_per_step: List[Dict[str, torch.Tensor]]

        :return: Total loss and metrics dict
        :rtype: tuple[torch.Tensor, Dict[str, float]]
        """
        if self.rl_config.algorithm == 'reinforce':
            return self._compute_reinforce_loss(rewards, log_probs, mean_residuals_per_step)
        elif self.rl_config.algorithm == 'grpo':
            return self._compute_grpo_loss(rewards, log_probs, mean_residuals_per_step)
        elif self.rl_config.algorithm == 'ppo':
            # For now, fall back to REINFORCE
            # Full PPO would require storing old log probs and multiple epochs
            return self._compute_reinforce_loss(rewards, log_probs, mean_residuals_per_step)
        else:
            raise ValueError(f"Unknown algorithm: {self.rl_config.algorithm}")

    def _compute_reinforce_loss(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        mean_residuals_per_step: List[Dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute REINFORCE loss."""
        # Normalize rewards (baseline)
        normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Policy loss: -E[reward * log_prob]
        policy_loss = -(normalized_rewards * log_probs).mean()

        # Regularization
        reg_loss = self._compute_regularization_loss(mean_residuals_per_step)

        # Total loss
        total_loss = policy_loss + reg_loss

        metrics = {
            'policy_loss': policy_loss.item(),
            'regularization_loss': reg_loss.item(),
        }

        return total_loss, metrics

    def _compute_grpo_loss(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        mean_residuals_per_step: List[Dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute GRPO loss."""
        # Split into groups and normalize within each group
        group_size = self.rl_config.grpo_group_size
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
        reg_loss = self._compute_regularization_loss(mean_residuals_per_step)

        # Total loss
        total_loss = policy_loss + reg_loss

        metrics = {
            'policy_loss': policy_loss.item(),
            'regularization_loss': reg_loss.item(),
        }

        return total_loss, metrics

    def _compute_regularization_loss(
        self,
        mean_residuals_per_step: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute regularization loss averaged over trajectory."""
        reg_loss = 0.0
        for mean_res in mean_residuals_per_step:
            reg_loss = reg_loss + self.residual_model.compute_regularization_loss(mean_res)
        return reg_loss / len(mean_residuals_per_step)

    def training_step(self, batch: OMGData, batch_idx: int) -> torch.Tensor:
        """
        Training step (Lightning method).

        :param batch: Batch of data
        :type batch: OMGData
        :param batch_idx: Batch index
        :type batch_idx: int

        :return: Loss value
        :rtype: torch.Tensor
        """
        # Generate trajectories
        structures, log_probs, mean_residuals = self.generate_trajectories(batch)

        # Compute rewards
        rewards = self.reward_function.compute(structures).to(self.device)

        # Compute loss
        loss, metrics = self.compute_loss(rewards, log_probs, mean_residuals)

        # Log metrics (Lightning handles this)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_reward_mean', rewards.mean(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_reward_std', rewards.std(), on_step=False, on_epoch=True)
        self.log('train_policy_loss', metrics['policy_loss'], on_step=False, on_epoch=True)
        self.log('train_reg_loss', metrics['regularization_loss'], on_step=False, on_epoch=True)

        # Anneal noise if configured
        if self.rl_config.noise_anneal:
            current_step = self.global_step
            new_noise = self.rl_config.noise_scale * (self.rl_config.noise_anneal_factor ** current_step)
            self.residual_model.set_noise_scale(new_noise)
            self.log('noise_scale', new_noise, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: OMGData, batch_idx: int) -> torch.Tensor:
        """
        Validation step (Lightning method).

        :param batch: Batch of data
        :type batch: OMGData
        :param batch_idx: Batch index
        :type batch_idx: int

        :return: Loss value
        :rtype: torch.Tensor
        """
        # Generate trajectories
        structures, log_probs, mean_residuals = self.generate_trajectories(batch)

        # Compute rewards
        rewards = self.reward_function.compute(structures).to(self.device)

        # Compute loss
        loss, metrics = self.compute_loss(rewards, log_probs, mean_residuals)

        # Log validation metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_reward_mean', rewards.mean(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_reward_std', rewards.std(), on_step=False, on_epoch=True)
        self.log('val_policy_loss', metrics['policy_loss'], on_step=False, on_epoch=True)
        self.log('val_reg_loss', metrics['regularization_loss'], on_step=False, on_epoch=True)

        # Store structures for end-of-epoch analysis
        if self.save_structures:
            self.validation_structures.extend(structures)
            self.validation_rewards.extend(rewards.cpu().tolist())

        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of validation epoch (Lightning callback).

        Saves best structures if configured.
        """
        if self.save_structures and len(self.validation_structures) > 0:
            # Save top-k structures by reward
            k = min(10, len(self.validation_structures))
            top_indices = np.argsort(self.validation_rewards)[-k:]

            save_path = self.structure_save_dir / f"epoch_{self.current_epoch}"
            save_path.mkdir(exist_ok=True)

            for rank, idx in enumerate(top_indices):
                atoms = self.validation_structures[idx]
                reward = self.validation_rewards[idx]
                from ase.io import write
                write(
                    save_path / f"rank_{rank}_reward_{reward:.4f}.xyz",
                    atoms
                )

            # Clear for next epoch
            self.validation_structures = []
            self.validation_rewards = []

    def configure_optimizers(self):
        """
        Configure optimizers (Lightning method).

        :return: Optimizer
        """
        optimizer = Adam(
            self.residual_model.parameters(),
            lr=self.rl_config.learning_rate,
        )

        # Could add learning rate scheduler here
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, ...)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

        return optimizer

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch (Lightning callback)."""
        # Could add custom logic here (e.g., update reward function)
        pass

    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        """
        Called when saving checkpoint (Lightning callback).

        :param checkpoint: Checkpoint dictionary
        :type checkpoint: Dict
        """
        # Add custom data to checkpoint if needed
        checkpoint['rl_config'] = self.rl_config
        checkpoint['reward_function_type'] = type(self.reward_function).__name__

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        """
        Called when loading checkpoint (Lightning callback).

        :param checkpoint: Checkpoint dictionary
        :type checkpoint: Dict
        """
        # Restore custom data from checkpoint if needed
        pass
