from pathlib import Path
from typing import Dict, List, Optional
from ase import Atoms
import numpy as np
import lightning
import torch
from torch_geometric.data import Batch
from omg.datamodule import OMGData
from omg.model.model import Model
from omg_tf.abstracts import Combiner, Reward
from omg_tf.base_modules import base_modules


class OMGTFLightning(lightning.LightningModule):
    """
    Lightning module for RL training of residual models.

    This class integrates with the existing OMatG Lightning infrastructure,
    providing a consistent interface for RL-based residual policy learning.
    """

    def __init__(
        self,
        residual_model: Model,
        combiner: Combiner,
        reward: Reward,
        grpo_group_size: int = 32,
        grpo_batch_size: int = 16,
        save_structures: bool = False,
        structure_save_dir: Optional[Path] = None,
    ):
        """Constructor for OMGRTFLightning."""
        super().__init__()

        self.residual_model = residual_model
        self.combiner = combiner
        self.reward = reward

        if not grpo_group_size > 0:
            raise ValueError("GRPO group size must be positive.")
        if not grpo_batch_size > 0:
            raise ValueError("GRPO batch size must be positive.")
        self.grpo_group_size = grpo_group_size
        self.grpo_batch_size = grpo_batch_size
        # Change batch size of base datamodule that is used by this class to grpo_batch_size.
        base_modules["datamodule"].kwargs["batch_size"] = grpo_batch_size

        # Structure saving
        self.save_structures = save_structures
        self.structure_save_dir = structure_save_dir
        if save_structures and structure_save_dir is not None:
            structure_save_dir.mkdir(parents=True, exist_ok=True)

        # Metrics tracking
        self.validation_structures = []
        self.validation_rewards = []

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
        # Replicate batch from training data according to GRPO group size.
        grpo_batch = Batch.from_data_list(
            [batch[i // self.grpo_group_size] for i in range(self.grpo_group_size * len(batch))]).to(self.device)

        # Sample initial structures independently for each structure in the GRPO groups.
        x_0 = base_modules["model"].sampler.sample_p_0(grpo_batch).to(self.device)

        x_1, log_probs = self.combiner.training_integrate(self.residual_model, x_0)
        assert len(log_probs) == len(x_0.n_atoms)

        # Convert to ASE Atoms.
        x_1 = x_1.to('cpu')
        structures = []
        for i in range(len(x_1.n_atoms)):
            sl = slice(x_1.ptr[i], x_1.ptr[i + 1])
            if x_1.pos_is_fractional[i]:
                structures.append(Atoms(numbers=x_1.species[sl], scaled_positions=x_1.pos[sl, :],
                                        cell=x_1.cell[i, :, :], pbc=True))
            else:
                structures.append(Atoms(numbers=x_1.species[sl], positions=x_1.pos[sl, :],
                                        cell=x_1.cell[i, :, :], pbc=True))

        rewards = self.reward.compute(structures)

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
