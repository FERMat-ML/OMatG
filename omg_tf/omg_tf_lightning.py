from pathlib import Path
from typing import Dict, List, Optional
from ase import Atoms
import numpy as np
import lightning
import torch
from torch_geometric.data import Batch
from omg.datamodule import OMGData
from omg.model.model import Model
from omg.utils import DataField
from omg_tf.abstracts import Combiner, Reward
from omg_tf.base_modules import base_modules


class OMGTFLightning(lightning.LightningModule):
    """
    Lightning module for RL training of residual models.

    This class integrates with the existing OMatG Lightning infrastructure,
    providing a consistent interface for RL-based residual policy learning.
    """

    def __init__(self, residual_model: Model, combiner: Combiner, reward: Reward,
                 relative_costs: dict[str, float], grpo_group_size: int = 32, grpo_num_groups: int = 16,
                 save_structures: bool = False, structure_save_dir: Optional[Path] = None) -> None:
        """Constructor for OMGRTFLightning."""
        super().__init__()

        self.residual_model = residual_model
        self.combiner = combiner
        self.reward = reward

        if not all(cost >= 0.0 for cost in relative_costs.values()):
            raise ValueError("All relative costs must be non-negative.")
        if not abs(sum(relative_costs.values()) - 1.0) < 1e-10:
            raise ValueError("The sum of all cost factors should be equal to 1.")
        integrated_data_fields = [df.name for df in self.combiner.integrated_data_fields()]
        for field in integrated_data_fields:
            if field + "_policy" not in relative_costs:
                raise ValueError(f"Missing relative cost for policy of integrated data field '{field}'.")
            if field + "_regularization" not in relative_costs:
                raise ValueError(f"Missing relative cost for regularization of data field '{field}'.")
        for field in relative_costs.keys():
            if field.endswith("_regularization"):
                base_field = field[:-len("_regularization")]
                if base_field not in integrated_data_fields:
                    raise ValueError(f"Relative cost for regularization provided for unknown data "
                                     f"field '{base_field}'.")
            elif field.endswith("_policy"):
                base_field = field[:-len("_policy")]
                if base_field not in integrated_data_fields:
                    raise ValueError(f"Relative cost for policy provided for unknown data field '{base_field}'.")
            else:
                raise ValueError(f"Relative cost provided for unknown term '{field}'.")

        if not grpo_group_size > 1:
            raise ValueError("GRPO group size must be bigger than one.")
        if not grpo_num_groups > 0:
            raise ValueError("GRPO batch size must be positive.")
        self.grpo_group_size = grpo_group_size
        self.grpo_num_groups = grpo_num_groups
        # Change batch size of base datamodule that is used by this class to grpo_num_groups.
        # We can then replicate each batch element grpo_group_size times during training and validation.
        base_modules["datamodule"].kwargs["batch_size"] = grpo_num_groups

        # Structure saving
        self.save_structures = save_structures
        self.structure_save_dir = structure_save_dir
        if save_structures and structure_save_dir is not None:
            structure_save_dir.mkdir(parents=True, exist_ok=True)

        # Metrics tracking
        self.validation_structures = []
        self.validation_rewards = []

    def _compute_grpo_losses(self, rewards: torch.Tensor, log_probs: dict[DataField, torch.Tensor],
                             mean_squared_residuals: dict[DataField, torch.Tensor]) -> dict[str, torch.Tensor]:
        # TODO: ADD CLIPPING!
        assert len(rewards) == self.grpo_num_groups * self.grpo_group_size
        assert all(len(lp) == len(rewards) for lp in log_probs.values())
        assert all(len(msr) == len(rewards) for msr in mean_squared_residuals.values())

        advantages = torch.zeros_like(rewards).detach()

        for i in range(self.grpo_num_groups):
            sl = slice(i * self.grpo_group_size, (i + 1) * self.grpo_group_size)
            group_rewards = rewards[sl]
            group_mean = group_rewards.mean()
            # This is not a statistical estimator but the population std within the group. Use unbiased=False.
            group_std = group_rewards.std(unbiased=False)
            # Add small epsilon to avoid division by zero.
            advantages[sl] = (group_rewards - group_mean) / (group_std + 1.0e-8)

        losses = {
            field.name + "_policy": -(advantages * log_probs[field]).mean()
            for field in self.combiner.integrated_data_fields()
        }
        losses.update({
            field.name + "_regularization": mean_squared_residuals[field].mean()
            for field in self.combiner.integrated_data_fields()}
        )

        return losses

    def training_step(self, batch: OMGData, batch_idx: int) -> torch.Tensor:
        # Replicate batch from training data according to GRPO group size.
        # noinspection PyTypeChecker
        grpo_batch = Batch.from_data_list(
            [batch[i // self.grpo_group_size] for i in range(self.grpo_group_size * len(batch))]).to(self.device)

        # Sample initial structures independently for each structure in the GRPO groups.
        x_0 = base_modules["model"].sampler.sample_p_0(grpo_batch).to(self.device)

        x_1, log_probs, mean_squared_residuals = self.combiner.training_integrate(self.residual_model, x_0)
        assert all(len(lp) == len(x_0.n_atoms) for lp in log_probs.values())
        assert (all(len(msr) == len(x_0.n_atoms) for msr in mean_squared_residuals.values()))

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

        rewards = torch.from_numpy(self.reward.compute(structures)).detach().to(self.device)

        # Compute loss
        losses = self._compute_grpo_losses(rewards, log_probs, mean_squared_residuals)
        total_loss = torch.tensor(0.0, device=self.device)

        for loss_key in losses:
            weight = self.rl_config.relative_costs[loss_key]
            losses[loss_key] = weight * losses[loss_key]
            total_loss += losses[loss_key]

        self.log_dict(losses, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(grpo_batch))
        self.log("loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(grpo_batch))
        self.log('reward_mean', rewards.mean(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('reward_std', rewards.std(), on_step=False, on_epoch=True)

        return total_loss

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
