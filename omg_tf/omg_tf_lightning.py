from pathlib import Path
import time
from typing import Optional
import lightning
import torch
from torch_geometric.data import Batch
from omg.datamodule import OMGData, Structure
from omg.model.model import Model
from omg.utils import DataField, xyz_saver
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
                 grpo_share_x_0: bool = True, normalize_log_probs: bool = True,
                 generation_xyz_filename: Optional[str] = None) -> None:
        """
        Constructor for OMGTFLightning.

        :param residual_model:
            The residual model that predicts velocity corrections.
        :type residual_model: Model
        :param combiner:
            The combiner that integrates base and residual velocities.
        :type combiner: Combiner
        :param reward:
            The reward function to optimize.
        :type reward: Reward
        :param relative_costs:
            Dictionary of relative costs for each loss term (policy and regularization per field).
            Must sum to 1.0.
        :type relative_costs: dict[str, float]
        :param grpo_group_size:
            Number of samples per GRPO group.
        :type grpo_group_size: int
        :param grpo_num_groups:
            Number of GRPO groups per batch.
        :type grpo_num_groups: int
        :param grpo_share_x_0:
            If True, all group members share the same initial structure x_0.
        :type grpo_share_x_0: bool
        :param normalize_log_probs:
            If True, scale log probabilities by number of dimensions before computing policy loss.
            This ensures that structures with different numbers of atoms contribute equally to
            gradients, and that position and cell policy losses are comparable in scale.
            Note: This is gradient scaling for practical purposes; the combiners still return
            mathematically correct (summed) log probabilities.
        :type normalize_log_probs: bool
        :param generation_xyz_filename:
            Filename for saving generated structures during prediction.
        :type generation_xyz_filename: Optional[str]
        """
        super().__init__()

        self.residual_model = residual_model
        self.combiner = combiner
        self.reward = reward

        if normalize_log_probs and DataField.pos not in self.combiner.integrated_data_fields():
            raise ValueError("Log probability normalization requested but position is not an integrated data field.")
        self.normalize_log_probs = normalize_log_probs

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
        self.relative_costs = relative_costs

        if not grpo_group_size > 1:
            raise ValueError("GRPO group size must be bigger than one.")
        if not grpo_num_groups > 0:
            raise ValueError("GRPO batch size must be positive.")
        self.grpo_group_size = grpo_group_size
        self.grpo_num_groups = grpo_num_groups
        self.grpo_share_x_0 = grpo_share_x_0
        # Change training batch size of base datamodule that is used by this class to grpo_num_groups.
        # We can then replicate each batch element grpo_group_size times during training and validation.
        # noinspection PyUnresolvedReferences
        base_modules["datamodule"].train_batch_size = grpo_num_groups

        self.generation_xyz_filename = generation_xyz_filename

    # noinspection PyUnresolvedReferences
    def setup(self, stage: str) -> None:
        """
        Set up the reward function with the training and validation datasets.

        This is called by Lightning at the beginning of fit (train + validate), validate, test, or predict.

        :param stage:
            Stage of setup: 'fit', 'validate', 'test', or 'predict'.
        :type stage: str
        """
        if stage == "fit":
            self.reward.set_train_dataset(self.trainer.datamodule.train_dataset)
            self.reward.set_val_dataset(self.trainer.datamodule.val_dataset)
        elif stage == "validate":
            self.reward.set_val_dataset(self.trainer.datamodule.val_dataset)
        elif stage == "predict":
            self.reward.set_pred_dataset(self.trainer.datamodule.pred_dataset)

    def _compute_grpo_losses(self, rewards: torch.Tensor, log_probs: dict[DataField, torch.Tensor],
                             mean_squared_residuals: dict[DataField, torch.Tensor],
                             n_atoms: torch.Tensor) -> dict[str, torch.Tensor]:
        # TODO: ADD CLIPPING!
        # Partial batch possible.
        assert len(rewards) % self.grpo_group_size == 0
        assert len(rewards) // self.grpo_group_size <= self.grpo_num_groups
        assert all(len(lp) == len(rewards) for lp in log_probs.values())
        assert all(len(msr) == len(rewards) for msr in mean_squared_residuals.values())
        assert len(n_atoms) == len(rewards)

        advantages = torch.zeros_like(rewards).detach()

        for i in range(len(rewards) // self.grpo_group_size):
            sl = slice(i * self.grpo_group_size, (i + 1) * self.grpo_group_size)
            group_rewards = rewards[sl]
            group_mean = group_rewards.mean()
            # This is not a statistical estimator but the population std within the group. Use unbiased=False.
            group_std = group_rewards.std(unbiased=False)
            # Add small epsilon to avoid division by zero.
            advantages[sl] = (group_rewards - group_mean) / (group_std + 1.0e-8)

        # Apply normalization to log probs if requested.
        # This ensures structures with different sizes contribute equally to gradients,
        # and that position and cell policy losses are comparable in scale.
        if self.normalize_log_probs:
            assert DataField.pos in log_probs
            log_probs[DataField.pos] = log_probs[DataField.pos] / n_atoms

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
        # TODO: COMPARE TO RANDOM x_0 FOR EVERY GROUP MEMBER AND COMPUTING BASELINE REWARD FROM UNNOISED X_0
        if self.grpo_share_x_0:
            # Sample one x_0 per unique structure (not per group member).
            x_0_per_structure = base_modules["model"].sampler.sample_p_0(batch).to(self.device)

            # Replicate each x_0 sample grpo_group_size times.
            # This ensures all group members start from the same initial structure,
            # so reward variance within groups comes from residual actions, not x_0 differences.
            # noinspection PyTypeChecker,PyUnresolvedReferences
            x_0 = Batch.from_data_list(
                [x_0_per_structure[i // self.grpo_group_size]
                for i in range(self.grpo_group_size * len(batch))]).to(self.device)
        else:
            # Replicate batch from training data according to GRPO group size.
            # noinspection PyTypeChecker,PyUnresolvedReferences
            grpo_batch = Batch.from_data_list(
                [batch[i // self.grpo_group_size] for i in range(self.grpo_group_size * len(batch))]).to(self.device)

            # Sample initial structures independently for each structure in the GRPO groups.
            x_0 = base_modules["model"].sampler.sample_p_0(grpo_batch).to(self.device)

        x_1, log_probs, mean_squared_residuals = self.combiner.training_integrate(self.residual_model, x_0)
        assert all(len(lp) == len(x_0.n_atoms) for lp in log_probs.values())
        assert (all(len(msr) == len(x_0.n_atoms) for msr in mean_squared_residuals.values()))

        # Convert to Structures.
        x_1 = x_1.to('cpu')
        structures = []
        for i in range(len(x_1.n_atoms)):
            sl = slice(x_1.ptr[i], x_1.ptr[i + 1])
            structures.append(Structure(cell=x_1.cell[i, :, :].detach(),
                                        atomic_numbers=x_1.species[sl].detach(),
                                        pos=x_1.pos[sl, :].detach(),
                                        pos_is_fractional=x_1.pos_is_fractional[i]))

        rewards, info_dict = self.reward.compute(structures, Reward.ComputeStage.TRAIN)
        rewards = torch.tensor(rewards, dtype=self.dtype, device=self.device).detach()
        info_dict = {key: torch.tensor(value, dtype=self.dtype, device=self.device).detach().mean()
                     for key, value in info_dict.items()}

        losses = self._compute_grpo_losses(rewards, log_probs, mean_squared_residuals, x_0.n_atoms)
        total_loss = torch.tensor(0.0, device=self.device)

        for loss_key in losses.keys():
            weight = self.relative_costs[loss_key]
            losses[loss_key] = weight * losses[loss_key]
            total_loss += losses[loss_key]

        # Compute diagnostics for GRPO correlation
        group_reward_stds = []
        for i in range(len(rewards) // self.grpo_group_size):
            sl = slice(i * self.grpo_group_size, (i + 1) * self.grpo_group_size)
            group_reward_stds.append(rewards[sl].std(unbiased=False).item())
        within_group_std = torch.tensor(group_reward_stds).mean()

        self.log_dict(losses, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch))
        self.log("loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log("reward_mean", rewards.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log("reward_std", rewards.std(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log("reward_std_within_group", within_group_std, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=len(batch))
        self.log_dict(info_dict, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch))

        return total_loss

    def validation_step(self, batch: OMGData, batch_idx: int) -> None:
        # Sample initial structures independently.
        x_0 = base_modules["model"].sampler.sample_p_0(batch).to(self.device)

        # Use deterministic integration (no noise) for validation.
        x_1 = self.combiner.integrate(self.residual_model, x_0)

        # Convert to ASE Atoms.
        x_1 = x_1.to('cpu')
        structures = []
        for i in range(len(x_1.n_atoms)):
            sl = slice(x_1.ptr[i], x_1.ptr[i + 1])
            structures.append(Structure(cell=x_1.cell[i, :, :].detach(),
                                        atomic_numbers=x_1.species[sl].detach(),
                                        pos=x_1.pos[sl, :].detach(),
                                        pos_is_fractional=x_1.pos_is_fractional[i]))

        rewards, info_dict = self.reward.compute(structures, Reward.ComputeStage.VAL)
        rewards = torch.tensor(rewards, dtype=self.dtype, device=self.device)
        info_dict = {f"val_{key}": torch.tensor(value, dtype=self.dtype, device=self.device).mean()
                     for key, value in info_dict.items()}

        self.log("val_reward_mean", rewards.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log("val_reward_std", rewards.std(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log_dict(info_dict, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch))

    def predict_step(self, batch: OMGData) -> OMGData:
        # Sample initial structures independently.
        x_0 = base_modules["model"].sampler.sample_p_0(batch).to(self.device)

        # Use deterministic integration (no noise) for validation.
        x_1 = self.combiner.integrate(self.residual_model, x_0)

        # Store initial and final structures as XYZ files.
        filename = (Path(self.generation_xyz_filename) if self.generation_xyz_filename is not None
                    else Path(f"{time.strftime('%Y%m%d-%H%M%S')}.xyz"))
        init_filename = filename.with_stem(filename.stem + "_init")
        xyz_saver(x_0.to('cpu'), init_filename)
        xyz_saver(x_1.to('cpu'), filename)

        return x_1
