from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import strftime
from typing import Optional
import lightning
import torch
from torch_geometric.data import Batch
from omg.datamodule import OMGData, Structure
from omg.si import (DiscreteFlowMatchingMask, DiscreteFlowMatchingUniform, SingleStochasticInterpolant,
                    SingleStochasticInterpolantIdentity, SingleStochasticInterpolantOS)
from omg.utils import DataField, xyz_saver
from omg.omg_irl.base_modules import base_modules
from omg.omg_irl.rewards import ComputeStage, Reward


@dataclass
class TrajectoryData:
    """Stores trajectory data from rollout for PPO-style training."""
    # List of x_t states at each timestep (detached).
    states: list[OMGData]
    # List of sampled policy actions at each timestep per field.
    sampled_actions: list[dict[DataField, torch.Tensor]]
    # List of old log probabilities at each timestep per field.
    old_log_probs: list[dict[DataField, torch.Tensor]]
    # List of base model outputs at each timestep (only necessary for full mode).
    base_model_outputs: list[dict[DataField, torch.Tensor]]
    # Rewards for final structures.
    rewards: torch.Tensor
    # Advantages for final structures.
    advantages: torch.Tensor
    # Info dictionary for final structures.
    info_dict: dict[str, torch.Tensor]


class OMGIRLLightningAbstract(ABC, lightning.LightningModule):
    """
    Abstract base class for Open Materials Generation with Inference-time Reinforcement Learning (OMatG-IRL) using
    group-relative policy optimization (GRPO) and proximal policy optimization (PPO) within the PyTorch Lightning
    framework.

    This class provides the foundational structure for implementing OMatG-IRL algorithms, including methods for
    rolling out trajectories, performing PPO updates, and integrating structures. It supports GRPO by grouping
    multiple samples per structure and normalizing rewards within these groups.

    This class relies on a globally set OMatG base model and datamodule in the `base_modules` dictionary. This OMatG
    model will be reinforced using the specified reward function.

    :param reward:
        Reward function to evaluate the generated structures.
    :type reward: Reward
    :param grpo_group_size:
        Number of samples per structure in each GRPO group.
        Must be greater than 1.
        Defaults to 32.
    :type grpo_group_size: int
    :param grpo_num_groups:
        Number of GRPO groups per training batch.
        The total number of structures in a training batch is grpo_group_size * grpo_num_groups.
        Must be greater than 0.
        Defaults to 16.
    :type grpo_num_groups: int
    :param grpo_share_x_0:
        If True, all group members share the same initial structure x_0, i.e., x_0 is sampled once per GRPO group.
        If False, x_0 is sampled independently for each group member.
        Defaults to True.
    :type grpo_share_x_0: bool
    :param ppo_clip_epsilon:
        PPO clipping epsilon for the surrogate objective.
        Must be non-negative.
        Defaults to 0.2.
    :type ppo_clip_epsilon: float
    :param ppo_epochs:
        Number of PPO epochs (passes over the same trajectory) per training step.
        Must be at least 1.
        Defaults to 1.
    :type ppo_epochs: int
    :param gradient_clip_val:
        Value for gradient clipping.
        If None, no gradient clipping is applied.
        Defaults to 1.0.
    :type gradient_clip_val: Optional[float]
    :param gradient_clip_algorithm:
        Algorithm for gradient clipping. Options are "norm" or "value".
        Defaults to "norm".
    :type gradient_clip_algorithm: str
    :param generation_xyz_filename:
        If provided, the filename to store predicted structures during prediction.
        If None, a timestamped filename will be generated.
        Must be an .xyz file if provided.
        Defaults to None.
    :type generation_xyz_filename: Optional[str]
    :param validation_xyz_filename:
        If provided, the filename to store validation structures during validation.
        This filename will be used as a prefix, and epoch and step information will be appended to it for each
        validation batch.
        If None, validation structures will not be stored.
        Must be an .xyz file if provided.
        Defaults to None.
    :type validation_xyz_filename: Optional[str]
    :param enable_progress_bar:
        If True or None, enables progress bars during reward computation and subclass-specific operations such as
        rollout, PPO update, and integration.
        We allow for None to mirror the setup of Lightning's Trainer (see also omg_irl_cli.py).
        Defaults to None.
    :type enable_progress_bar: bool

    :raises ValueError:
        If the base model was not set globally before initialization.
        If grpo_group_size is less than or equal to 1.
        If grpo_num_groups is less than or equal to 0.
        If ppo_clip_epsilon is negative.
        If ppo_epochs is less than 1.
        If generation_xyz_filename is not an .xyz file.
        If validation_xyz_filename is not an .xyz file.
        If none of position, cell, or species is integrated by the base model.
    """

    def __init__(self, reward: Reward, grpo_group_size: int = 32, grpo_num_groups: int = 16,
                 grpo_share_x_0: bool = True, ppo_clip_epsilon: float = 0.2, ppo_epochs: int = 1,
                 gradient_clip_val: Optional[float] = 1.0, gradient_clip_algorithm: str = "norm",
                 generation_xyz_filename: Optional[str] = None, validation_xyz_filename: Optional[str] = None,
                 enable_progress_bar: Optional[bool] = None) -> None:
        """Constructor of the OMGIRLLightningAbstract class."""
        super().__init__()
        # Use manual optimization for multiple optimizer steps per training_step.
        # See https://lightning.ai/docs/pytorch/stable/common/optimization.html.
        self.automatic_optimization = False
        self.reward = reward

        base_model = base_modules["model"]
        if base_model is None:
            raise ValueError("Base model must be set globally before initializing OMGIRLLightningAbstract.")
        self.integration_time_steps = base_model.si.get_integration_time_steps()
        base_model.freeze()

        if grpo_group_size <= 1:
            raise ValueError("GRPO group size must be > 1.")
        if grpo_num_groups <= 0:
            raise ValueError("GRPO num groups must be > 0.")
        self.grpo_group_size = grpo_group_size
        self.grpo_num_groups = grpo_num_groups
        self.grpo_share_x_0 = grpo_share_x_0
        base_modules["datamodule"].train_batch_size = grpo_num_groups

        if not ppo_clip_epsilon >= 0.0:
            raise ValueError("PPO clip epsilon must be non-negative.")
        if not ppo_epochs >= 1:
            raise ValueError("PPO epochs must be at least 1.")
        self.ppo_clip_epsilon = ppo_clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.gradient_clip_val: Optional[float] = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        if generation_xyz_filename is not None:
            if not generation_xyz_filename.endswith(".xyz"):
                raise ValueError("generation_xyz_filename must be an .xyz file.")
        self.generation_xyz_filename: Optional[str] = generation_xyz_filename
        self.enable_progress_bar = enable_progress_bar if enable_progress_bar is not None else True
        if validation_xyz_filename is not None:
            if not validation_xyz_filename.endswith(".xyz"):
                raise ValueError("validation_xyz_filename must be an .xyz file.")
        self.validation_xyz_filename: Optional[str] = validation_xyz_filename

        self.integrated_data_fields = set()
        self.fixed_data_fields = set()

        self.pos_interpolant = base_model.si.get_stochastic_interpolant(DataField.pos.name)
        self.integrate_pos = isinstance(self.pos_interpolant,
                                        (SingleStochasticInterpolant, SingleStochasticInterpolantOS))
        assert self.integrate_pos or isinstance(self.pos_interpolant, SingleStochasticInterpolantIdentity)
        self.pos_corrector = self.pos_interpolant.get_corrector() if self.integrate_pos else None
        if self.integrate_pos:
            self.integrated_data_fields.add(DataField.pos)
        else:
            self.fixed_data_fields.add(DataField.pos)

        self.cell_interpolant = base_model.si.get_stochastic_interpolant(DataField.cell.name)
        self.integrate_cell = isinstance(self.cell_interpolant,
                                         (SingleStochasticInterpolant, SingleStochasticInterpolantOS))
        assert self.integrate_cell or isinstance(self.cell_interpolant, SingleStochasticInterpolantIdentity)
        self.cell_corrector = self.cell_interpolant.get_corrector() if self.integrate_cell else None
        if self.integrate_cell:
            self.integrated_data_fields.add(DataField.cell)
        else:
            self.fixed_data_fields.add(DataField.cell)

        self.species_interpolant = base_model.si.get_stochastic_interpolant(DataField.species.name)
        self.integrate_species = isinstance(self.species_interpolant,
                                            (DiscreteFlowMatchingMask, DiscreteFlowMatchingUniform))
        assert self.integrate_species or isinstance(self.species_interpolant, SingleStochasticInterpolantIdentity)
        if self.integrate_species:
            self.integrated_data_fields.add(DataField.species)
        else:
            self.fixed_data_fields.add(DataField.species)

        assert len(self.fixed_data_fields) + len(self.integrated_data_fields) == 3
        if not len(self.integrated_data_fields) > 0:
            raise ValueError("At least one of position, cell, or species must be integrated by the base model.")

    # noinspection PyUnresolvedReferences
    def setup(self, stage: str) -> None:
        """
        Set up the reward function with the training, validation, and prediction datasets.

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

    @torch.no_grad()
    def rollout(self, x_0: OMGData) -> TrajectoryData:
        """
        Generate a trajectory rollout starting from initial structures x_0 without gradients, storing data for PPO
        updates.

        This method wraps the abstract _rollout method, which must be implemented in subclasses. It ensures that the
        rollout is performed without tracking gradients, making it suitable for generating trajectories used in PPO
        updates.

        :param x_0:
            Initial structures at time 0 sampled from p_0.
        :type x_0: OMGData

        :return:
            Trajectory data containing states, sampled actions, old log probabilities, base model outputs,
            rewards, advantages, and info dictionary for logging.
        :rtype: TrajectoryData
        """
        return self._rollout(x_0)

    @abstractmethod
    def _rollout(self, x_0: OMGData) -> TrajectoryData:
        """
        Generate a trajectory rollout starting from initial structures x_0 without gradients, storing data for PPO
        updates.

        This method must be implemented in subclasses to define the specific rollout procedure. It is called in no_grad
        context by the public rollout method.

        :param x_0:
            Initial structures at time 0 sampled from p_0.
        :type x_0: OMGData

        :return:
            Trajectory data containing states, sampled actions, old log probabilities, base model outputs,
            rewards, advantages, and info dictionary for logging.
        :rtype: TrajectoryData
        """
        raise NotImplementedError

    def _create_trajectory_data(self, states: list[OMGData],
                                sampled_actions: list[dict[DataField, torch.Tensor]],
                                old_log_probs: list[dict[DataField, torch.Tensor]],
                                base_model_outputs: list[dict[DataField, torch.Tensor]],
                                additional_info_dict: dict[str, torch.Tensor]) -> TrajectoryData:
        """
        Helper method to create TrajectoryData from rollout data.

        This method takes the states, sampled actions, old log probabilities, base model outputs, and any
        additional info collected during the rollout and computes the rewards and advantages for the final structures.

        :param states:
            List of x_t states at each timestep. The last state in the list should correspond to the final structures
            for which rewards will be computed.
        :type states: list[OMGData]
        :param sampled_actions:
            List of sampled policy actions at each timestep per field.
        :type sampled_actions: list[dict[DataField, torch.Tensor]]
        :param old_log_probs:
            List of old log probabilities at each timestep per field.
        :type old_log_probs: list[dict[DataField, torch.Tensor]]
        :param base_model_outputs:
            List of base model outputs at each timestep (only necessary for full mode).
        :type base_model_outputs: list[dict[DataField, torch.Tensor]]
        :param additional_info_dict:
            Additional info collected during rollout to be included in the info_dict of TrajectoryData.
        :type additional_info_dict: dict[str, torch.Tensor]

        :return:
            Trajectory data containing states, sampled actions, old log probabilities, base model outputs,
            rewards, advantages, and info dictionary for logging.
        :rtype: TrajectoryData
        """
        assert len(states) > 0
        # Convert to Structures.
        x_1 = states[-1].to('cpu')
        structures = []
        for i in range(len(x_1.n_atoms)):
            sl = slice(x_1.ptr[i], x_1.ptr[i + 1])
            structures.append(Structure(cell=x_1.cell[i, :, :],
                                        atomic_numbers=x_1.species[sl],
                                        pos=x_1.pos[sl, :],
                                        pos_is_fractional=x_1.pos_is_fractional[i]))

        # Compute rewards for final structures.
        rewards, info_dict = self.reward.compute(structures, ComputeStage.TRAIN,
                                                 enable_progress_bar=self.enable_progress_bar)
        rewards = torch.tensor(rewards, dtype=self.dtype, device=self.device)
        info_dict = {key: torch.tensor(value, dtype=self.dtype, device=self.device).mean()
                     for key, value in info_dict.items()}
        assert all(key not in info_dict for key in additional_info_dict)
        full_info_dict = info_dict | additional_info_dict

        # Compute GRPO advantages.
        # Partial batch possible.
        assert len(rewards) % self.grpo_group_size == 0
        assert len(rewards) // self.grpo_group_size <= self.grpo_num_groups
        advantages = torch.zeros_like(rewards)
        for i in range(len(rewards) // self.grpo_group_size):
            sl = slice(i * self.grpo_group_size, (i + 1) * self.grpo_group_size)
            group_rewards = rewards[sl]
            group_mean = group_rewards.mean()
            # This is not a statistical estimator but the population std within the group. Use unbiased=False.
            group_std = group_rewards.std(unbiased=False)
            # Add small epsilon to avoid division by zero.
            advantages[sl] = (group_rewards - group_mean) / (group_std + 1.0e-8)

        return TrajectoryData(
            states=states,
            sampled_actions=sampled_actions,
            old_log_probs=old_log_probs,
            base_model_outputs=base_model_outputs,
            rewards=rewards,
            advantages=advantages,
            info_dict=full_info_dict,
        )

    @abstractmethod
    def ppo_update(self, trajectory: TrajectoryData) -> tuple[float, dict[str, float]]:
        """
        Perform a PPO update using the provided trajectory data.

        This method must be implemented in subclasses to define the specific PPO update procedure. It takes the
        trajectory data generated during the rollout and performs one PPO update, returning the computed total loss and
        an additional dictionary for logging.

        The implementation of this method should use manual optimization, as the class is set up for manual optimization
        in the constructor. For instructions, see https://lightning.ai/docs/pytorch/stable/common/optimization.html.
        If gradient clipping is enabled, it should be applied before the optimizer step.

        :param trajectory:
            Trajectory data containing states, sampled actions, old log probabilities, base model outputs,
            rewards, advantages, and info dictionary for logging.
            This object is generated by the rollout method.
        :type trajectory: TrajectoryData

        :return:
            A tuple containing two elements:
            - The total loss as a float value.
            - A dictionary mapping arbitrary string keys to float values for logging. This can include individual loss
                components, clip fractions, or any other relevant metrics.
        :rtype: tuple[float, dict[str, float]]
        """
        raise NotImplementedError

    @torch.no_grad()
    def integrate(self, x_0: OMGData) -> OMGData:
        """
        Integrate the reinforced model starting from initial structures x_0.

        This method wraps the abstract _integrate method, which must be implemented in subclasses. It ensures that the
        integration is performed without tracking gradients.

        :param x_0:
            Initial structures at time 0 sampled from p_0.
        :type x_0: OMGData

        :return:
            Integrated structures at time 1 after applying the reinforced model.
        :rtype: OMGData
        """
        return self._integrate(x_0)

    @abstractmethod
    def _integrate(self, x_0: OMGData) -> OMGData:
        """
        Integrate the reinforced model starting from initial structures x_0.

        This method must be implemented in subclasses to define the specific integration procedure. It is called in
        no_grad context by the public integrate method.

        :param x_0:
            Initial structures at time 0 sampled from p_0.
        :type x_0: OMGData

        :return:
            Integrated structures at time 1 after applying the reinforced model.
        :rtype: OMGData
        """
        raise NotImplementedError

    def training_step(self, batch: OMGData, batch_idx: int) -> None:
        """
        Take a training step using GRPO with a PPO-like objective.

        The training step consists of the following steps:
        1. Sample initial structures x_0 for the GRPO groups.
        2. Perform a rollout without gradients to generate trajectories.
        3. Perform multiple PPO epochs with gradients over the same trajectory to update the model.

        :param batch:
            Batch of training data from the datamodule.
            This batch contains grpo_num_groups unique structures.
        :type batch: OMGData
        :param batch_idx:
            Index of the current batch.
        :type batch_idx: int
        """
        if self.grpo_share_x_0:
            # Sample one x_0 per unique structure (not per group member).
            x_0_per_structure = base_modules["model"].sampler.sample_p_0(batch).to(self.device)

            # Replicate each x_0 sample grpo_group_size times.
            # This ensures all group members start from the same initial structure.
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
            # Note that even though initial structures are independent, they might still be related. For example, for
            # the CSP task, all structures in a GRPO group have the same composition.
            x_0 = base_modules["model"].sampler.sample_p_0(grpo_batch).to(self.device)

        # Rollout without gradients.
        trajectory = self.rollout(x_0)

        # PPO update with gradients.
        # Each call to ppo_update performs one optimizer step.
        all_total_loss = 0.0
        all_logging_info = {}
        for epoch in range(self.ppo_epochs):
            total_loss, logging_info = self.ppo_update(trajectory)
            all_total_loss += total_loss
            for key, value in logging_info.items():
                if key not in all_logging_info:
                    all_logging_info[key] = 0.0
                all_logging_info[key] += value

        # Average over PPO epochs.
        all_total_loss /= self.ppo_epochs
        for key in all_logging_info:
            all_logging_info[key] /= self.ppo_epochs

        # Compute diagnostics.
        rewards = trajectory.rewards
        group_reward_stds = []
        for i in range(len(rewards) // self.grpo_group_size):
            sl = slice(i * self.grpo_group_size, (i + 1) * self.grpo_group_size)
            group_reward_stds.append(rewards[sl].std(unbiased=False).item())
        within_group_std = torch.tensor(group_reward_stds).mean()

        self.log_dict(all_logging_info, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch))
        self.log("loss_total", all_total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log("reward_mean", rewards.mean(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log("reward_std", rewards.std(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log("reward_std_within_group", within_group_std, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=len(batch))
        self.log_dict(trajectory.info_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                      batch_size=len(batch))

    def validation_step(self, batch: OMGData, batch_idx: int) -> None:
        """
        Take a validation step by sampling initial structures, integrating them, and computing rewards for the final
        structures.

        This method does not rely on GRPO groups and simply samples initial structures for the entire batch.

        The integration of the reinforced model is defined by the integrate method, which can be implemented in
        subclasses to define specific integration procedures.

        :param batch:
            Batch of validation data from the datamodule.
        :type batch: OMGData
        :param batch_idx:
            Index of the current batch.
        :type batch_idx: int
        """
        # Sample initial structures independently.
        x_0 = base_modules["model"].sampler.sample_p_0(batch).to(self.device)

        x_1 = self.integrate(x_0)

        # Convert to Structures.
        x_1 = x_1.to('cpu')
        structures = []
        for i in range(len(x_1.n_atoms)):
            sl = slice(x_1.ptr[i], x_1.ptr[i + 1])
            structures.append(Structure(cell=x_1.cell[i, :, :].detach(),
                                        atomic_numbers=x_1.species[sl].detach(),
                                        pos=x_1.pos[sl, :].detach(),
                                        pos_is_fractional=x_1.pos_is_fractional[i]))

        # Compute rewards for final structures.
        rewards, info_dict = self.reward.compute(structures, ComputeStage.VAL,
                                                 enable_progress_bar=self.enable_progress_bar)
        rewards = torch.tensor(rewards, dtype=self.dtype, device=self.device)
        info_dict = {f"val_{key}": torch.tensor(value, dtype=self.dtype, device=self.device).mean()
                     for key, value in info_dict.items()}

        self.log("val_reward_mean", rewards.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log("val_reward_std", rewards.std(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log_dict(info_dict, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch))

        if self.validation_xyz_filename is not None:
            filename = Path(self.validation_xyz_filename)
            epoch_filename = filename.with_stem(
                f"{filename.stem}_epoch_{self.current_epoch:04d}_step_{self.global_step:06d}")
            if epoch_filename.exists() and batch_idx == 0:
                epoch_filename.unlink()
            xyz_saver(x_1, epoch_filename)

    def predict_step(self, batch: OMGData) -> OMGData:
        """
        Take a prediction step by sampling initial structures, integrating them, and returning the final structures.

        This method does not rely on GRPO groups and simply samples initial structures for the entire batch.

        The integration of the reinforced model is defined by the integrate method, which can be implemented in
        subclasses to define specific integration procedures.

        :param batch:
            Batch of prediction data from the datamodule.
        :type batch: OMGData
        """
        # Sample initial structures independently.
        x_0 = base_modules["model"].sampler.sample_p_0(batch).to(self.device)

        x_1 = self.integrate(x_0)

        # Store initial and final structures as XYZ files.
        filename = (Path(self.generation_xyz_filename) if self.generation_xyz_filename is not None
                    else Path(f"{strftime('%Y%m%d-%H%M%S')}.xyz"))
        init_filename = filename.with_stem(filename.stem + "_init")
        xyz_saver(x_0.to('cpu'), init_filename)
        xyz_saver(x_1.to('cpu'), filename)

        return x_1
