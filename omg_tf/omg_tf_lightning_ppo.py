from dataclasses import dataclass
from pathlib import Path
from enum import auto, Enum
from time import strftime
from typing import Optional, Union
import lightning
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_scatter import scatter_add, scatter_mean
from tqdm import trange
from omg.datamodule import OMGData, Structure
from omg.globals import BIG_TIME, SMALL_TIME
from omg.model.model import Model
from omg.si import (DifferentialEquationType, SingleStochasticInterpolant, SingleStochasticInterpolantOS,
                    SingleStochasticInterpolantIdentity)
from omg.utils import DataField, xyz_saver
from omg_tf.abstracts import Reward, NoiseSchedule
from omg_tf.base_modules import base_modules
from omg_tf.time_mlp import TimeMLP


@dataclass
class TrajectoryData:
    """Stores trajectory data from rollout for PPO-style training."""
    # List of x_t states at each timestep (detached).
    states: list[OMGData]
    # List of residual model effects at each timestep per field.
    sampled_residual_effects: list[dict[DataField, torch.Tensor]]
    # List of old log probabilities at each timestep per field.
    old_log_probs: list[dict[DataField, torch.Tensor]]
    # Rewards for final structures.
    rewards: torch.Tensor
    # Advantages for final structures.
    advantages: torch.Tensor
    # Info dictionary for final structures.
    info_dict: dict[str, torch.Tensor]


# noinspection DuplicatedCode
class OMGTFLightningPPO(lightning.LightningModule):
    """
    Lightning module for PPO-style RL training of residual models.

    This class uses a two-phase approach inspired by Chemeleon2:

    Phase 1 (Rollout): Generate trajectories WITHOUT gradients, storing:
    - States x_t at each timestep
    - Sampled residual actions
    - Old log probabilities

    Phase 2 (Training): Re-evaluate the policy at each timestep WITH gradients:
    - Compute current log probabilities
    - Compute PPO-style clipped surrogate loss
    - Backpropagate through single timestep at a time (memory efficient)

    This allows noise injection at every timestep without memory explosion,
    providing stronger learning signal than single-step approaches.
    """

    class PositionNormalization(Enum):
        """
        Enumeration of position normalization modes for position residuals.
        """

        NONE = auto()
        """
        Use joint-action PPO ratio without normalization.

        This is the standard PPO approach where the log probabilities of all atoms in a structure are summed
        to compute the joint action log probability for the entire structure.
        """
        PER_STRUCTURE_WEIGHT = auto()
        """
        Use joint-action PPO ratio, but weight per-structure advantage by 1 / n_atoms.

        This normalization reduces the influence of larger structures on the policy update.
        """
        PER_ATOM_SURROGATE = auto()
        """
        Use per-atom PPO ratio averaged per structure.

        This normalization computes the PPO ratio for each atom individually and averages them per structure,
        effectively normalizing the policy update on a per-atom basis.
        """

    class ResidualMode(Enum):
        """
        Enumeration of residual application modes.
        """

        ADDITIVE = auto()
        """
        Additive residual: v_total = base_b + res_b.

        The residual model output is added to the base velocity.
        """
        SCALE = auto()
        """
        Multiplicative residual: v_total = (1 + res_b) * base_b.

        The residual model output is used to scale the base velocity.
        This is useful for learning speed adjustments to the base flow.
        """

    def __init__(self, residual_model: Union[Model, TimeMLP], reward: Reward, noise_schedules: dict[str, NoiseSchedule],
                 relative_costs: dict[str, float], grpo_group_size: int = 32, grpo_num_groups: int = 16,
                 grpo_share_x_0: bool = True, ppo_clip_epsilon: float = 0.2, ppo_epochs: int = 1,
                 gradient_clip_val: Optional[float] = 1.0, gradient_clip_algorithm: str = "norm",
                 generation_xyz_filename: Optional[str] = None, reference_sigma: float = 1e-3,
                 position_normalization: str = "none", residual_mode: str = "additive") -> None:
        """
        Constructor for OMGTFLightningPPO.

        :param residual_model:
            The residual model that predicts velocity corrections.
        :type residual_model: Model
        :param reward:
            The reward function to optimize.
        :type reward: Reward
        :param noise_schedules:
            Dictionary mapping data field names to noise schedules for stochastic policy.
        :type noise_schedules: dict[str, float]
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
        :param ppo_clip_epsilon:
            PPO clipping parameter epsilon. Ratio is clipped to [1-eps, 1+eps].
        :type ppo_clip_epsilon: float
        :param ppo_epochs:
            Number of PPO update epochs per rollout.
        :type ppo_epochs: int
        :param gradient_clip_val:
            Maximum gradient norm for clipping. Use None to disable gradient clipping.
        :type gradient_clip_val: Optional[float]
        :param gradient_clip_algorithm:
            Algorithm for gradient clipping: "norm" (clip by total norm) or "value" (clip by value).
        :type gradient_clip_algorithm: str
        :param generation_xyz_filename:
            Filename for saving generated structures during prediction.
        :type generation_xyz_filename: Optional[str]
        :param reference_sigma:
            Reference sigma value.
        :type reference_sigma: float
        :param position_normalization:
            Position normalization mode for position residuals. Options:
            - "none": joint-action PPO ratio without normalization
            - "per_structure_weight": joint-action ratio, but weight per-structure advantage by 1 / n_atoms
            - "per_atom_surrogate": per-atom PPO ratio averaged per structure
        :type position_normalization: str
        :param residual_mode:
            Mode for applying residual model output. Options:
            - "additive": v_total = base_b + res_b (default)
            - "scale": v_total = (1 + res_b) * base_b
        :type residual_mode: str
        """
        super().__init__()

        # Use manual optimization for multiple optimizer steps per training_step.
        # See https://lightning.ai/docs/pytorch/stable/common/optimization.html.
        self.automatic_optimization = False

        base_model = base_modules["model"]
        if base_model is None:
            raise ValueError("Base model must be set globally before initializing OMGTFLightningPPO.")
        base_model.freeze()

        self.residual_model = residual_model
        self.reward = reward

        try:
            self.residual_mode = self.ResidualMode[residual_mode.upper()]
        except KeyError:
            raise ValueError(f"Invalid residual mode '{residual_mode}'. Must be 'additive' or 'scale'.")
        if self.residual_mode == self.ResidualMode.ADDITIVE:
            if not isinstance(self.residual_model, Model):
                raise ValueError("Additive residual mode requires residual model to be of type Model.")
        else:
            assert self.residual_mode == self.ResidualMode.SCALE
            if not isinstance(self.residual_model, TimeMLP):
                raise ValueError("Scale residual mode requires residual model to be of type TimeMLP.")

        self.integrated_data_fields = []
        pos_interpolant = base_model.si.get_stochastic_interpolant(DataField.pos.name)
        self.integrate_pos = not isinstance(pos_interpolant, SingleStochasticInterpolantIdentity)
        self.pos_corrector = None
        if self.integrate_pos:
            if isinstance(pos_interpolant, SingleStochasticInterpolant):
                if not pos_interpolant.differential_equation_type == DifferentialEquationType.ODE:
                    raise ValueError("Residual integration for positions is only supported for ODE-based stochastic "
                                     "interpolants.")
                if pos_interpolant.velocity_annealing_factor != 0.0:
                    raise ValueError("Residual integration for positions is not supported with velocity annealing.")
                self.pos_corrector = pos_interpolant.get_corrector()
            elif isinstance(pos_interpolant, SingleStochasticInterpolantOS):
                if not pos_interpolant.differential_equation_type == DifferentialEquationType.ODE:
                    raise ValueError("Residual integration for positions is only supported for ODE-based stochastic "
                                     "interpolants.")
                if not pos_interpolant.predict_velocity:
                    raise ValueError("Residual integration for positions is only supported when predicting velocity.")
                if pos_interpolant.velocity_annealing_factor != 0.0:
                    raise ValueError("Residual integration for positions is not supported with velocity annealing.")
                self.pos_corrector = pos_interpolant.get_corrector()
            else:
                raise ValueError("Unsupported stochastic interpolant for position residual integration.")
            self.integrated_data_fields.append(DataField.pos)
            if DataField.pos.name not in noise_schedules:
                raise ValueError("Noise schedule for position residuals must be provided when integrating positions.")
        else:
            if DataField.pos.name in noise_schedules:
                raise ValueError("Noise schedule for position residuals provided but positions are not integrated.")
        assert ((self.integrate_pos and self.pos_corrector is not None)
                or (not self.integrate_pos and self.pos_corrector is None))

        cell_interpolant = base_model.si.get_stochastic_interpolant(DataField.cell.name)
        self.integrate_cell = not isinstance(cell_interpolant, SingleStochasticInterpolantIdentity)
        self.cell_corrector = None
        if self.integrate_cell:
            if isinstance(cell_interpolant, SingleStochasticInterpolant):
                if not cell_interpolant.differential_equation_type == DifferentialEquationType.ODE:
                    raise ValueError("Residual integration for cell is only supported for ODE-based stochastic "
                                     "interpolants.")
                if cell_interpolant.velocity_annealing_factor != 0.0:
                    raise ValueError("Residual integration for cell is not supported with velocity annealing.")
                self.cell_corrector = cell_interpolant.get_corrector()
            elif isinstance(cell_interpolant, SingleStochasticInterpolantOS):
                if not cell_interpolant.differential_equation_type == DifferentialEquationType.ODE:
                    raise ValueError("Residual integration for cell is only supported for ODE-based stochastic "
                                     "interpolants.")
                if not cell_interpolant.predict_velocity:
                    raise ValueError("Residual integration for cell is only supported when predicting velocity.")
                if cell_interpolant.velocity_annealing_factor != 0.0:
                    raise ValueError("Residual integration for cell is not supported with velocity annealing.")
                self.cell_corrector = cell_interpolant.get_corrector()
            else:
                raise ValueError("Unsupported stochastic interpolant for cell residual integration.")
            self.integrated_data_fields.append(DataField.cell)
            if DataField.cell.name not in noise_schedules:
                raise ValueError("Noise schedule for cell residuals must be provided when integrating cell.")
        else:
            if DataField.cell.name in noise_schedules:
                raise ValueError("Noise schedule for cell residuals provided but cell is not integrated.")
        assert ((self.integrate_cell and self.cell_corrector is not None)
                or (not self.integrate_cell and self.cell_corrector is None))

        species_interpolant = base_model.si.get_stochastic_interpolant(DataField.species.name)
        self.integrate_species = not isinstance(species_interpolant, SingleStochasticInterpolantIdentity)
        if self.integrate_species:
            raise ValueError("Residual integration for species is not supported yet.")
        else:
            if DataField.species.name in noise_schedules:
                raise ValueError("Noise schedule for species residuals provided but species are not integrated.")

        try:
            self.noise_schedules = {DataField[key]: value for key, value in noise_schedules.items()}
        except KeyError as e:
            raise ValueError(f"Invalid data field key in noise schedules: {e}")
        # Store learnable noise schedules in ModuleDict for proper registration.
        self.noise_schedule_modules = nn.ModuleDict()
        for field, schedule in self.noise_schedules.items():
            if isinstance(schedule, nn.Module):
                assert schedule.learnable()
                self.noise_schedule_modules[field.name] = schedule

        if not len(self.integrated_data_fields) > 0:
            raise ValueError("At least one of position, cell, or species must be integrated.")

        self.integration_time_steps = base_model.si.integration_time_steps

        try:
            normalization = self.PositionNormalization[position_normalization.upper()]
        except KeyError:
            raise ValueError(f"Invalid position normalization: {position_normalization}")
        if normalization != self.PositionNormalization.NONE and DataField.pos not in self.integrated_data_fields:
            raise ValueError("Position normalization requires position to be an integrated field.")
        if normalization != self.PositionNormalization.NONE and self.residual_mode != self.ResidualMode.ADDITIVE:
            raise ValueError("Position normalization is only supported for additive residual mode.")
        self.position_normalization = normalization

        if not all(cost >= 0.0 for cost in relative_costs.values()):
            raise ValueError("All relative costs must be non-negative.")
        if not abs(sum(relative_costs.values()) - 1.0) < 1e-10:
            raise ValueError("The sum of all cost factors should be equal to 1.")
        for field in self.integrated_data_fields:
            if field.name + "_policy" not in relative_costs:
                raise ValueError(f"Missing relative cost for policy of integrated data field '{field.name}'.")
            if field.name + "_regularization" not in relative_costs:
                raise ValueError(f"Missing relative cost for regularization of data field '{field.name}'.")
            if self.noise_schedules[field].learnable() and field.name + "_entropy" not in relative_costs:
                raise ValueError(f"Missing relative cost for entropy of data field '{field.name}' with learnable noise "
                                 f"schedule.")
        integrated_data_fields = [df.name for df in self.integrated_data_fields]
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
            elif field.endswith("_entropy"):
                base_field = field[:-len("_entropy")]
                if base_field not in integrated_data_fields:
                    raise ValueError(f"Relative cost for entropy provided for unknown data field '{base_field}'.")
                if not self.noise_schedules[DataField[base_field]].learnable():
                    raise ValueError(f"Relative cost for entropy provided for data field '{base_field}' without "
                                     f"learnable noise schedule.")
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

        if not ppo_clip_epsilon >= 0.0:
            raise ValueError("PPO clip ratio must be non-negative.")
        if not ppo_epochs >= 1:
            raise ValueError("PPO epochs must be at least 1.")
        self.ppo_clip_epsilon = ppo_clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        self.generation_xyz_filename = generation_xyz_filename

        if not reference_sigma > 0.0:
            raise ValueError("Reference sigma must be positive.")
        self.reference_sigma = reference_sigma

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

    @torch.no_grad()
    def rollout(self, x_0: OMGData) -> TrajectoryData:
        """
        Generate trajectory without gradients, storing data for PPO training.

        :param x_0:
            Initial structures at time 0.
        :type x_0: OMGData

        :return:
            TrajectoryData containing states, actions, and old log probs.
        :rtype: TrajectoryData
        """
        base_model = base_modules["model"].model
        assert base_model is not None
        batch_size = len(x_0.n_atoms)
        times = torch.linspace(SMALL_TIME, BIG_TIME, self.integration_time_steps, device=self.device)
        x_t = x_0.clone()
        states = []
        old_log_probs = []
        sampled_residual_effects = []

        # Integrate over time with residuals.
        for t_index in trange(1, len(times), desc="Rollout with residuals", position=1, leave=False):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            sqrt_dt = torch.sqrt(dt)
            time = t.repeat(batch_size)
            base_model_output = base_model(x_t, time)
            residual_output = self.residual_model(x_t, time)

            states.append(x_t.clone())
            step_sampled_residual_effects = {}
            step_old_log_probs = {}

            if self.integrate_pos:
                base_b = base_model_output[DataField.pos.name + "_b"]
                sigma = self.noise_schedules[DataField.pos].noise(t)  # Scalar.
                assert sigma.ndim == 0

                if self.residual_mode == self.ResidualMode.ADDITIVE:
                    res_b = residual_output[DataField.pos.name + "_b"]
                    assert res_b.shape == base_b.shape
                    velocity = base_b + res_b
                    randn = torch.randn_like(res_b)
                    noise = sigma * randn
                    # Store deviation effect for PPO update: res_b * dt + noise * sqrt(dt).
                    # Effectively x_t+dt - x_t - base_b * dt. Shape (sum(n_atoms), 3).
                    step_sampled_residual_effects[DataField.pos] = res_b * dt + noise * sqrt_dt
                    # Log probability of x_t+dt given x_t for SDE.
                    # Sum log probs over x, y, and z for additive mode yielding tensor of shape (sum(n_atoms),).
                    log_probs_atoms = -0.5 * (
                            torch.log(2.0 * torch.pi * (sigma ** 2) * dt) + (randn ** 2)
                    ).sum(dim=tuple(range(1, randn.ndim)))
                    step_old_log_probs[DataField.pos] = log_probs_atoms
                else:
                    assert self.residual_mode == self.ResidualMode.SCALE
                    res_s = residual_output[DataField.pos.name + "_s"]  # Tensor of shape (batch_size,).
                    assert res_s.shape == time.shape
                    # Tensor of shape (sum(n_atoms), 1) so that it can be broadcast to base_b shape (sum(n_atoms), 3).
                    res_s_per_atom = res_s[x_t.batch].unsqueeze(-1)
                    assert res_s_per_atom.shape[:1] == base_b.shape[:1]
                    velocity = (1.0 + res_s_per_atom) * base_b
                    randn = torch.randn_like(res_s)  # Tensor of shape (batch_size).
                    randn_per_atom = randn[x_t.batch].unsqueeze(-1)  # Tensor of shape (sum(n_atoms), 1).
                    noise = sigma * randn_per_atom * base_b
                    # Store deviation effect for PPO update: res_s * dt + noise * sqrt(dt).
                    # Effectively x_t+dt - x_t - base_b * dt, however, ignoring base_b direction.
                    # Noise is effectively one-dimensional. Shape (batch_size,).
                    step_sampled_residual_effects[DataField.pos] = res_s * dt + sigma * randn * sqrt_dt
                    # Log probability of x_t+dt given x_t for SDE.
                    # Tensor of shape (batch_size,).
                    log_prob = -0.5 * (torch.log(2.0 * torch.pi * (sigma ** 2) * dt) + (randn ** 2))
                    step_old_log_probs[DataField.pos] = log_prob

                # Euler-Maruyama update for SDE.
                x_t.pos = self.pos_corrector.correct(x_t.pos + velocity * dt + noise * sqrt_dt)

            if self.integrate_cell:
                base_b = base_model_output[DataField.cell.name + "_b"]
                sigma = self.noise_schedules[DataField.cell].noise(t)  # Scalar.
                assert sigma.ndim == 0

                if self.residual_mode == self.ResidualMode.ADDITIVE:
                    res_b = residual_output[DataField.cell.name + "_b"]
                    assert res_b.shape == base_b.shape
                    velocity = base_b + res_b
                    randn = torch.randn_like(res_b)
                    noise = sigma * randn
                    # Store deviation effect for PPO update: res_b * dt + noise * sqrt(dt).
                    # Effectively x_t+dt - x_t - base_b * dt. Shape (batch_size, 3, 3).
                    step_sampled_residual_effects[DataField.cell] = res_b * dt + noise * sqrt_dt
                    # Log probability of x_t+dt given x_t for SDE.
                    # Sum log probs over all dimensions except batch yielding tensor of shape (batch_size,).
                    step_old_log_probs[DataField.cell] = -0.5 * (
                            torch.log(2.0 * torch.pi * (sigma ** 2) * dt) + (randn ** 2)
                    ).sum(dim=tuple(range(1, randn.ndim)))
                else:
                    assert self.residual_mode == self.ResidualMode.SCALE
                    # Tensor of shape (batch_size, 1, 1) so that it can be broadcast to base_b shape (batch_size, 3, 3).
                    res_s = residual_output[DataField.cell.name + "_s"].unsqueeze(-1).unsqueeze(-1)
                    assert res_s.shape[:1] == base_b.shape[:1]
                    velocity = (1.0 + res_s) * base_b
                    randn = torch.randn_like(res_s)
                    noise = sigma * randn * base_b
                    # Store deviation effect for PPO update: res_s * dt + noise * sqrt(dt).
                    # Effectively x_t+dt - x_t - base_b * dt, however, ignoring base_b direction.
                    # Noise is effectively one-dimensional. Shape (batch_size,).
                    step_sampled_residual_effects[DataField.cell] = (
                            res_s.squeeze(dim=(1, 2)) * dt + sigma * randn.squeeze(dim=(1, 2)) * sqrt_dt)
                    # Log probability of x_t+dt given x_t for SDE.
                    # Tensor of shape (batch_size,).
                    step_old_log_probs[DataField.cell] = -0.5 * (
                            torch.log(2.0 * torch.pi * (sigma ** 2) * dt) + (randn.squeeze(dim=(1, 2)) ** 2)
                    )

                # Euler-Maruyama update for SDE.
                x_t.cell = self.cell_corrector.correct(x_t.cell + velocity * dt + noise * sqrt_dt)

            sampled_residual_effects.append(step_sampled_residual_effects)
            old_log_probs.append(step_old_log_probs)

        # Append final state.
        states.append(x_t.clone())

        # Convert to Structures.
        x_1 = x_t.to('cpu')
        structures = []
        for i in range(len(x_1.n_atoms)):
            sl = slice(x_1.ptr[i], x_1.ptr[i + 1])
            structures.append(Structure(cell=x_1.cell[i, :, :],
                                        atomic_numbers=x_1.species[sl],
                                        pos=x_1.pos[sl, :],
                                        pos_is_fractional=x_1.pos_is_fractional[i]))

        # Compute rewards for final structures.
        rewards, info_dict = self.reward.compute(structures, Reward.ComputeStage.TRAIN)
        rewards = torch.tensor(rewards, dtype=self.dtype, device=self.device)
        info_dict = {key: torch.tensor(value, dtype=self.dtype, device=self.device).mean()
                     for key, value in info_dict.items()}

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
            sampled_residual_effects=sampled_residual_effects,
            old_log_probs=old_log_probs,
            rewards=rewards,
            advantages=advantages,
            info_dict=info_dict,
        )

    def ppo_update(self, trajectory: TrajectoryData) -> tuple[dict[str, float], dict[str, float]]:
        """
        Perform PPO update by re-evaluating policy at each timestep.

        Accumulates gradients across all timesteps, then performs a single optimizer step.

        :param trajectory:
            Trajectory data from rollout.
        :type trajectory: TrajectoryData

        :return:
            Tuple of (average losses, clip fractions) for logging.
        :rtype: tuple[dict[str, float], dict[str, float]]
        """
        batch_size = len(trajectory.rewards)
        times = torch.linspace(SMALL_TIME, BIG_TIME, self.integration_time_steps, device=self.device)
        num_timesteps = len(times) - 1

        # Track losses for logging.
        total_policy_losses = {field: 0.0 for field in self.integrated_data_fields}
        total_reg_losses = {field: 0.0 for field in self.integrated_data_fields}
        total_entropy_losses = {field: 0.0 for field in self.integrated_data_fields}
        total_clip_fractions = {field: 0.0 for field in self.integrated_data_fields}

        # Zero gradients once at the start, then accumulate across all timesteps.
        opt = self.optimizers()
        opt.zero_grad()

        for t_index in trange(num_timesteps, desc="Perform PPO update", position=1, leave=False):
            t = times[t_index]
            dt = times[t_index + 1] - times[t_index]
            sqrt_dt = torch.sqrt(dt)
            time = t.repeat(batch_size)
            # Get stored state and move to device.
            x_t = trajectory.states[t_index]
            # Re-evaluate residual model with gradients.
            residual_output = self.residual_model(x_t, time)
            # Compute loss for this timestep.
            timestep_loss = torch.tensor(0.0, device=self.device)
            sigma_ref = torch.tensor(self.reference_sigma, device=self.device)  # Scalar.

            if self.integrate_pos:
                sigma = self.noise_schedules[DataField.pos].noise(t)  # Scalar.
                assert sigma.ndim == 0

                if self.residual_mode == self.ResidualMode.ADDITIVE:
                    res_b = residual_output[DataField.pos.name + "_b"]  # Shape (sum(n_atoms), 3).
                    old_log_probs_atoms = trajectory.old_log_probs[t_index][DataField.pos]  # Shape (sum(n_atoms),).
                    # This is effectively x_t+dt - x_t - base_b * dt. Shape (sum(n_atoms), 3).
                    sampled_residual_effect = trajectory.sampled_residual_effects[t_index][DataField.pos]
                    # Sum log probs over x, y, z.
                    current_log_probs_atoms = -0.5 * (
                            torch.log(2.0 * torch.pi * (sigma ** 2) * dt)
                            + ((sampled_residual_effect.detach() - res_b * dt) / (sigma * sqrt_dt)) ** 2
                    ).sum(dim=tuple(range(1, sampled_residual_effect.ndim)))

                    if self.position_normalization == self.PositionNormalization.NONE:
                        # Sum log probs over all atoms in each structure to get batch-wise log probs.
                        old_log_prob = scatter_add(old_log_probs_atoms, x_t.batch)
                        current_log_prob = scatter_add(current_log_probs_atoms, x_t.batch)
                        ratio = torch.exp(current_log_prob - old_log_prob.detach())
                        clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon)
                        clip_fraction = ((ratio < (1.0 - self.ppo_clip_epsilon))
                                         | (ratio > (1.0 + self.ppo_clip_epsilon))).float().mean()
                        # Take mean over batch.
                        policy_loss = -torch.min(ratio * trajectory.advantages,
                                                 clipped_ratio * trajectory.advantages).mean()
                    elif self.position_normalization == self.PositionNormalization.PER_ATOM_SURROGATE:
                        ratio_atoms = torch.exp(current_log_probs_atoms - old_log_probs_atoms.detach())
                        clipped_ratio_atoms = torch.clamp(ratio_atoms, 1.0 - self.ppo_clip_epsilon,
                                                          1.0 + self.ppo_clip_epsilon)
                        clip_mask_atoms = ((ratio_atoms < (1.0 - self.ppo_clip_epsilon))
                                           | (ratio_atoms > (1.0 + self.ppo_clip_epsilon))).float()
                        clip_fraction = scatter_mean(clip_mask_atoms, x_t.batch).mean()
                        # Expand advantages to per-atom.
                        advantages_atoms = trajectory.advantages[x_t.batch]
                        loss_atoms = -torch.min(ratio_atoms * advantages_atoms,
                                                clipped_ratio_atoms * advantages_atoms)
                        # Average per structure and then take mean over batch.
                        policy_loss = scatter_mean(loss_atoms, x_t.batch).mean()
                    else:
                        assert self.position_normalization == self.PositionNormalization.PER_STRUCTURE_WEIGHT
                        # Sum log probs over all atoms in each structure to get batch-wise log probs.
                        old_log_prob = scatter_add(old_log_probs_atoms, x_t.batch)
                        current_log_prob = scatter_add(current_log_probs_atoms, x_t.batch)
                        ratio = torch.exp(current_log_prob - old_log_prob.detach())
                        clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon)
                        clip_fraction = ((ratio < (1.0 - self.ppo_clip_epsilon))
                                         | (ratio > (1.0 + self.ppo_clip_epsilon))).float().mean()
                        # Weight advantages by 1 / n_atoms so that large structures do not dominate.
                        weighted_advantages = trajectory.advantages / x_t.n_atoms.to(dtype=ratio.dtype)
                        # Take mean over batch.
                        policy_loss = -torch.min(ratio * weighted_advantages,
                                                 clipped_ratio * weighted_advantages).mean()

                    # Regularization loss as KL divergence between modified and base policy.
                    # This is the KL divergence per position dimension.
                    kl_div = (torch.log(sigma_ref) - torch.log(sigma)
                              + (sigma * sigma + res_b * res_b * dt) / (2.0 * sigma_ref * sigma_ref) - 0.5)
                    # Sum over position dimensions to get per-atom KL.
                    per_atom_kl = kl_div.sum(dim=-1)
                    if self.position_normalization == self.PositionNormalization.NONE:
                        # Sum over atoms to get per-structure KL and take mean over batch.
                        reg_loss = scatter_add(per_atom_kl, x_t.batch).mean()
                    else:
                        # Average over atoms to get per-structure KL and take mean over batch.
                        reg_loss = scatter_mean(per_atom_kl, x_t.batch).mean()

                    # Entropy bonus when learning sigma.
                    if self.noise_schedules[DataField.pos].learnable():
                        # Entropy of Gaussian grows with log(sigma). Take negative since we want to maximize entropy.
                        # Multiply by position dimensions.
                        entropy_loss_per_atom = ((-0.5 * torch.log(2.0 * torch.pi * sigma * sigma) - 0.5)
                                                 * res_b.shape[-1])
                        if self.position_normalization == self.PositionNormalization.NONE:
                            # Sum over atoms to get per-structure entropy and take mean over batch.
                            entropy_loss = (entropy_loss_per_atom * x_t.n_atoms).mean()
                        else:
                            # Take mean over batch.
                            entropy_loss = entropy_loss_per_atom.mean()
                    else:
                        entropy_loss = None
                else:
                    assert self.residual_mode == self.ResidualMode.SCALE
                    res_s = residual_output[DataField.pos.name + "_s"]  # Shape (batch_size,).
                    old_log_prob = trajectory.old_log_probs[t_index][DataField.pos]  # Shape (batch_size,).
                    # This is effectively x_t+dt - x_t - base_b * dt, however, ignoring base_b direction.
                    # Noise is effectively one-dimensional. Shape (batch_size,).
                    sampled_residual_effect = trajectory.sampled_residual_effects[t_index][DataField.pos]
                    # Tensor of shape (batch_size,).
                    current_log_prob = -0.5 * (
                            torch.log(2.0 * torch.pi * (sigma ** 2) * dt)
                            + ((sampled_residual_effect.detach() - res_s * dt) / (sigma * sqrt_dt)) ** 2
                    )
                    # No special treatment of position normalization since log probs are already per-structure.
                    ratio = torch.exp(current_log_prob - old_log_prob.detach())
                    clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon)
                    clip_fraction = ((ratio < (1.0 - self.ppo_clip_epsilon))
                                     | (ratio > (1.0 + self.ppo_clip_epsilon))).float().mean()
                    # Take mean over batch.
                    policy_loss = -torch.min(ratio * trajectory.advantages,
                                             clipped_ratio * trajectory.advantages).mean()

                    # Regularization loss as KL divergence between modified and base policy.
                    # This is already the KL divergence per structure. Shape (batch_size,).
                    kl_div = (torch.log(sigma_ref) - torch.log(sigma)
                              + (sigma * sigma + res_s * res_s * dt) / (2.0 * sigma_ref * sigma_ref) - 0.5)
                    # Take mean over batch.
                    reg_loss = kl_div.mean()

                    # Entropy bonus when learning sigma.
                    if self.noise_schedules[DataField.pos].learnable():
                        # Entropy of Gaussian grows with log(sigma). Take negative since we want to maximize entropy.
                        entropy_loss = (-0.5 * torch.log(2.0 * torch.pi * sigma * sigma) - 0.5)
                    else:
                        entropy_loss = None

                # Add weighted losses.
                timestep_loss += self.relative_costs[DataField.pos.name + "_policy"] * policy_loss
                timestep_loss += self.relative_costs[DataField.pos.name + "_regularization"] * reg_loss
                if self.noise_schedules[DataField.pos].learnable():
                    timestep_loss += self.relative_costs[DataField.pos.name + "_entropy"] * entropy_loss

                # Track for logging.
                total_policy_losses[DataField.pos] += policy_loss.item()
                total_reg_losses[DataField.pos] += reg_loss.item()
                total_clip_fractions[DataField.pos] += clip_fraction.item()
                if self.noise_schedules[DataField.pos].learnable():
                    total_entropy_losses[DataField.pos] += entropy_loss.item()

            if self.integrate_cell:
                sigma = self.noise_schedules[DataField.cell].noise(t)  # Scalar.
                assert sigma.ndim == 0

                if self.residual_mode == self.ResidualMode.ADDITIVE:
                    res_b = residual_output[DataField.cell.name + "_b"]  # Shape (batch_size, 3, 3).
                    old_log_prob = trajectory.old_log_probs[t_index][DataField.cell]  # Shape (batch_size,).
                    # This is effectively x_t+dt - x_t - base_b * dt. Shape (batch_size, 3, 3).
                    sampled_residual_effect = trajectory.sampled_residual_effects[t_index][DataField.cell]

                    # Here we effectively choose the NONE normalization since the cell shape is always the same.
                    # Sum log probs over all cell dimensions except batch.
                    current_log_prob = -0.5 * (
                            torch.log(2.0 * torch.pi * (sigma ** 2) * dt)
                            + ((sampled_residual_effect.detach() - res_b * dt) / (sigma * sqrt_dt)) ** 2
                    ).sum(dim=tuple(range(1, sampled_residual_effect.ndim)))

                    ratio = torch.exp(current_log_prob - old_log_prob.detach())
                    clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon)
                    clip_fraction = ((ratio < (1.0 - self.ppo_clip_epsilon))
                                     | (ratio > (1.0 + self.ppo_clip_epsilon))).float().mean()
                    # Take mean over batch.
                    policy_loss = -torch.min(ratio * trajectory.advantages,
                                             clipped_ratio * trajectory.advantages).mean()

                    # Regularization loss as KL divergence between modified and base policy.
                    # This is the KL divergence per cell dimension.
                    kl_div = (torch.log(sigma_ref) - torch.log(sigma)
                              + (sigma * sigma + res_b * res_b * dt) / (2.0 * sigma_ref * sigma_ref) - 0.5)
                    # Sum over cell dimensions and take mean over batch.
                    reg_loss = kl_div.sum(dim=tuple(range(1, kl_div.ndim))).mean()

                    # Entropy bonus when learning sigma.
                    if self.noise_schedules[DataField.cell].learnable():
                        # Entropy of Gaussian grows with log(sigma). Take negative since we want to maximize entropy.
                        # Multiply by cell dimensions.
                        entropy_loss = ((-0.5 * torch.log(2.0 * torch.pi * sigma * sigma) - 0.5)
                                        * res_b.shape[1:].numel())
                    else:
                        entropy_loss = None
                else:
                    assert self.residual_mode == self.ResidualMode.SCALE
                    res_s = residual_output[DataField.cell.name + "_s"]  # Shape (batch_size,).
                    old_log_prob = trajectory.old_log_probs[t_index][DataField.cell]  # Shape (batch_size,).
                    # This is effectively x_t+dt - x_t - base_b * dt, however, ignoring base_b direction.
                    # Noise is effectively one-dimensional. Shape (batch_size,).
                    sampled_residual_effect = trajectory.sampled_residual_effects[t_index][DataField.cell]
                    # Tensor of shape (batch_size,).
                    current_log_prob = -0.5 * (
                            torch.log(2.0 * torch.pi * (sigma ** 2) * dt)
                            + ((sampled_residual_effect.detach() - res_s * dt) / (sigma * sqrt_dt)) ** 2
                    )

                    ratio = torch.exp(current_log_prob - old_log_prob.detach())
                    clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon)
                    clip_fraction = ((ratio < (1.0 - self.ppo_clip_epsilon))
                                     | (ratio > (1.0 + self.ppo_clip_epsilon))).float().mean()
                    # Take mean over batch.
                    policy_loss = -torch.min(ratio * trajectory.advantages,
                                             clipped_ratio * trajectory.advantages).mean()

                    # Regularization loss as KL divergence between modified and base policy.
                    # This is already the KL divergence per structure. Shape (batch_size,).
                    kl_div = (torch.log(sigma_ref) - torch.log(sigma)
                              + (sigma * sigma + res_s * res_s * dt) / (2.0 * sigma_ref * sigma_ref) - 0.5)
                    # Take mean over batch.
                    reg_loss = kl_div.mean()

                    # Entropy bonus when learning sigma.
                    if self.noise_schedules[DataField.cell].learnable():
                        # Entropy of Gaussian grows with log(sigma). Take negative since we want to maximize entropy.
                        entropy_loss = (-0.5 * torch.log(2.0 * torch.pi * sigma * sigma) - 0.5)
                    else:
                        entropy_loss = None

                # Add weighted losses.
                timestep_loss += self.relative_costs[DataField.cell.name + "_policy"] * policy_loss
                timestep_loss += self.relative_costs[DataField.cell.name + "_regularization"] * reg_loss
                if self.noise_schedules[DataField.cell].learnable():
                    timestep_loss += self.relative_costs[DataField.cell.name + "_entropy"] * entropy_loss

                # Track for logging.
                total_policy_losses[DataField.cell] += policy_loss.item()
                total_reg_losses[DataField.cell] += reg_loss.item()
                total_clip_fractions[DataField.cell] += clip_fraction.item()
                if self.noise_schedules[DataField.cell].learnable():
                    total_entropy_losses[DataField.cell] += entropy_loss.item()

            # Scale loss by 1/num_timesteps and accumulate gradients.
            scaled_loss = timestep_loss / num_timesteps
            self.manual_backward(scaled_loss)

        # After all timesteps: clip gradients and perform single optimizer step.
        if self.gradient_clip_val is not None:
            # noinspection PyTypeChecker
            self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_val,
                                gradient_clip_algorithm=self.gradient_clip_algorithm)
        opt.step()

        # Return average losses for logging.
        losses = {}
        clip_fractions = {}
        for field in self.integrated_data_fields:
            losses[field.name + "_policy"] = total_policy_losses[field] / num_timesteps
            losses[field.name + "_regularization"] = total_reg_losses[field] / num_timesteps
            if self.noise_schedules[field].learnable():
                losses[field.name + "_entropy"] = total_entropy_losses[field] / num_timesteps
            clip_fractions[field.name + "_clip_fraction"] = total_clip_fractions[field] / num_timesteps

        return losses, clip_fractions

    @torch.no_grad()
    def integrate(self, x_0: OMGData) -> OMGData:
        """Integrate for validation/prediction."""
        base_model = base_modules["model"].model
        assert base_model is not None
        batch_size = len(x_0.n_atoms)
        times = torch.linspace(SMALL_TIME, BIG_TIME, self.integration_time_steps, device=self.device)
        x_t = x_0.clone()

        # Integrate over time with residuals.
        for t_index in trange(1, len(times), desc="Integrating with residuals", position=1, leave=False):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            sqrt_dt = torch.sqrt(dt)
            time = t.repeat(batch_size)
            base_model_output = base_model(x_t, time)
            residual_output = self.residual_model(x_t, time)
            if self.integrate_pos:
                base_b = base_model_output[DataField.pos.name + "_b"]
                sigma = self.noise_schedules[DataField.pos].noise(t)  # Scalar.
                assert sigma.ndim == 0
                if self.residual_mode == self.ResidualMode.ADDITIVE:
                    res_b = residual_output[DataField.pos.name + "_b"]
                    assert res_b.shape == base_b.shape
                    velocity = base_b + res_b
                    noise = sigma * torch.randn_like(res_b)
                else:
                    assert self.residual_mode == self.ResidualMode.SCALE
                    res_s = residual_output[DataField.pos.name + "_s"]  # Tensor of shape (batch_size,).
                    assert res_s.shape == time.shape
                    # Tensor of shape (sum(n_atoms), 1) so that it can be broadcast to base_b shape (sum(n_atoms), 3).
                    res_s_per_atom = res_s[x_t.batch].unsqueeze(-1)
                    assert res_s_per_atom.shape[:1] == base_b.shape[:1]
                    velocity = (1.0 + res_s_per_atom) * base_b
                    randn = torch.randn_like(res_s)  # Tensor of shape (batch_size,).
                    randn_per_atom = randn[x_t.batch].unsqueeze(-1)  # Tensor of shape (sum(n_atoms), 1).
                    noise = sigma * randn_per_atom * base_b
                # Euler-Maruyama update for SDE.
                x_t.pos = self.pos_corrector.correct(x_t.pos + velocity * dt + noise * sqrt_dt)
            if self.integrate_cell:
                base_b = base_model_output[DataField.cell.name + "_b"]
                sigma = self.noise_schedules[DataField.cell].noise(t)  # Scalar.
                assert sigma.ndim == 0
                if self.residual_mode == self.ResidualMode.ADDITIVE:
                    res_b = residual_output[DataField.cell.name + "_b"]
                    assert res_b.shape == base_b.shape
                    velocity = base_b + res_b
                    noise = sigma * torch.randn_like(res_b)
                else:
                    assert self.residual_mode == self.ResidualMode.SCALE
                    # Tensor of shape (batch_size, 1, 1) so that it can be broadcast to base_b shape (batch_size, 3, 3).
                    res_s = residual_output[DataField.cell.name + "_s"].unsqueeze(-1).unsqueeze(-1)
                    assert res_s.shape[:1] == base_b.shape[:1]
                    velocity = (1.0 + res_s) * base_b
                    noise = sigma * torch.randn_like(res_s) * base_b
                # Euler-Maruyama update for SDE.
                x_t.cell = self.cell_corrector.correct(x_t.cell + velocity * dt + noise * sqrt_dt)
        return x_t

    def training_step(self, batch: OMGData, batch_idx: int) -> None:
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

        # Rollout without gradients.
        trajectory = self.rollout(x_0)

        # PPO update with gradients.
        # Each call to ppo_update performs one optimizer step.
        # Multiple epochs = multiple passes over the same trajectory.
        all_losses = {}
        all_clip_fractions = {}
        for epoch in range(self.ppo_epochs):
            losses, clip_fractions = self.ppo_update(trajectory)
            # Accumulate for logging (losses are already floats, not tensors).
            for key, value in losses.items():
                if key not in all_losses:
                    all_losses[key] = 0.0
                all_losses[key] += value
            for key, value in clip_fractions.items():
                if key not in all_clip_fractions:
                    all_clip_fractions[key] = 0.0
                all_clip_fractions[key] += value

        # Average losses over PPO epochs for logging.
        for key in all_losses:
            all_losses[key] /= self.ppo_epochs
        for key in all_clip_fractions:
            all_clip_fractions[key] /= self.ppo_epochs

        # Compute total loss for logging (not used for backprop since we use manual optimization).
        total_loss = sum(self.relative_costs[key] * all_losses[key] for key in all_losses)

        # Compute diagnostics.
        rewards = trajectory.rewards
        group_reward_stds = []
        for i in range(len(rewards) // self.grpo_group_size):
            sl = slice(i * self.grpo_group_size, (i + 1) * self.grpo_group_size)
            group_reward_stds.append(rewards[sl].std(unbiased=False).item())
        within_group_std = torch.tensor(group_reward_stds).mean()

        self.log_dict(all_losses, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch))
        self.log("loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log_dict(all_clip_fractions, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                      batch_size=len(batch))
        self.log("reward_mean", rewards.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log("reward_std", rewards.std(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log("reward_std_within_group", within_group_std, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=len(batch))
        self.log_dict(trajectory.info_dict, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                      batch_size=len(batch))

        for field in self.integrated_data_fields:
            if self.noise_schedules[field].learnable():
                times = torch.linspace(SMALL_TIME, BIG_TIME, self.integration_time_steps, device=self.device)
                sigmas = self.noise_schedules[field].noise(times)
                self.log(f"sigma_{field.name}_mean", sigmas.mean(), on_step=False, on_epoch=True, prog_bar=True,
                         sync_dist=True, batch_size=len(batch))
                self.log(f"sigma_{field.name}_min", sigmas.min(), on_step=False, on_epoch=True, sync_dist=True,
                         batch_size=len(batch))
                self.log(f"sigma_{field.name}_max", sigmas.max(), on_step=False, on_epoch=True, sync_dist=True,
                         batch_size=len(batch))

    def validation_step(self, batch: OMGData, batch_idx: int) -> None:
        # Sample initial structures independently.  # TODO: VALIDATE that NOISING LEAVES METRIC IN TACT?
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

        x_1 = self.integrate(x_0)

        # Store initial and final structures as XYZ files.
        filename = (Path(self.generation_xyz_filename) if self.generation_xyz_filename is not None
                    else Path(f"{strftime('%Y%m%d-%H%M%S')}.xyz"))
        init_filename = filename.with_stem(filename.stem + "_init")
        xyz_saver(x_0.to('cpu'), init_filename)
        xyz_saver(x_1.to('cpu'), filename)

        return x_1
