from copy import deepcopy
from typing import Optional, Sequence
import warnings
import torch
from torch_scatter import scatter_add, scatter_mean
from tqdm import trange
from omg.datamodule import OMGData
from omg.globals import SMALL_TIME, BIG_TIME
from omg.si import DifferentialEquationType, SingleStochasticInterpolantOS
from omg.utils import DataField
from omg.omg_irl import base_modules, NoiseSchedule, Reward
from .abstracts import TrajectoryData, OMGIRLLightningAbstract
from .omg_irl_velocity import PositionNormalization


class OMGIRLScore(OMGIRLLightningAbstract):
    """
    Score-based Open Materials Generation with Inference-time Reinforcement Learning (OMatG-IRL) using
    group-relative policy optimization (GRPO) and proximal policy optimization (PPO) within the PyTorch Lightning
    framework.

    This class reinforces a velocity field and denoiser of a base OMatG model.

    The following Euler-Maruyama update for the variable x_t based on the velocity field b(x_t, t) and denoiser
    eta(x_t, t) already yields exploration:
    x_t+dt = x_t + b(x_t, t) * dt - 0.5 * sigma(t)^2 / gamma(t) * eta(x_t, t) * dt + sigma(t) * randn * sqrt(dt),
    where randn is a standard normal random variable with the same dimension as b(x_t, t), sigma(t) is a noise schedule,
    and gamma(t) is a scaling function.

    The GRPO framework is used to reinforce the velocity fields and denoisers for the atomic positions and the lattice
    vectors, given that they are integrated by the base model. Optionally, one can switch off the reinforcement for each
    of these data fields. In this case, the base model's velocity field is used without modification, and no noise is
    added during integration.

    For every data field that is reinforced, the loss consists of a policy loss, a regularization loss, and a
    distillation loss. The relative costs of these losses should be specified with the keys '<data_field>_policy',
    '<data_field>_regularization', and '<data_field>_distillation' in the relative_costs dictionary.

    If the velocity field of the atomic positions are reinforced, one should consider that there can be variable numbers
    of atoms per structure. To prevent policy updates from being biased toward larger crystals, different position
    normalization modes can be selected.

    :param reward:
        Reward function to evaluate the generated structures.
    :type reward: Reward
    :param noise_schedules:
        Dictionary mapping data field names to noise schedules sigma(t).
        The noise schedules should not be learnable.
    :type noise_schedules: dict[str, NoiseSchedule]
    :param relative_costs:
        Dictionary mapping cost types to their relative costs for each data field.
    :type relative_costs: dict[str, float]
    :param normalize_relative_costs:
        If True, all relative costs are normalized so that they sum to 1.
        Defaults to True.
    :type normalize_relative_costs: bool
    :param disable_fields:
        Sequence of data field names for which to disable reinforcement of the velocity field.
        For these data fields, the base model velocity is used without modification, and no noise is added.
        Defaults to an empty sequence.
    :type disable_fields: Sequence[str]
    :param position_normalization:
        Position normalization mode for the atomic positions.
        Options are "none", "per_structure_weight", and "per_atom_surrogate".
        Defaults to "per_atom_surrogate".
    :type position_normalization: str
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
        If True, enables progress bars during reward computation, rollout, PPO update, and integration.
        Defaults to True.
    :type enable_progress_bar: bool

    :raises ValueError:
        If noise_schedules contains an invalid data field name.
        If any noise schedule is learnable.
        If disable_fields contains an invalid data field name.
        If any relative cost is negative.
        If all relative costs are zero.
        If any relative cost key does not follow the format '<data_field>_<cost_type>', where <cost_type> is one of
        'policy', 'regularization', or 'distillation'.
        If integrating species is enabled.
        If an integrated data field is missing a noise schedule or relative costs.
        If an integrated data field is not disabled but does not use an SDE integrated interpolant.
        If velocity prediction is not enabled for position or cell data fields when using SingleStochasticInterpolantOS.
        If position normalization is invalid.
    """

    def __init__(self, reward: Reward, noise_schedules: dict[str, NoiseSchedule], relative_costs: dict[str, float],
                 normalize_relative_costs: bool = True, disable_fields: Sequence[str] = (),
                 position_normalization: str = "per_atom_surrogate", grpo_group_size: int = 32,
                 grpo_num_groups: int = 16, grpo_share_x_0: bool = True, ppo_clip_epsilon: float = 0.2,
                 ppo_epochs: int = 1, gradient_clip_val: Optional[float] = 1.0, gradient_clip_algorithm: str = "norm",
                 generation_xyz_filename: Optional[str] = None, validation_xyz_filename: Optional[str] = None,
                 enable_progress_bar: bool = True) -> None:
        # noinspection PyTypeChecker
        super().__init__(reward=reward, grpo_group_size=grpo_group_size, grpo_num_groups=grpo_num_groups,
                         grpo_share_x_0=grpo_share_x_0, ppo_clip_epsilon=ppo_clip_epsilon, ppo_epochs=ppo_epochs,
                         gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm,
                         generation_xyz_filename=generation_xyz_filename,
                         validation_xyz_filename=validation_xyz_filename, enable_progress_bar=enable_progress_bar)

        try:
            self.position_normalization: PositionNormalization = PositionNormalization[position_normalization.upper()]
        except KeyError:
            raise ValueError(f"Invalid position normalization: {position_normalization}")

        if self.position_normalization != PositionNormalization.NONE:
            if (self.integrate_pos and DataField.pos in self.disable_fields) or not self.integrate_pos:
                warnings.warn(f"Position normalization '{self.position_normalization.name}' is enabled, "
                              f"but position integration is disabled. Position normalization is ignored.")

        # Copy over the OMatG model so that it can be reinforced.
        base_model = base_modules["model"]
        assert base_model is not None
        self.model = deepcopy(base_model.model)
        self.model.unfreeze()  # Base model was frozen in base class.

        try:
            self.noise_schedules: dict[DataField, NoiseSchedule] = {
                DataField[field.lower()]: ns for field, ns in noise_schedules.items()}
        except KeyError as e:
            raise ValueError(f"Invalid data field in noise_schedules: {e}") from e
        if any(schedule.learnable() for schedule in self.noise_schedules.values()):
            raise ValueError("Noise schedules must not be learnable.")

        try:
            self.disable_fields: set[DataField] = set(DataField[field.lower()] for field in disable_fields)
        except KeyError as e:
            raise ValueError(f"Invalid data field in disable_fields: {e}") from e

        if not all(cost >= 0.0 for cost in relative_costs.values()):
            raise ValueError("All relative costs must be non-negative.")
        sum_costs = sum(relative_costs.values())
        if sum_costs == 0.0:
            raise ValueError("At least one relative cost must be positive.")
        self.normalize_relative_costs = normalize_relative_costs
        if self.normalize_relative_costs:
            self.relative_costs = {key: value / sum_costs for key, value in relative_costs.items()}
            assert abs(sum(self.relative_costs.values()) - 1.0) < 1e-10
        else:
            self.relative_costs = relative_costs
        for field in self.relative_costs.keys():
            split_field = field.split("_")
            if len(split_field) != 2:
                raise ValueError(f"Invalid relative cost key: {field}. Expected format '<data_field>_<cost_type>'.")
            field_name, cost_type = split_field
            try:
                DataField[field_name.lower()]
            except KeyError as e:
                raise ValueError(f"Invalid data field in relative costs: {field_name}") from e
            if cost_type not in {"policy", "regularization", "distillation"}:
                raise ValueError(f"Invalid cost type in relative costs: {cost_type}. "
                                 f"Expected 'policy', 'regularization', or 'distillation'.")

        if self.integrate_species:
            raise ValueError("Integrating species is not supported in OMGIRLScore.")

        for integrated_data_field in self.integrated_data_fields:
            if integrated_data_field in self.disable_fields:
                if integrated_data_field in self.noise_schedules:
                    warnings.warn(f"Noise schedule for disabled integrated data field "
                                  f"{integrated_data_field.name} will be ignored.")
                if integrated_data_field.name + "_policy" in self.relative_costs:
                    warnings.warn(f"Relative policy cost for disabled integrated data field "
                                  f"{integrated_data_field.name} will be ignored.")
                if integrated_data_field.name + "_regularization" in self.relative_costs:
                    warnings.warn(f"Relative regularization cost for disabled integrated data field "
                                  f"{integrated_data_field.name} will be ignored.")
                if integrated_data_field.name + "_distillation" in self.relative_costs:
                    warnings.warn(f"Relative distillation cost for disabled integrated data field "
                                  f"{integrated_data_field.name} will be ignored.")
            else:
                if integrated_data_field not in self.noise_schedules:
                    raise ValueError(f"Missing noise schedule for integrated data field {integrated_data_field.name}.")
                if integrated_data_field.name + "_policy" not in self.relative_costs:
                    raise ValueError(f"Missing relative policy cost for integrated data field "
                                     f"{integrated_data_field.name}.")
                if integrated_data_field.name + "_regularization" not in self.relative_costs:
                    raise ValueError(f"Missing relative regularization cost for integrated data field "
                                     f"{integrated_data_field.name}.")
                if integrated_data_field.name + "_distillation" not in self.relative_costs:
                    raise ValueError(f"Missing relative distillation cost for integrated data field "
                                     f"{integrated_data_field.name}.")

        for fixed_data_field in self.fixed_data_fields:
            if fixed_data_field in self.disable_fields:
                warnings.warn(f"Disabled field {fixed_data_field.name} is not an integrated data field and will be "
                              f"ignored.")
            if fixed_data_field in self.noise_schedules:
                warnings.warn(f"Noise schedule for fixed data field {fixed_data_field.name} will be ignored.")

        self.gammas = {}
        if self.integrate_pos:
            if DataField.pos in self.disable_fields:
                if self.pos_interpolant.differential_equation_type != DifferentialEquationType.ODE:
                    warnings.warn("OMGIRLScore will ignore predicted scores for position data field and only work with "
                                  "the velocity field.")
            else:
                if self.pos_interpolant.differential_equation_type != DifferentialEquationType.SDE:
                    raise ValueError("OMGIRLScore requires scores for position data field to be predicted (i.e., an"
                                     "SDE integrated interpolant).")
                self.gammas[DataField.pos] = self.pos_interpolant.gamma
                if self.gammas[DataField.pos] is None:
                    raise ValueError("OMGIRLScore requires gamma to be specified for position data field.")
            if self.pos_interpolant.velocity_annealing_factor != 0.0:
                warnings.warn("OMGIRLScale will ignore velocity annealing for position data field.")
            if isinstance(self.pos_interpolant, SingleStochasticInterpolantOS):
                if not self.pos_interpolant.predict_velocity:
                    raise ValueError("OMGIRLScore requires velocity prediction for position data field when using "
                                     "SingleStochasticInterpolantOS.")

        if self.integrate_cell:
            if DataField.cell in self.disable_fields:
                if self.cell_interpolant.differential_equation_type != DifferentialEquationType.ODE:
                    warnings.warn("OMGIRLScore will ignore predicted scores for cell data field and only work with "
                                  "the velocity field.")
            else:
                if self.cell_interpolant.differential_equation_type != DifferentialEquationType.SDE:
                    raise ValueError("OMGIRLScore requires scores for cell data field to be predicted (i.e., an SDE "
                                     "integrated interpolant).")
                self.gammas[DataField.cell] = self.cell_interpolant.gamma
                if self.gammas[DataField.cell] is None:
                    raise ValueError("OMGIRLScore requires gamma to be specified for cell data field.")
            if self.cell_interpolant.velocity_annealing_factor != 0.0:
                warnings.warn("OMGIRLScale will ignore velocity annealing for cell data field.")
            if isinstance(self.cell_interpolant, SingleStochasticInterpolantOS):
                if not self.cell_interpolant.predict_velocity:
                    raise ValueError("OMGIRLScale requires velocity prediction for cell data field when using "
                                     "SingleStochasticInterpolantOS.")

    def _rollout(self, x_0: OMGData) -> TrajectoryData:
        """
        Generate a trajectory rollout starting from initial structures x_0 without gradients, storing data for PPO
        updates.

        This method is called in no_grad context by the public rollout method.

        :param x_0:
            Initial structures at time 0 sampled from p_0.
        :type x_0: OMGData

        :return:
            Trajectory data containing states, sampled actions, old log probabilities, base model outputs, rewards,
            advantages, and info dictionary for logging.
        :rtype: TrajectoryData
        """
        base_model = base_modules["model"].model
        assert base_model is not None
        batch_size = len(x_0.n_atoms)
        # noinspection PyTypeChecker
        times = torch.linspace(SMALL_TIME, BIG_TIME, self.integration_time_steps, device=self.device)
        x_t = x_0.clone()
        states = []
        old_log_probs = []
        sampled_actions = []
        base_model_outputs = []
        ratios_total = {field: torch.tensor(0.0, device=self.device)
                        for field in self.integrated_data_fields if field not in self.disable_fields}

        # Integrate over time with reinforced model.
        for t_index in trange(1, len(times), desc="Rollout with reinforced model", position=1, leave=False,
                              disable=not self.enable_progress_bar):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            sqrt_dt = torch.sqrt(dt)
            time = t.repeat(batch_size)
            # Base model output necessary for KL regularization.
            base_model_output = base_model(x_t, time)
            model_output = self.model(x_t, time)

            states.append(x_t.clone())
            step_sampled_actions = {}
            step_old_log_probs = {}
            step_base_model_outputs = {}

            if self.integrate_pos:
                base_b = base_model_output[DataField.pos.name + "_b"]

                if DataField.pos in self.disable_fields:
                    # Take a standard Euler step (i.e., Euler-Maruyama without noise).
                    drift = base_b
                    noise = torch.zeros_like(base_b)
                else:
                    sigma = self.noise_schedules[DataField.pos].noise(t)  # Scalar.
                    gamma = self.gammas[DataField.pos].gamma(t)  # Scalar.
                    assert sigma.ndim == gamma.ndim == 0
                    base_eta = base_model_output[DataField.pos.name + "_eta"]  # Shape (sum(n_atoms), 3).
                    velocity = model_output[DataField.pos.name + "_b"]  # Shape (sum(n_atoms), 3).
                    eta = model_output[DataField.pos.name + "_eta"]  # Shape (sum(n_atoms), 3).
                    assert velocity.shape == base_eta.shape == eta.shape == base_b.shape
                    drift = velocity - 0.5 * sigma * sigma / gamma * eta
                    randn = torch.randn_like(velocity)
                    noise = sigma * randn
                    # Store base model outputs.
                    step_base_model_outputs[DataField.pos] = (base_b, base_eta)
                    # Store action (basically x_t+dt - x_t). Shape (sum(n_atoms), 3).
                    step_sampled_actions[DataField.pos] = drift * dt + noise * sqrt_dt
                    # Log probability of x_t+dt given x_t for SDE.
                    # Sum log probs over x, y, and z yielding tensor of shape (sum(n_atoms),).
                    log_probs_atoms = -0.5 * (
                            torch.log(2.0 * torch.pi * (sigma ** 2) * dt) + (randn ** 2)
                    ).sum(dim=tuple(range(1, randn.ndim)))
                    step_old_log_probs[DataField.pos] = log_probs_atoms
                    # Norm over x, y, z.
                    diff_b = velocity - base_b
                    base_norm_atoms = torch.linalg.norm(base_b, dim=-1)
                    diff_b_norm_atoms = torch.linalg.norm(diff_b, dim=-1)
                    # Mean over atoms.
                    base_norm_struct = scatter_mean(base_norm_atoms, x_t.batch)
                    diff_b_norm_struct = scatter_mean(diff_b_norm_atoms, x_t.batch)
                    # Mean over batch.
                    ratios_total[DataField.pos] += (diff_b_norm_struct / (base_norm_struct + 1.0e-8)).mean()

                # Euler-Maruyama update for SDE.
                x_t.pos = self.pos_corrector.correct(x_t.pos + drift * dt + noise * sqrt_dt)

            if self.integrate_cell:
                base_b = base_model_output[DataField.cell.name + "_b"]

                if DataField.cell in self.disable_fields:
                    # Take a standard Euler step (i.e., Euler-Maruyama without noise).
                    drift = base_b
                    noise = torch.zeros_like(base_b)
                else:
                    sigma = self.noise_schedules[DataField.cell].noise(t)  # Scalar.
                    gamma = self.gammas[DataField.cell].gamma(t)  # Scalar.
                    assert sigma.ndim == gamma.ndim == 0
                    base_eta = base_model_output[DataField.cell.name + "_eta"]  # Shape (batch_size, 3, 3).
                    velocity = model_output[DataField.cell.name + "_b"]  # Shape (batch_size, 3, 3).
                    eta = model_output[DataField.cell.name + "_eta"]  # Shape (batch_size, 3, 3).
                    assert velocity.shape == base_eta.shape == eta.shape == base_b.shape
                    drift = velocity - 0.5 * sigma * sigma / gamma * eta
                    randn = torch.randn_like(velocity)
                    noise = sigma * randn
                    # Store base model outputs.
                    step_base_model_outputs[DataField.cell] = (base_b, base_eta)
                    # Store action (basically x_t+dt - x_t). Shape (batch_size, 3, 3).
                    step_sampled_actions[DataField.cell] = drift * dt + noise * sqrt_dt
                    # Log probability of x_t+dt given x_t for SDE.
                    # Sum log probs over all dimensions except batch yielding tensor of shape (batch_size,).
                    step_old_log_probs[DataField.cell] = -0.5 * (
                            torch.log(2.0 * torch.pi * (sigma ** 2) * dt) + (randn ** 2)
                    ).sum(dim=tuple(range(1, randn.ndim)))
                    # Norm over cell dimensions.
                    diff_b = velocity - base_b
                    base_norm_struct = torch.linalg.norm(base_b.reshape(base_b.shape[0], -1), dim=-1)
                    diff_b_norm_struct = torch.linalg.norm(diff_b.reshape(diff_b.shape[0], -1), dim=-1)
                    # Mean over batch.
                    ratios_total[DataField.cell] += (diff_b_norm_struct / (base_norm_struct + 1.0e-8)).mean()

                # Euler-Maruyama update for SDE.
                x_t.cell = self.cell_corrector.correct(x_t.cell + drift * dt + noise * sqrt_dt)

            sampled_actions.append(step_sampled_actions)
            old_log_probs.append(step_old_log_probs)
            base_model_outputs.append(step_base_model_outputs)

        # Append final state.
        states.append(x_t.clone())

        # Average over time steps.
        info_dict = {f"{key.name}_ratio": ratio / (len(times) - 1) for key, ratio in ratios_total.items()}

        return self._create_trajectory_data(states, sampled_actions, old_log_probs, base_model_outputs, info_dict)

    def ppo_update(self, trajectory: TrajectoryData) -> tuple[float, dict[str, float]]:
        """
        Perform a PPO update using the provided trajectory data.

        This method takes the trajectory data generated during the rollout and performs one PPO update, returning the
        computed total loss and an additional dictionary for logging.

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
        batch_size = len(trajectory.rewards)
        # noinspection PyTypeChecker
        times = torch.linspace(SMALL_TIME, BIG_TIME, self.integration_time_steps, device=self.device)
        num_timesteps = len(times) - 1

        # Track losses for logging (only for non-disabled integrated fields).
        active_fields = [field for field in self.integrated_data_fields if field not in self.disable_fields]
        total_loss = 0.0
        total_policy_losses = {field: 0.0 for field in active_fields}
        total_reg_losses = {field: 0.0 for field in active_fields}
        total_distillation_losses = {field: 0.0 for field in active_fields}
        total_clip_fractions = {field: 0.0 for field in active_fields}

        # Zero gradients once at the start, then accumulate across all timesteps.
        opt = self.optimizers()
        opt.zero_grad()

        for t_index in trange(num_timesteps, desc="Perform PPO update", position=1, leave=False,
                              disable=not self.enable_progress_bar):
            t = times[t_index]
            dt = times[t_index + 1] - times[t_index]
            sqrt_dt = torch.sqrt(dt)
            time = t.repeat(batch_size)
            x_t = trajectory.states[t_index]
            # Re-evaluate model with gradients.
            model_output = self.model(x_t, time)
            timestep_loss = torch.tensor(0.0, device=self.device)

            if self.integrate_pos and DataField.pos not in self.disable_fields:
                sigma = self.noise_schedules[DataField.pos].noise(t)  # Scalar.
                gamma = self.gammas[DataField.pos].gamma(t)  # Scalar.
                assert sigma.ndim == gamma.ndim == 0
                velocity = model_output[DataField.pos.name + "_b"]  # Shape (sum(n_atoms), 3).
                eta = model_output[DataField.pos.name + "_eta"]  # Shape (sum(n_atoms), 3).
                base_b, base_eta = trajectory.base_model_outputs[t_index][DataField.pos]
                drift = velocity - 0.5 * sigma * sigma / gamma * eta
                drift_base = base_b.detach() - 0.5 * sigma * sigma / gamma * base_eta.detach()

                old_log_probs_atoms = trajectory.old_log_probs[t_index][DataField.pos]  # Shape (sum(n_atoms),).
                # Action basically stores x_t+dt - x_t. Shape (sum(n_atoms), 3).
                sampled_action = trajectory.sampled_actions[t_index][DataField.pos]
                # Sum log probs over x, y, z.
                current_log_probs_atoms = -0.5 * (
                        torch.log(2.0 * torch.pi * (sigma ** 2) * dt)
                        + ((sampled_action.detach() - drift * dt) / (sigma * sqrt_dt)) ** 2
                ).sum(dim=tuple(range(1, sampled_action.ndim)))
                if self.position_normalization == PositionNormalization.NONE:
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
                elif self.position_normalization == PositionNormalization.PER_ATOM_SURROGATE:
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
                    assert self.position_normalization == PositionNormalization.PER_STRUCTURE_WEIGHT
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
                kl_div = (drift - drift_base.detach()) ** 2 * dt / (2.0 * sigma * sigma)
                # Sum over position dimensions to get per-atom KL.
                per_atom_kl = kl_div.sum(dim=-1)
                if self.position_normalization == PositionNormalization.NONE:
                    # Sum over atoms to get per-structure KL and take mean over batch.
                    reg_loss = scatter_add(per_atom_kl, x_t.batch).mean()
                else:
                    # Average over atoms to get per-structure KL and take mean over batch.
                    reg_loss = scatter_mean(per_atom_kl, x_t.batch).mean()

                diff_eta = (eta - base_eta.detach()) ** 2  # Shape (sum(n_atoms), 3).
                # Sum over x, y, z. Shape (sum(n_atoms),).
                distill_per_atom = diff_eta.sum(dim=-1)
                # Mean over structure and then mean over batch.
                distillation_loss = scatter_mean(distill_per_atom, x_t.batch).mean()

                # Add weighted losses.
                timestep_loss += self.relative_costs[DataField.pos.name + "_policy"] * policy_loss
                timestep_loss += self.relative_costs[DataField.pos.name + "_regularization"] * reg_loss
                timestep_loss += self.relative_costs[DataField.pos.name + "_distillation"] * distillation_loss

                # Track for logging.
                total_policy_losses[DataField.pos] += policy_loss.item()
                total_reg_losses[DataField.pos] += reg_loss.item()
                total_clip_fractions[DataField.pos] += clip_fraction.item()
                total_distillation_losses[DataField.pos] += distillation_loss.item()

            if self.integrate_cell and DataField.cell not in self.disable_fields:
                sigma = self.noise_schedules[DataField.cell].noise(t)  # Scalar.
                gamma = self.gammas[DataField.cell].gamma(t)  # Scalar.
                assert sigma.ndim == gamma.ndim == 0
                velocity = model_output[DataField.cell.name + "_b"]  # Shape (batch_size, 3, 3).
                eta = model_output[DataField.cell.name + "_eta"]  # Shape (batch_size, 3, 3).
                base_b, base_eta = trajectory.base_model_outputs[t_index][DataField.cell]
                drift = velocity - 0.5 * sigma * sigma / gamma * eta
                drift_base = base_b.detach() - 0.5 * sigma * sigma / gamma * base_eta.detach()

                old_log_prob = trajectory.old_log_probs[t_index][DataField.cell]  # Shape (batch_size,).
                # Action basically stores x_t+dt - x_t. Shape (batch_size, 3, 3).
                sampled_action = trajectory.sampled_actions[t_index][DataField.cell]
                # Here we effectively choose the NONE normalization since the cell shape is always the same.
                # Sum log probs over all cell dimensions except batch.
                current_log_prob = -0.5 * (
                        torch.log(2.0 * torch.pi * (sigma ** 2) * dt)
                        + ((sampled_action.detach() - drift * dt) / (sigma * sqrt_dt)) ** 2
                ).sum(dim=tuple(range(1, sampled_action.ndim)))
                ratio = torch.exp(current_log_prob - old_log_prob.detach())
                clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon)
                clip_fraction = ((ratio < (1.0 - self.ppo_clip_epsilon))
                                 | (ratio > (1.0 + self.ppo_clip_epsilon))).float().mean()
                # Take mean over batch.
                policy_loss = -torch.min(ratio * trajectory.advantages,
                                         clipped_ratio * trajectory.advantages).mean()

                # Regularization loss as KL divergence between modified and base policy.
                # This is the KL divergence per cell dimension.
                kl_div = (drift - drift_base.detach()) ** 2 * dt / (2.0 * sigma * sigma)
                # Sum over cell dimensions and take mean over batch.
                reg_loss = kl_div.sum(dim=tuple(range(1, kl_div.ndim))).mean()

                diff_eta = (eta - base_eta.detach()) ** 2  # Shape (batch_size, 3, 3).
                # Sum over cell dimensions and mean over batch.
                distillation_loss = diff_eta.sum(dim=tuple(range(1, diff_eta.ndim))).mean()

                # Add weighted losses.
                timestep_loss += self.relative_costs[DataField.cell.name + "_policy"] * policy_loss
                timestep_loss += self.relative_costs[DataField.cell.name + "_regularization"] * reg_loss
                timestep_loss += self.relative_costs[DataField.cell.name + "_distillation"] * distillation_loss

                # Track for logging.
                total_policy_losses[DataField.cell] += policy_loss.item()
                total_reg_losses[DataField.cell] += reg_loss.item()
                total_clip_fractions[DataField.cell] += clip_fraction.item()
                total_distillation_losses[DataField.cell] += distillation_loss.item()

            # Scale loss by 1/num_timesteps and accumulate gradients.
            scaled_loss = timestep_loss / num_timesteps
            self.manual_backward(scaled_loss)
            total_loss += scaled_loss.item()

        # After all timesteps: clip gradients and perform single optimizer step.
        if self.gradient_clip_val is not None:
            # noinspection PyTypeChecker
            self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_val,
                                gradient_clip_algorithm=self.gradient_clip_algorithm)
        opt.step()

        # Return average losses and clip fractions for logging.
        logging_info = {}
        for field in active_fields:
            logging_info[field.name + "_loss_policy"] = total_policy_losses[field] / num_timesteps
            logging_info[field.name + "_loss_regularization"] = total_reg_losses[field] / num_timesteps
            logging_info[field.name + "_loss_distillation"] = total_distillation_losses[field] / num_timesteps
            logging_info[field.name + "_clip_fraction"] = total_clip_fractions[field] / num_timesteps

        return total_loss, logging_info

    def _integrate(self, x_0: OMGData) -> OMGData:
        """
        Integrate the reinforced model starting from initial structures x_0.

        This method is called in no_grad context by the public integrate method.

        :param x_0:
            Initial structures at time 0 sampled from p_0.
        :type x_0: OMGData

        :return:
            Integrated structures at time 1 after applying the reinforced model.
        :rtype: OMGData
        """
        base_model = base_modules["model"].model
        assert base_model is not None
        batch_size = len(x_0.n_atoms)
        # noinspection PyTypeChecker
        times = torch.linspace(SMALL_TIME, BIG_TIME, self.integration_time_steps, device=self.device)
        x_t = x_0.clone()
        need_base_output = any(field in self.disable_fields for field in self.integrated_data_fields)

        for t_index in trange(1, len(times), desc="Integrating with reinforced model", position=1, leave=False,
                              disable=not self.enable_progress_bar):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            sqrt_dt = torch.sqrt(dt)
            time = t.repeat(batch_size)
            base_model_output = base_model(x_t, time) if need_base_output else None
            model_output = self.model(x_t, time)

            if self.integrate_pos:
                if DataField.pos in self.disable_fields:
                    # Take a standard Euler step (i.e., Euler-Maruyama without noise).
                    assert base_model_output is not None
                    drift = base_model_output[DataField.pos.name + "_b"]
                    noise = torch.zeros_like(drift)
                else:
                    sigma = self.noise_schedules[DataField.pos].noise(t)  # Scalar.
                    gamma = self.gammas[DataField.pos].gamma(t)  # Scalar.
                    assert sigma.ndim == gamma.ndim == 0
                    velocity = model_output[DataField.pos.name + "_b"]  # Shape (sum(n_atoms), 3).
                    eta = model_output[DataField.pos.name + "_eta"]  # Shape (sum(n_atoms), 3).
                    drift = velocity - 0.5 * sigma * sigma / gamma * eta
                    noise = sigma * torch.randn_like(velocity)

                # Euler-Maruyama update for SDE.
                x_t.pos = self.pos_corrector.correct(x_t.pos + drift * dt + noise * sqrt_dt)

            if self.integrate_cell:
                if DataField.cell in self.disable_fields:
                    # Take a standard Euler step (i.e., Euler-Maruyama without noise).
                    assert base_model_output is not None
                    drift = base_model_output[DataField.cell.name + "_b"]
                    noise = torch.zeros_like(drift)
                else:
                    sigma = self.noise_schedules[DataField.cell].noise(t)  # Scalar.
                    gamma = self.gammas[DataField.cell].gamma(t)  # Scalar.
                    assert sigma.ndim == gamma.ndim == 0
                    velocity = model_output[DataField.cell.name + "_b"]  # Shape (batch_size, 3, 3).
                    eta = model_output[DataField.cell.name + "_eta"]  # Shape (batch_size, 3, 3).
                    drift = velocity - 0.5 * sigma * sigma / gamma * eta
                    noise = sigma * torch.randn_like(velocity)

                # Euler-Maruyama update for SDE.
                x_t.cell = self.cell_corrector.correct(x_t.cell + drift * dt + noise * sqrt_dt)

        return x_t
