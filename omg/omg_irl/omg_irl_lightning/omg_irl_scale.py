from typing import Optional, Sequence
import warnings
import torch
import torch.nn as nn
from tqdm import trange
from omg.datamodule import OMGData
from omg.globals import SMALL_TIME, BIG_TIME
from omg.si import DifferentialEquationType, SingleStochasticInterpolantOS
from omg.utils import DataField
from omg.omg_irl import base_modules, NoiseSchedule, Reward, TimeMLP
from .abstracts import OMGIRLLightningAbstract, TrajectoryData


class OMGIRLScale(OMGIRLLightningAbstract):
    def __init__(self, reward: Reward, reference_noise_schedules: dict[str, NoiseSchedule],
                 noise_schedules: dict[str, NoiseSchedule], relative_costs: dict[str, float],
                 scale_model: TimeMLP, normalize_relative_costs: bool = True, disable_fields: Sequence[str] = (),
                 grpo_group_size: int = 32, grpo_num_groups: int = 16, grpo_share_x_0: bool = True,
                 ppo_clip_epsilon: float = 0.2, ppo_epochs: int = 1, gradient_clip_val: Optional[float] = 1.0,
                 gradient_clip_algorithm: str = "norm", generation_xyz_filename: Optional[str] = None,
                 validation_xyz_filename: Optional[str] = None, enable_progress_bar: bool = True) -> None:
        super().__init__(reward=reward, grpo_group_size=grpo_group_size, grpo_num_groups=grpo_num_groups,
                         grpo_share_x_0=grpo_share_x_0, ppo_clip_epsilon=ppo_clip_epsilon, ppo_epochs=ppo_epochs,
                         gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm,
                         generation_xyz_filename=generation_xyz_filename,
                         validation_xyz_filename=validation_xyz_filename, enable_progress_bar=enable_progress_bar)

        self.scale_model = scale_model

        try:
            self.reference_noise_schedules = {DataField[field.lower()]: ns
                                              for field, ns in reference_noise_schedules.items()}
        except KeyError as e:
            raise ValueError(f"Invalid data field in reference_noise_schedules: {e}") from e
        if any(schedule.learnable() for schedule in self.reference_noise_schedules.values()):
            raise ValueError("Reference noise schedules must not be learnable.")

        try:
            self.noise_schedules = {DataField[field.lower()]: ns for field, ns in noise_schedules.items()}
        except KeyError as e:
            raise ValueError(f"Invalid data field in noise_schedules: {e}") from e
        # Store learnable noise schedules in ModuleDict for proper registration.
        noise_schedule_modules = {field.name: schedule for field, schedule in self.noise_schedules.items()
                                  if isinstance(schedule, nn.Module)}
        assert all(schedule.learnable() for schedule in noise_schedule_modules.values())
        # Prevents empty ModuleList that would be printed out by Lightning.
        if len(noise_schedule_modules) > 0:
            self.noise_schedule_modules = nn.ModuleDict(noise_schedule_modules)
        else:
            self.noise_schedule_modules = None

        try:
            self.disable_fields = set(DataField[field.lower()] for field in disable_fields)
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
            if cost_type not in {"policy", "regularization", "entropy"}:
                raise ValueError(f"Invalid cost type in relative costs: {cost_type}. "
                                 f"Expected 'policy', 'regularization', or 'entropy'.")

        if self.integrate_species:
            raise ValueError("Integrating species is not supported in OMGIRLScale.")

        for integrated_data_field in self.integrated_data_fields:
            if integrated_data_field in self.disable_fields:
                if integrated_data_field in self.reference_noise_schedules:
                    warnings.warn(f"Reference noise schedule for disabled integrated data field "
                                  f"{integrated_data_field.name} will be ignored.")
                if integrated_data_field in self.noise_schedules:
                    warnings.warn(f"Noise schedule for disabled integrated data field "
                                  f"{integrated_data_field.name} will be ignored.")
                if integrated_data_field.name + "_policy" in self.relative_costs:
                    warnings.warn(f"Relative policy cost for disabled integrated data field "
                                  f"{integrated_data_field.name} will be ignored.")
                if integrated_data_field.name + "_regularization" in self.relative_costs:
                    warnings.warn(f"Relative regularization cost for disabled integrated data field "
                                  f"{integrated_data_field.name} will be ignored.")
                if integrated_data_field.name + "_entropy" in self.relative_costs:
                    warnings.warn(f"Relative entropy cost for disabled integrated data field "
                                  f"{integrated_data_field.name} will be ignored.")
            else:
                if integrated_data_field not in self.reference_noise_schedules:
                    raise ValueError(f"Missing reference noise schedule for integrated data field "
                                     f"{integrated_data_field.name}.")
                if integrated_data_field not in self.noise_schedules:
                    raise ValueError(f"Missing noise schedule for integrated data field {integrated_data_field.name}.")
                if integrated_data_field.name + "_policy" not in self.relative_costs:
                    raise ValueError(f"Missing relative policy cost for integrated data field "
                                     f"{integrated_data_field.name}.")
                if integrated_data_field.name + "_regularization" not in self.relative_costs:
                    raise ValueError(f"Missing relative regularization cost for integrated data field "
                                     f"{integrated_data_field.name}.")
                if (self.noise_schedules[integrated_data_field].learnable()
                        and integrated_data_field.name + "_entropy" not in self.relative_costs):
                    raise ValueError(f"Missing relative entropy cost for integrated data field "
                                     f"{integrated_data_field.name} with learnable noise schedule.")
                if (not self.noise_schedules[integrated_data_field].learnable()
                        and integrated_data_field.name + "_entropy" in self.relative_costs):
                    warnings.warn(f"Relative entropy cost for integrated data field "
                                  f"{integrated_data_field.name} with non-learnable noise schedule will be ignored.")

        for fixed_data_field in self.fixed_data_fields:
            if fixed_data_field in self.disable_fields:
                warnings.warn(f"Disabled field {fixed_data_field.name} is not an integrated data field and will be "
                              f"ignored.")
            if fixed_data_field.name in self.reference_noise_schedules:
                warnings.warn(f"Reference noise schedule for fixed data field {fixed_data_field.name} will be ignored.")
            if fixed_data_field.name in self.noise_schedules:
                warnings.warn(f"Noise schedule for fixed data field {fixed_data_field.name} will be ignored.")

        if self.integrate_pos:
            if self.pos_interpolant.differential_equation_type != DifferentialEquationType.ODE:
                warnings.warn("OMGIRLScale will ignore predicted scores for position data field and only work with"
                              "the velocity field.")
            if self.pos_interpolant.velocity_annealing_factor != 0.0:
                warnings.warn("OMGIRLScale will ignore velocity annealing for position data field.")
            if isinstance(self.pos_interpolant, SingleStochasticInterpolantOS):
                if not self.pos_interpolant.predict_velocity:
                    raise ValueError("OMGIRLScale requires velocity prediction for position data field when using "
                                     "SingleStochasticInterpolantOS.")

        if self.integrate_cell:
            if self.cell_interpolant.differential_equation_type != DifferentialEquationType.ODE:
                warnings.warn("OMGIRLScale will ignore predicted scores for cell data field and only work with"
                              "the velocity field.")
            if self.cell_interpolant.velocity_annealing_factor != 0.0:
                warnings.warn("OMGIRLScale will ignore velocity annealing for cell data field.")
            if isinstance(self.cell_interpolant, SingleStochasticInterpolantOS):
                if not self.cell_interpolant.predict_velocity:
                    raise ValueError("OMGIRLScale requires velocity prediction for cell data field when using "
                                     "SingleStochasticInterpolantOS.")

    def _rollout(self, x_0: OMGData) -> TrajectoryData:
        base_model = base_modules["model"].model
        assert base_model is not None
        batch_size = len(x_0.n_atoms)
        # noinspection PyTypeChecker
        times = torch.linspace(SMALL_TIME, BIG_TIME, self.integration_time_steps, device=self.device)
        x_t = x_0.clone()
        states = []
        old_log_probs = []
        sampled_residual_effects = []
        base_model_outputs = []
        scale_total = {field.name: torch.tensor(0.0, device=self.device)
                       for field in self.integrated_data_fields if field not in self.disable_fields}

        # Integrate over time with residuals.
        for t_index in trange(1, len(times), desc="Rollout with residuals", position=1, leave=False,
                              disable=not self.enable_progress_bar):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            sqrt_dt = torch.sqrt(dt)
            time = t.repeat(batch_size)
            base_model_output = base_model(x_t, time)
            residual_output = self.scale_model(x_t, time)

            states.append(x_t.clone())
            step_sampled_residual_effects = {}
            step_old_log_probs = {}
            step_base_model_outputs = {}

            if self.integrate_pos:
                base_b = base_model_output[DataField.pos.name + "_b"]

                if DataField.pos in self.disable_fields:
                    # Take a standard Euler step (i.e., Euler-Maruyama without noise).
                    velocity = base_b
                    noise = torch.zeros_like(base_b)
                else:
                    sigma = self.noise_schedules[DataField.pos].noise(t)  # Scalar.
                    assert sigma.ndim == 0
                    scale = residual_output[DataField.pos.name + "_s"]  # Tensor of shape (batch_size,).
                    assert scale.shape == time.shape
                    # Tensor of shape (sum(n_atoms), 1) so that it can be broadcast to base_b shape (sum(n_atoms), 3).
                    scale_per_atom = scale[x_t.batch].unsqueeze(-1)
                    assert scale_per_atom.shape[:1] == base_b.shape[:1]
                    velocity = (1.0 + scale_per_atom) * base_b
                    randn = torch.randn_like(scale)  # Tensor of shape (batch_size).
                    randn_per_atom = randn[x_t.batch].unsqueeze(-1)  # Tensor of shape (sum(n_atoms), 1).
                    noise = sigma * randn_per_atom * base_b
                    # Store deviation effect for PPO update: res_s * dt + noise * sqrt(dt).
                    # Effectively x_t+dt - x_t - base_b * dt, however, ignoring base_b direction.
                    # Noise is effectively one-dimensional. Shape (batch_size,).
                    step_sampled_residual_effects[DataField.pos] = scale * dt + sigma * randn * sqrt_dt
                    # Log probability of x_t+dt given x_t for SDE.
                    # Tensor of shape (batch_size,).
                    log_prob = -0.5 * (torch.log(2.0 * torch.pi * (sigma ** 2) * dt) + (randn ** 2))
                    step_old_log_probs[DataField.pos] = log_prob
                    # Log absolute scale mean over batch.
                    assert DataField.pos.name in scale_total
                    scale_total[DataField.pos.name] += scale.abs().mean()

                # Euler-Maruyama update for SDE.
                x_t.pos = self.pos_corrector.correct(x_t.pos + velocity * dt + noise * sqrt_dt)

            if self.integrate_cell:
                base_b = base_model_output[DataField.cell.name + "_b"]

                if DataField.cell in self.disable_fields:
                    velocity = base_b
                    noise = torch.zeros_like(base_b)
                else:
                    sigma = self.noise_schedules[DataField.cell].noise(t)  # Scalar.
                    assert sigma.ndim == 0
                    # Tensor of shape (batch_size, 1, 1) so that it can be broadcast to base_b shape (batch_size, 3, 3).
                    scale = residual_output[DataField.cell.name + "_s"].unsqueeze(-1).unsqueeze(-1)
                    assert scale.shape[:1] == base_b.shape[:1]
                    velocity = (1.0 + scale) * base_b
                    randn = torch.randn_like(scale)
                    noise = sigma * randn * base_b
                    # Store deviation effect for PPO update: res_s * dt + noise * sqrt(dt).
                    # Effectively x_t+dt - x_t - base_b * dt, however, ignoring base_b direction.
                    # Noise is effectively one-dimensional. Shape (batch_size,).
                    step_sampled_residual_effects[DataField.cell] = (
                            scale.squeeze(dim=(1, 2)) * dt + sigma * randn.squeeze(dim=(1, 2)) * sqrt_dt)
                    # Log probability of x_t+dt given x_t for SDE.
                    # Tensor of shape (batch_size,).
                    step_old_log_probs[DataField.cell] = -0.5 * (
                            torch.log(2.0 * torch.pi * (sigma ** 2) * dt) + (randn.squeeze(dim=(1, 2)) ** 2)
                    )
                    # Log absolute scale mean over batch.
                    assert DataField.cell.name in scale_total
                    scale_total[DataField.cell.name] += scale.squeeze(dim=(1, 2)).abs().mean()

                # Euler-Maruyama update for SDE.
                x_t.cell = self.cell_corrector.correct(x_t.cell + velocity * dt + noise * sqrt_dt)

            sampled_residual_effects.append(step_sampled_residual_effects)
            old_log_probs.append(step_old_log_probs)
            base_model_outputs.append(step_base_model_outputs)

        # Append final state.
        states.append(x_t.clone())

        # Average over time steps.
        info_dict = {f"{key}_scale": scale / (len(times) - 1) for key, scale in scale_total.items()}

        return self._create_trajectory_data(states, sampled_residual_effects, old_log_probs, base_model_outputs,
                                            info_dict)

    def ppo_update(self, trajectory: TrajectoryData) -> tuple[float, dict[str, float]]:
        batch_size = len(trajectory.rewards)
        # noinspection PyTypeChecker
        times = torch.linspace(SMALL_TIME, BIG_TIME, self.integration_time_steps, device=self.device)
        num_timesteps = len(times) - 1

        # Track losses for logging (only for non-disabled integrated fields).
        active_fields = [field for field in self.integrated_data_fields if field not in self.disable_fields]
        total_loss = 0.0
        total_policy_losses = {field: 0.0 for field in active_fields}
        total_reg_losses = {field: 0.0 for field in active_fields}
        total_entropy_losses = {field: 0.0 for field in active_fields}
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
            # Re-evaluate residual model with gradients.
            residual_output = self.scale_model(x_t, time)
            timestep_loss = torch.tensor(0.0, device=self.device)

            if self.integrate_pos and DataField.pos not in self.disable_fields:
                sigma_ref = self.reference_noise_schedules[DataField.pos].noise(t)  # Scalar.
                sigma = self.noise_schedules[DataField.pos].noise(t)  # Scalar.
                assert sigma.ndim == 0
                scale = residual_output[DataField.pos.name + "_s"]  # Shape (batch_size,).
                old_log_prob = trajectory.old_log_probs[t_index][DataField.pos]  # Shape (batch_size,).
                # This is effectively x_t+dt - x_t - base_b * dt, however, ignoring base_b direction.
                # Noise is effectively one-dimensional. Shape (batch_size,).
                sampled_residual_effect = trajectory.sampled_residual_effects[t_index][DataField.pos]
                # Tensor of shape (batch_size,).
                current_log_prob = -0.5 * (
                        torch.log(2.0 * torch.pi * (sigma ** 2) * dt)
                        + ((sampled_residual_effect.detach() - scale * dt) / (sigma * sqrt_dt)) ** 2
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
                          + (sigma * sigma + scale * scale * dt) / (2.0 * sigma_ref * sigma_ref) - 0.5)
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

            if self.integrate_cell and DataField.cell not in self.disable_fields:
                sigma_ref = self.reference_noise_schedules[DataField.cell].noise(t)  # Scalar.
                sigma = self.noise_schedules[DataField.cell].noise(t)  # Scalar.
                assert sigma.ndim == 0
                scale = residual_output[DataField.cell.name + "_s"]  # Shape (batch_size,).
                old_log_prob = trajectory.old_log_probs[t_index][DataField.cell]  # Shape (batch_size,).
                # This is effectively x_t+dt - x_t - base_b * dt, however, ignoring base_b direction.
                # Noise is effectively one-dimensional. Shape (batch_size,).
                sampled_residual_effect = trajectory.sampled_residual_effects[t_index][DataField.cell]
                # Tensor of shape (batch_size,).
                current_log_prob = -0.5 * (
                        torch.log(2.0 * torch.pi * (sigma ** 2) * dt)
                        + ((sampled_residual_effect.detach() - scale * dt) / (sigma * sqrt_dt)) ** 2
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
                          + (sigma * sigma + scale * scale * dt) / (2.0 * sigma_ref * sigma_ref) - 0.5)
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
            if self.noise_schedules[field].learnable():
                logging_info[field.name + "_loss_entropy"] = total_entropy_losses[field] / num_timesteps
            logging_info[field.name + "_clip_fraction"] = total_clip_fractions[field] / num_timesteps

        return total_loss, logging_info

    def _integrate(self, x_0: OMGData) -> OMGData:
        base_model = base_modules["model"].model
        assert base_model is not None
        batch_size = len(x_0.n_atoms)
        # noinspection PyTypeChecker
        times = torch.linspace(SMALL_TIME, BIG_TIME, self.integration_time_steps, device=self.device)
        x_t = x_0.clone()

        for t_index in trange(1, len(times), desc="Integrating with scales", position=1, leave=False,
                              disable=not self.enable_progress_bar):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            sqrt_dt = torch.sqrt(dt)
            time = t.repeat(batch_size)
            base_model_output = base_model(x_t, time)
            residual_output = self.scale_model(x_t, time)

            if self.integrate_pos:
                base_b = base_model_output[DataField.pos.name + "_b"]

                if DataField.pos in self.disable_fields:
                    # Take a standard Euler step (i.e., Euler-Maruyama without noise).
                    velocity = base_b
                    noise = torch.zeros_like(base_b)
                else:
                    sigma = self.noise_schedules[DataField.pos].noise(t)  # Scalar.
                    assert sigma.ndim == 0
                    scale = residual_output[DataField.pos.name + "_s"]  # Tensor of shape (batch_size,).
                    assert scale.shape == time.shape
                    # Tensor of shape (sum(n_atoms), 1) so that it can be broadcast to base_b shape (sum(n_atoms), 3).
                    scale_per_atom = scale[x_t.batch].unsqueeze(-1)
                    assert scale_per_atom.shape[:1] == base_b.shape[:1]
                    velocity = (1.0 + scale_per_atom) * base_b
                    randn = torch.randn_like(scale)  # Tensor of shape (batch_size,).
                    randn_per_atom = randn[x_t.batch].unsqueeze(-1)  # Tensor of shape (sum(n_atoms), 1).
                    noise = sigma * randn_per_atom * base_b

                # Euler-Maruyama update for SDE.
                x_t.pos = self.pos_corrector.correct(x_t.pos + velocity * dt + noise * sqrt_dt)

            if self.integrate_cell:
                base_b = base_model_output[DataField.cell.name + "_b"]

                if DataField.cell in self.disable_fields:
                    # Take a standard Euler step (i.e., Euler-Maruyama without noise).
                    velocity = base_b
                    noise = torch.zeros_like(base_b)
                else:
                    sigma = self.noise_schedules[DataField.cell].noise(t)  # Scalar.
                    assert sigma.ndim == 0
                    # Tensor of shape (batch_size, 1, 1) so that it can be broadcast to base_b shape (batch_size, 3, 3).
                    scale = residual_output[DataField.cell.name + "_s"].unsqueeze(-1).unsqueeze(-1)
                    assert scale.shape[:1] == base_b.shape[:1]
                    velocity = (1.0 + scale) * base_b
                    noise = sigma * torch.randn_like(scale) * base_b

                # Euler-Maruyama update for SDE.
                x_t.cell = self.cell_corrector.correct(x_t.cell + velocity * dt + noise * sqrt_dt)

        return x_t

    def training_step(self, batch: OMGData, batch_idx: int) -> None:
        """
        Take a training step using GRPO with a PPO-like objective.

        This method extends the training_step method of the OMGIRLLightningAbstract class by logging
        statistics about the learnable noise schedules after the standard training step.

        :param batch:
            Batch of training data from the datamodule.
            This batch contains grpo_num_groups unique structures.
        :type batch: OMGData
        :param batch_idx:
            Index of the current batch.
        :type batch_idx: int
        """
        super().training_step(batch, batch_idx)

        # Log sigma schedule statistics for learnable noise schedules.
        for field in self.integrated_data_fields:
            if field not in self.disable_fields and self.noise_schedules[field].learnable():
                # noinspection PyTypeChecker
                times = torch.linspace(SMALL_TIME, BIG_TIME, self.integration_time_steps, device=self.device)
                sigmas = self.noise_schedules[field].noise(times)
                self.log(f"sigma_{field.name}_mean", sigmas.mean(), on_step=True, on_epoch=True, prog_bar=True,
                         sync_dist=True, batch_size=len(batch))
                self.log(f"sigma_{field.name}_min", sigmas.min(), on_step=True, on_epoch=True, sync_dist=True,
                         batch_size=len(batch))
                self.log(f"sigma_{field.name}_max", sigmas.max(), on_step=True, on_epoch=True, sync_dist=True,
                         batch_size=len(batch))
