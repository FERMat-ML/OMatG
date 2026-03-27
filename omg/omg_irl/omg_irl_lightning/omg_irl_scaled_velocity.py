from copy import deepcopy
from typing import Optional, Sequence
import warnings
import torch
from torch_scatter import scatter_add, scatter_mean
from tqdm import trange
from omg.datamodule import OMGData
from omg.globals import SMALL_TIME, BIG_TIME
from omg.utils import DataField
from omg.omg_irl.base_modules import base_modules
from omg.omg_irl.noise_schedules import NoiseSchedule
from omg.omg_irl.rewards import Reward
from .abstracts import TrajectoryData
from .omg_irl_scale import OMGIRLScale
from .omg_irl_velocity import PositionNormalization
from .scale_mlp import ScaleMLP


class OMGIRLScaledVelocity(OMGIRLScale):
    """
    Scaled-velocity Open Materials Generation with Inference-time Reinforcement Learning (OMatG-IRL) using
    group-relative policy optimization (GRPO) and proximal policy optimization (PPO) within the PyTorch Lightning
    framework.

    This class combines the full velocity learning of OMGIRLVelocity with the time-dependent velocity-annealing
    schedule of OMGIRLScale. The learned velocity is scaled by (1 + s(t)), where s(t) is provided by a ScaleMLP,
    resulting in the following Euler-Maruyama update for the variable x_t:
    x_t+dt = x_t + (1 + s(t)) * b_theta(x_t, t) * dt + sigma(t) * randn * sqrt(dt),
    where b_theta(x_t, t) is the reinforced velocity field, randn is a standard normal random variable with the same
    dimension as b_theta(x_t, t), and sigma(t) is a (potentially learnable) noise schedule.

    Analogously, the reference policy is given by
    x_t+dt = x_t + b_ref(x_t, t) * dt + sigma_ref(t) * randn * sqrt(dt),
    where b_ref(x_t, t) is the velocity field predicted by the frozen base model, and sigma_ref(t) is a fixed reference
    noise schedule.

    The scale model must be pre-trained using OMGIRLScale. It is frozen upon construction and not updated during
    training, serving as a fixed warm start for the velocity learning.

    The GRPO framework is used to reinforce the velocity fields for the atomic positions and the lattice vectors,
    given that they are integrated by the base model. Any scores predicted by the base model are ignored. Optionally,
    one can switch off the reinforcement for each of these data fields. In this case, the base model's velocity field is
    used without modification, and no noise is added during integration.

    For every data field that is reinforced, the loss consists of a policy loss, a regularization loss, and optionally,
    if the noise schedule for that data field is learnable, an entropy loss. The relative costs of these losses should
    be specified with the keys '<data_field>_policy', '<data_field>_regularization', and '<data_field>_entropy' in the
    relative_costs dictionary.

    If the velocity field of the atomic positions are reinforced, one should consider that there can be variable numbers
    of atoms per structure. To prevent policy updates from being biased toward larger crystals, different position
    normalization modes can be selected.

    One cannot use this class to reinforce the discrete species field. If species are integrated by the base model, they
    are passively integrated using the frozen base model at each timestep without any RL. This enables de novo
    generation where species are predicted alongside positions and lattice vectors.

    :param reward:
        Reward function to evaluate the generated structures.
    :type reward: Reward
    :param reference_noise_schedules:
        Dictionary mapping data field names to reference noise schedules sigma_ref(t).
        The reference noise schedules must not be learnable.
    :type reference_noise_schedules: dict[str, NoiseSchedule]
    :param noise_schedules:
        Dictionary mapping data field names to noise schedules sigma(t).
        The noise schedules can be learnable or non-learnable.
    :type noise_schedules: dict[str, NoiseSchedule]
    :param relative_costs:
        Dictionary mapping cost types to their relative costs for each data field.
    :type relative_costs: dict[str, float]
    :param scale_model:
        A pre-trained scale model providing the frozen time-dependent velocity-annealing schedule s(t).
        Must be pre-trained using OMGIRLScale. The model is frozen upon construction.
    :type scale_model: ScaleMLP
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
        If reference_noise_schedules contains an invalid data field name.
        If any reference noise schedule is learnable.
        If noise_schedules contains an invalid data field name.
        If disable_fields contains an invalid data field name.
        If any relative cost is negative.
        If all relative costs are zero.
        If any relative cost key does not follow the format '<data_field>_<cost_type>', where <cost_type> is one of
        'policy', 'regularization', or 'entropy'.
        If an integrated data field is missing a reference noise schedule, noise schedule, or relative costs.
        If velocity prediction is not enabled for position or cell data fields when using SingleStochasticInterpolantOS.
        If position_normalization is invalid.
    """

    def __init__(self, reward: Reward, reference_noise_schedules: dict[str, NoiseSchedule],
                 noise_schedules: dict[str, NoiseSchedule], relative_costs: dict[str, float],
                 scale_model: ScaleMLP, normalize_relative_costs: bool = True,
                 disable_fields: Sequence[str] = (), position_normalization: str = "per_atom_surrogate",
                 grpo_group_size: int = 32, grpo_num_groups: int = 16, grpo_share_x_0: bool = True,
                 ppo_clip_epsilon: float = 0.2, ppo_epochs: int = 1, gradient_clip_val: Optional[float] = 1.0,
                 gradient_clip_algorithm: str = "norm", generation_xyz_filename: Optional[str] = None,
                 validation_xyz_filename: Optional[str] = None, enable_progress_bar: bool = True) -> None:
        super().__init__(reward=reward, reference_noise_schedules=reference_noise_schedules,
                         noise_schedules=noise_schedules, relative_costs=relative_costs, scale_model=scale_model,
                         normalize_relative_costs=normalize_relative_costs, disable_fields=disable_fields,
                         grpo_group_size=grpo_group_size, grpo_num_groups=grpo_num_groups,
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
        base_model.unfreeze()  # Base model was frozen in OMGIRLLightningAbstract, so unfreeze it before copying.
        self.model = deepcopy(base_model.model)
        base_model.freeze()

        # Freeze the pre-trained scale model — it is not updated during training.
        assert self.scale_model is not None
        self.scale_model.requires_grad_(False)

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
        for t_index in trange(1, len(times), desc="Rollout with scaled reinforced model", position=1, leave=False,
                              disable=not self.enable_progress_bar):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            sqrt_dt = torch.sqrt(dt)
            time = t.repeat(batch_size)
            # Base model output necessary for KL regularization.
            base_model_output = base_model(x_t, time)
            model_output = self.model(x_t, time)
            scale_output = self.scale_model(time)

            states.append(x_t.clone())
            step_sampled_actions = {}
            step_old_log_probs = {}
            step_base_model_outputs = {}

            if self.integrate_pos:
                base_b = base_model_output[DataField.pos.name + "_b"]

                # Apply frozen scale from pre-trained scale model (always, regardless of disable_fields).
                # The scale modifies the ground-truth velocity: the reference becomes (1 + s(t)) * b_ref.
                pos_scale = scale_output[DataField.pos.name + "_s"]  # Shape (batch_size,).
                pos_scale_per_atom = pos_scale[x_t.batch].unsqueeze(-1)  # Shape (sum(n_atoms), 1).
                base_b = (1.0 + pos_scale_per_atom) * base_b

                if DataField.pos in self.disable_fields:
                    # Take a scaled Euler step (no noise, no RL, but with pre-trained scale).
                    velocity = base_b
                    noise = torch.zeros_like(base_b)
                else:
                    sigma = self.noise_schedules[DataField.pos].noise(t)  # Scalar.
                    assert sigma.ndim == 0
                    velocity = model_output[DataField.pos.name + "_b"]  # Shape (sum(n_atoms), 3).
                    assert velocity.shape == base_b.shape
                    velocity = (1.0 + pos_scale_per_atom) * velocity
                    diff_b = velocity - base_b
                    randn = torch.randn_like(velocity)
                    noise = sigma * randn
                    # Store base model output.
                    step_base_model_outputs[DataField.pos] = base_b
                    # Store deviation effect for PPO update: diff_b * dt + noise * sqrt(dt).
                    # Effectively x_t+dt - x_t - base_b * dt. Shape (sum(n_atoms), 3).
                    step_sampled_actions[DataField.pos] = diff_b * dt + noise * sqrt_dt
                    # Log probability of x_t+dt given x_t for SDE.
                    # Sum log probs over x, y, and z yielding tensor of shape (sum(n_atoms),).
                    log_probs_atoms = -0.5 * (
                            torch.log(2.0 * torch.pi * (sigma ** 2) * dt) + (randn ** 2)
                    ).sum(dim=tuple(range(1, randn.ndim)))
                    step_old_log_probs[DataField.pos] = log_probs_atoms
                    # Norm over x, y, z.
                    base_norm_atoms = torch.linalg.norm(base_b, dim=-1)
                    diff_b_norm_atoms = torch.linalg.norm(diff_b, dim=-1)
                    # Mean over atoms.
                    base_norm_struct = scatter_mean(base_norm_atoms, x_t.batch)
                    diff_b_norm_struct = scatter_mean(diff_b_norm_atoms, x_t.batch)
                    # Mean over batch.
                    ratios_total[DataField.pos] += (diff_b_norm_struct / (base_norm_struct + 1.0e-8)).mean()

                # Euler-Maruyama update for SDE.
                x_t.pos = self.pos_corrector.correct(x_t.pos + velocity * dt + noise * sqrt_dt)

            if self.integrate_cell:
                base_b = base_model_output[DataField.cell.name + "_b"]

                # Apply frozen scale from pre-trained scale model (always, regardless of disable_fields).
                # The scale modifies the ground-truth velocity: the reference becomes (1 + s(t)) * b_ref.
                cell_scale = scale_output[DataField.cell.name + "_s"]  # Shape (batch_size,).
                cell_scale_cell = cell_scale.unsqueeze(-1).unsqueeze(-1)  # Shape (batch_size, 1, 1).
                base_b = (1.0 + cell_scale_cell) * base_b

                if DataField.cell in self.disable_fields:
                    # Take a scaled Euler step (no noise, no RL, but with pre-trained scale).
                    velocity = base_b
                    noise = torch.zeros_like(base_b)
                else:
                    sigma = self.noise_schedules[DataField.cell].noise(t)  # Scalar.
                    assert sigma.ndim == 0
                    velocity = model_output[DataField.cell.name + "_b"]  # Shape (batch_size, 3, 3).
                    assert velocity.shape == base_b.shape
                    velocity = (1.0 + cell_scale_cell) * velocity
                    diff_b = velocity - base_b
                    randn = torch.randn_like(velocity)
                    noise = sigma * randn
                    # Store base model output.
                    step_base_model_outputs[DataField.cell] = base_b
                    # Store deviation effect for PPO update: diff_b * dt + noise * sqrt(dt).
                    # Effectively x_t+dt - x_t - base_b * dt. Shape (batch_size, 3, 3).
                    step_sampled_actions[DataField.cell] = diff_b * dt + noise * sqrt_dt
                    # Log probability of x_t+dt given x_t for SDE.
                    # Sum log probs over all dimensions except batch yielding tensor of shape (batch_size,).
                    step_old_log_probs[DataField.cell] = -0.5 * (
                            torch.log(2.0 * torch.pi * (sigma ** 2) * dt) + (randn ** 2)
                    ).sum(dim=tuple(range(1, randn.ndim)))
                    # Norm over cell dimensions.
                    base_norm_struct = torch.linalg.norm(base_b.reshape(base_b.shape[0], -1), dim=-1)
                    diff_b_norm_struct = torch.linalg.norm(diff_b.reshape(diff_b.shape[0], -1), dim=-1)
                    # Mean over batch.
                    ratios_total[DataField.cell] += (diff_b_norm_struct / (base_norm_struct + 1.0e-8)).mean()

                # Euler-Maruyama update for SDE.
                x_t.cell = self.cell_corrector.correct(x_t.cell + velocity * dt + noise * sqrt_dt)

            if self.integrate_species:
                # Integrate species using the frozen base model (no RL on species).
                species_b = base_model_output[DataField.species.name + "_b"]
                species_eta = base_model_output[DataField.species.name + "_eta"]
                x_t.species = self.species_interpolant.integrate(lambda _, __: (species_b, species_eta), x_t.species, t,
                                                                 dt, x_t.batch)

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
            # Re-evaluate model with gradients. Scale model is frozen (no gradients flow through it).
            model_output = self.model(x_t, time)
            scale_output = self.scale_model(time)
            timestep_loss = torch.tensor(0.0, device=self.device)

            if self.integrate_pos and DataField.pos not in self.disable_fields:
                sigma_ref = self.reference_noise_schedules[DataField.pos].noise(t)  # Scalar.
                sigma = self.noise_schedules[DataField.pos].noise(t)  # Scalar.
                assert sigma.ndim == sigma_ref.ndim == 0
                velocity = model_output[DataField.pos.name + "_b"]  # Shape (sum(n_atoms), 3).

                # Apply frozen scale from pre-trained scale model.
                scale = scale_output[DataField.pos.name + "_s"]  # Shape (batch_size,).
                scale_per_atom = scale[x_t.batch].unsqueeze(-1)  # Shape (sum(n_atoms), 1).
                velocity = (1.0 + scale_per_atom) * velocity

                diff_b = velocity - trajectory.base_model_outputs[t_index][DataField.pos]

                old_log_probs_atoms = trajectory.old_log_probs[t_index][DataField.pos]  # Shape (sum(n_atoms),).
                # This is effectively x_t+dt - x_t - base_b * dt. Shape (sum(n_atoms), 3).
                sampled_action = trajectory.sampled_actions[t_index][DataField.pos]
                # Sum log probs over x, y, z.
                current_log_probs_atoms = -0.5 * (
                        torch.log(2.0 * torch.pi * (sigma ** 2) * dt)
                        + ((sampled_action.detach() - diff_b * dt) / (sigma * sqrt_dt)) ** 2
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
                kl_div = (torch.log(sigma_ref) - torch.log(sigma)
                          + (sigma * sigma + diff_b * diff_b * dt) / (2.0 * sigma_ref * sigma_ref) - 0.5)
                # Sum over position dimensions to get per-atom KL.
                per_atom_kl = kl_div.sum(dim=-1)
                if self.position_normalization == PositionNormalization.NONE:
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
                                             * diff_b.shape[-1])
                    if self.position_normalization == PositionNormalization.NONE:
                        # Sum over atoms to get per-structure entropy and take mean over batch.
                        entropy_loss = (entropy_loss_per_atom * x_t.n_atoms).mean()
                    else:
                        # Take mean over batch.
                        entropy_loss = entropy_loss_per_atom.mean()
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
                assert sigma.ndim == sigma_ref.ndim == 0
                velocity = model_output[DataField.cell.name + "_b"]  # Shape (batch_size, 3, 3).

                # Apply frozen scale from pre-trained scale model.
                scale = scale_output[DataField.cell.name + "_s"]  # Shape (batch_size,).
                scale_cell = scale.unsqueeze(-1).unsqueeze(-1)  # Shape (batch_size, 1, 1).
                velocity = (1.0 + scale_cell) * velocity

                diff_b = velocity - trajectory.base_model_outputs[t_index][DataField.cell]

                old_log_prob = trajectory.old_log_probs[t_index][DataField.cell]  # Shape (batch_size,).
                # This is effectively x_t+dt - x_t - base_b * dt. Shape (batch_size, 3, 3).
                sampled_action = trajectory.sampled_actions[t_index][DataField.cell]
                # Here we effectively choose the NONE normalization since the cell shape is always the same.
                # Sum log probs over all cell dimensions except batch.
                current_log_prob = -0.5 * (
                        torch.log(2.0 * torch.pi * (sigma ** 2) * dt)
                        + ((sampled_action.detach() - diff_b * dt) / (sigma * sqrt_dt)) ** 2
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
                kl_div = (torch.log(sigma_ref) - torch.log(sigma)
                          + (sigma * sigma + diff_b * diff_b * dt) / (2.0 * sigma_ref * sigma_ref) - 0.5)
                # Sum over cell dimensions and take mean over batch.
                reg_loss = kl_div.sum(dim=tuple(range(1, kl_div.ndim))).mean()

                # Entropy bonus when learning sigma.
                if self.noise_schedules[DataField.cell].learnable():
                    # Entropy of Gaussian grows with log(sigma). Take negative since we want to maximize entropy.
                    # Multiply by cell dimensions.
                    entropy_loss = (-0.5 * torch.log(2.0 * torch.pi * sigma * sigma) - 0.5) * diff_b.shape[1:].numel()
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

        for t_index in trange(1, len(times), desc="Integrating with scaled reinforced model", position=1, leave=False,
                              disable=not self.enable_progress_bar):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            sqrt_dt = torch.sqrt(dt)
            time = t.repeat(batch_size)
            base_model_output = base_model(x_t, time) if need_base_output else None
            model_output = self.model(x_t, time)
            scale_output = self.scale_model(time)

            if self.integrate_pos:
                # Apply frozen scale from pre-trained scale model (always, regardless of disable_fields).
                pos_scale = scale_output[DataField.pos.name + "_s"]  # Shape (batch_size,).
                pos_scale_per_atom = pos_scale[x_t.batch].unsqueeze(-1)  # Shape (sum(n_atoms), 1).

                if DataField.pos in self.disable_fields:
                    # Take a scaled Euler step (no noise, no RL, but with pre-trained scale).
                    assert base_model_output is not None
                    velocity = (1.0 + pos_scale_per_atom) * base_model_output[DataField.pos.name + "_b"]
                    noise = torch.zeros_like(velocity)
                else:
                    sigma = self.noise_schedules[DataField.pos].noise(t)  # Scalar.
                    assert sigma.ndim == 0
                    velocity = model_output[DataField.pos.name + "_b"]  # Shape (sum(n_atoms), 3).
                    velocity = (1.0 + pos_scale_per_atom) * velocity

                    noise = sigma * torch.randn_like(velocity)

                # Euler-Maruyama update for SDE.
                x_t.pos = self.pos_corrector.correct(x_t.pos + velocity * dt + noise * sqrt_dt)

            if self.integrate_cell:
                # Apply frozen scale from pre-trained scale model (always, regardless of disable_fields).
                cell_scale = scale_output[DataField.cell.name + "_s"]  # Shape (batch_size,).
                cell_scale_cell = cell_scale.unsqueeze(-1).unsqueeze(-1)  # Shape (batch_size, 1, 1).

                if DataField.cell in self.disable_fields:
                    # Take a scaled Euler step (no noise, no RL, but with pre-trained scale).
                    assert base_model_output is not None
                    velocity = (1.0 + cell_scale_cell) * base_model_output[DataField.cell.name + "_b"]
                    noise = torch.zeros_like(velocity)
                else:
                    sigma = self.noise_schedules[DataField.cell].noise(t)  # Scalar.
                    assert sigma.ndim == 0
                    velocity = model_output[DataField.cell.name + "_b"]  # Shape (batch_size, 3, 3).
                    velocity = (1.0 + cell_scale_cell) * velocity

                    noise = sigma * torch.randn_like(velocity)

                # Euler-Maruyama update for SDE.
                x_t.cell = self.cell_corrector.correct(x_t.cell + velocity * dt + noise * sqrt_dt)

            if self.integrate_species:
                # Integrate species using the frozen base model (no RL on species).
                species_b = base_model_output[DataField.species.name + "_b"]
                species_eta = base_model_output[DataField.species.name + "_eta"]
                x_t.species = self.species_interpolant.integrate(lambda _, __: (species_b, species_eta), x_t.species, t,
                                                                 dt, x_t.batch)

        return x_t
