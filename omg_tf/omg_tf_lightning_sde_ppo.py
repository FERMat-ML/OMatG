"""
SDE-based PPO training for stochastic interpolants.

This module implements a more principled approach to RL fine-tuning of stochastic interpolant models
that use SDE sampling with both velocity (b) and denoiser (eta) outputs.

The key difference from omg_tf_lightning_ppo.py:
- Uses the full SDE dynamics: dx = [b - (ε/γ)·η] dt + √(2ε) dW
- The exploration noise comes from the SDE formulation, not an ad-hoc schedule
- Keeps η valid via distillation from the base model
"""
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from time import strftime
from typing import Optional
import lightning
import torch
from torch_geometric.data import Batch
from torch_scatter import scatter_add, scatter_mean
from tqdm import trange
from omg.datamodule import OMGData, Structure
from omg.globals import BIG_TIME, SMALL_TIME
from omg.si import (DifferentialEquationType, SingleStochasticInterpolant, SingleStochasticInterpolantOS,
                    SingleStochasticInterpolantIdentity)
from omg.si.abstracts import Epsilon
from omg.utils import DataField, xyz_saver
from omg_tf.abstracts import Reward
from omg_tf.base_modules import base_modules


@dataclass
class SDETrajectoryData:
    """Stores trajectory data from SDE rollout for PPO-style training."""
    # List of x_t states at each timestep (detached).
    states: list[OMGData]
    # List of actions (displacements) taken at each timestep. Shape (sum(n_atoms), 3) per step.
    actions: list[torch.Tensor]
    # List of old log probabilities per atom at each timestep. Shape (sum(n_atoms),) per step.
    old_log_probs_atoms: list[torch.Tensor]
    # List of base model outputs (b, eta) at each timestep for KL/distillation.
    base_outputs: list[tuple[torch.Tensor, torch.Tensor]]
    # Rewards for final structures.
    rewards: torch.Tensor
    # Advantages for final structures.
    advantages: torch.Tensor
    # Info dictionary for final structures.
    info_dict: dict[str, torch.Tensor]


class OMGTFLightningSDEPPO(lightning.LightningModule):
    """
    Lightning module for PPO-style RL training using principled SDE dynamics.

    This class uses the full SDE formulation from stochastic interpolants:
        dx = [b - (ε/γ)·η] dt + √(2ε) dW

    The policy model learns both b and η, with:
    - Policy gradient applied to the full drift
    - η kept valid via distillation loss from base model

    Simplifications compared to OMGTFLightningPPO:
    - Only supports FULL mode (policy model replaces base model entirely)
    - Only learns position residuals (cell uses base model directly)
    - Requires SDE-type stochastic interpolant with gamma and epsilon
    """

    class PositionNormalization(Enum):
        """Enumeration of position normalization modes for PPO surrogate."""

        NONE = auto()
        """
        Use joint-action PPO ratio without normalization.

        Log probabilities of all atoms in a structure are summed to compute the joint action
        log probability. Larger structures have more extreme ratios for the same per-atom change.
        """
        PER_STRUCTURE_WEIGHT = auto()
        """
        Use joint-action PPO ratio, but weight per-structure advantage by 1 / n_atoms.

        This reduces the influence of larger structures on the policy update while keeping
        the joint-action ratio formulation.
        """
        PER_ATOM_SURROGATE = auto()
        """
        Use per-atom PPO ratio averaged per structure.

        Computes the PPO ratio for each atom individually and averages them per structure.
        Most normalized approach - structure size has minimal effect on ratio magnitude.
        """

    def __init__(
        self,
        reward: Reward,
        epsilon: Epsilon,
        relative_costs: dict[str, float],
        normalize_relative_costs: bool = True,
        grpo_group_size: int = 32,
        grpo_num_groups: int = 16,
        grpo_share_x_0: bool = True,
        ppo_clip_epsilon: float = 0.2,
        ppo_epochs: int = 1,
        position_normalization: str = "none",
        gradient_clip_val: Optional[float] = 1.0,
        gradient_clip_algorithm: str = "norm",
        generation_xyz_filename: Optional[str] = None,
        enable_progress_bar: bool = True,
        store_validation_structures_path: Optional[str] = None
    ) -> None:
        """
        Constructor for OMGTFLightningSDEPPO.

        :param reward:
            The reward function to optimize.
        :param epsilon:
            Epsilon function ε(t) for the SDE diffusion coefficient.
        :param relative_costs:
            Dictionary with keys: "pos_policy", "pos_regularization", and optional "pos_eta_distillation".
        :param normalize_relative_costs:
            If True, normalize relative_costs to sum to 1.
        :param grpo_group_size:
            Number of samples per GRPO group.
        :param grpo_num_groups:
            Number of GRPO groups per batch.
        :param grpo_share_x_0:
            If True, all group members share the same initial structure x_0.
        :param ppo_clip_epsilon:
            PPO clipping parameter epsilon.
        :param ppo_epochs:
            Number of PPO update epochs per rollout.
        :param position_normalization:
            Position normalization mode for PPO surrogate. Options:
            - "none": joint-action PPO ratio without normalization
            - "per_structure_weight": joint-action ratio, but weight per-structure advantage by 1 / n_atoms
            - "per_atom_surrogate": per-atom PPO ratio averaged per structure
        :param gradient_clip_val:
            Maximum gradient norm for clipping.
        :param gradient_clip_algorithm:
            Algorithm for gradient clipping.
        :param generation_xyz_filename:
            Filename for saving generated structures during prediction.
        :param enable_progress_bar:
            If True, enable progress bar during training.
        :param store_validation_structures:
            If True, store validation structures for reward computation.
        """
        super().__init__()
        self.automatic_optimization = False

        self.reward = reward
        self.epsilon = epsilon

        # Get base model and create policy model as a copy.
        base_model = base_modules["model"]
        if base_model is None:
            raise ValueError("Base model must be set globally before initializing.")

        self.policy_model = deepcopy(base_model.model)
        base_model.freeze()

        # Validate that position uses SDE-type stochastic interpolant.
        pos_interpolant = base_model.si.get_stochastic_interpolant(DataField.pos.name)
        if isinstance(pos_interpolant, SingleStochasticInterpolantIdentity):
            raise ValueError("Position must use a non-identity stochastic interpolant.")
        if not isinstance(pos_interpolant, SingleStochasticInterpolant):
            raise ValueError("Position must use SingleStochasticInterpolant for SDE-based training.")
        if pos_interpolant.differential_equation_type != DifferentialEquationType.SDE:
            raise ValueError("Position stochastic interpolant must be SDE type.")
        if pos_interpolant.velocity_annealing_factor != 0.0:
            raise ValueError("Velocity annealing is not supported.")
        self.gamma = pos_interpolant.get_gamma()
        if self.gamma is None:
            raise ValueError("Position stochastic interpolant must provide gamma for SDE-based training.")
        self.pos_corrector = pos_interpolant.get_corrector()

        # Cell uses base model directly (no RL).
        cell_interpolant = base_model.si.get_stochastic_interpolant(DataField.cell.name)
        self.integrate_cell = not isinstance(cell_interpolant, SingleStochasticInterpolantIdentity)
        self.cell_corrector = None
        if self.integrate_cell:
            if isinstance(cell_interpolant, SingleStochasticInterpolant):
                if cell_interpolant.differential_equation_type != DifferentialEquationType.ODE:
                    raise ValueError("Cell stochastic interpolant must be ODE type for SDE PPO.")
                if cell_interpolant.velocity_annealing_factor != 0.0:
                    raise ValueError("Cell velocity annealing is not supported for SDE PPO.")
                self.cell_corrector = cell_interpolant.get_corrector()
            elif isinstance(cell_interpolant, SingleStochasticInterpolantOS):
                if cell_interpolant.differential_equation_type != DifferentialEquationType.ODE:
                    raise ValueError("Cell stochastic interpolant must be ODE type for SDE PPO.")
                if not cell_interpolant.predict_velocity:
                    raise ValueError("Cell stochastic interpolant must predict velocity for SDE PPO.")
                if cell_interpolant.velocity_annealing_factor != 0.0:
                    raise ValueError("Cell velocity annealing is not supported for SDE PPO.")
                self.cell_corrector = cell_interpolant.get_corrector()
            else:
                raise ValueError("Unsupported stochastic interpolant for cell integration.")
        assert ((self.integrate_cell and self.cell_corrector is not None)
                or (not self.integrate_cell and self.cell_corrector is None))

        self.integration_time_steps = base_model.si.integration_time_steps

        # Validate relative costs.
        required_costs = ["pos_policy", "pos_regularization"]
        for cost_name in required_costs:
            if cost_name not in relative_costs:
                raise ValueError(f"Missing required relative cost: {cost_name}")
        if "pos_eta_distillation" not in relative_costs:
            relative_costs = {**relative_costs, "pos_eta_distillation": 0.0}
        if not all(cost >= 0.0 for cost in relative_costs.values()):
            raise ValueError("All relative costs must be non-negative.")
        sum_costs = sum(relative_costs.values())
        if sum_costs == 0.0:
            raise ValueError("All relative costs are zero.")
        if normalize_relative_costs:
            relative_costs = {k: v / sum_costs for k, v in relative_costs.items()}
        self.relative_costs = relative_costs

        # GRPO settings.
        if grpo_group_size <= 1:
            raise ValueError("GRPO group size must be > 1.")
        if grpo_num_groups <= 0:
            raise ValueError("GRPO num groups must be > 0.")
        self.grpo_group_size = grpo_group_size
        self.grpo_num_groups = grpo_num_groups
        self.grpo_share_x_0 = grpo_share_x_0
        base_modules["datamodule"].train_batch_size = grpo_num_groups

        # PPO settings.
        self.ppo_clip_epsilon = ppo_clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        # Parse position normalization mode.
        try:
            self.position_normalization = self.PositionNormalization[position_normalization.upper()]
        except KeyError:
            valid_options = [name.lower() for name in self.PositionNormalization.__members__.keys()]
            raise ValueError(f"Invalid position_normalization: {position_normalization}. "
                             f"Must be one of {valid_options}.")

        self.generation_xyz_filename = generation_xyz_filename
        self.enable_progress_bar = enable_progress_bar

        if store_validation_structures_path is not None:
            if not store_validation_structures_path.endswith(".xyz"):
                raise ValueError("store_validation_structures_path must be an .xyz file.")
        self.store_validation_structures_path = store_validation_structures_path

    def setup(self, stage: str) -> None:
        """Set up datasets for reward computation."""
        if stage == "fit":
            self.reward.set_train_dataset(self.trainer.datamodule.train_dataset)
            self.reward.set_val_dataset(self.trainer.datamodule.val_dataset)
        elif stage == "validate":
            self.reward.set_val_dataset(self.trainer.datamodule.val_dataset)
        elif stage == "predict":
            self.reward.set_pred_dataset(self.trainer.datamodule.pred_dataset)

    @torch.no_grad()
    def rollout(self, x_0: OMGData) -> SDETrajectoryData:
        """
        Generate trajectory using SDE dynamics without gradients.

        SDE: dx = [b - (ε/γ)·η] dt + √(2ε) dW
        """
        base_model = base_modules["model"].model
        batch_size = len(x_0.n_atoms)
        times = torch.linspace(SMALL_TIME, BIG_TIME, self.integration_time_steps, device=self.device)
        x_t = x_0.clone()

        states = []
        old_log_probs_atoms = []
        actions = []
        base_outputs = []

        for t_index in trange(1, len(times), desc="SDE Rollout", position=1, leave=False,
                              disable=not self.enable_progress_bar):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            sqrt_dt = torch.sqrt(dt)
            time = t.repeat(batch_size)

            # Get epsilon and gamma at this time.
            eps_t = self.epsilon.epsilon(t)  # Scalar.
            gamma_t = self.gamma.gamma(t)  # Scalar.

            # Get base model outputs for distillation and cell integration.
            base_output = base_model(x_t, time)
            b_base = base_output[DataField.pos.name + "_b"]
            eta_base = base_output[DataField.pos.name + "_eta"]

            # Get policy model outputs (b, eta).
            policy_output = self.policy_model(x_t, time)
            b_policy = policy_output[DataField.pos.name + "_b"]
            eta_policy = policy_output[DataField.pos.name + "_eta"]

            states.append(x_t.clone())
            base_outputs.append((b_base, eta_base))

            # Compute drift: f = b - (ε/γ)·η
            drift = b_policy - (eps_t / gamma_t) * eta_policy

            # Sample noise for SDE: √(2ε) dW = √(2ε·dt) · z where z ~ N(0,1).
            z = torch.randn_like(b_policy)
            diffusion_coef = torch.sqrt(2.0 * eps_t)
            noise_term = diffusion_coef * sqrt_dt * z

            action = drift * dt + noise_term
            actions.append(action)

            # Log probability: action ~ N(drift*dt, 2ε·dt·I), so log p(action) uses z² since noise = √(2ε·dt)·z.
            # Sum log probs over x, y, and z yielding tensor of shape (sum(n_atoms),).
            log_prob_per_atom = -0.5 * (torch.log(2.0 * torch.pi * 2.0 * eps_t * dt) + (z ** 2)).sum(dim=-1)
            old_log_probs_atoms.append(log_prob_per_atom)

            # Euler-Maruyama update.
            x_t.pos = self.pos_corrector.correct(x_t.pos + drift * dt + noise_term)

            # Cell integration using base model (no RL).
            if self.integrate_cell:
                cell_b = base_output[DataField.cell.name + "_b"]
                # Use ODE for cell (no noise).
                x_t.cell = self.cell_corrector.correct(x_t.cell + cell_b * dt)

        # Append final state.
        states.append(x_t.clone())

        # Convert final structures and compute rewards.
        x_1 = x_t.to("cpu")
        structures = []
        for i in range(len(x_1.n_atoms)):
            sl = slice(x_1.ptr[i], x_1.ptr[i + 1])
            structures.append(Structure(
                cell=x_1.cell[i, :, :],
                atomic_numbers=x_1.species[sl],
                pos=x_1.pos[sl, :],
                pos_is_fractional=x_1.pos_is_fractional[i],
            ))

        rewards, info_dict = self.reward.compute(
            structures, Reward.ComputeStage.TRAIN, enable_progress_bar=self.enable_progress_bar
        )
        rewards = torch.tensor(rewards, dtype=self.dtype, device=self.device)
        info_dict = {k: torch.tensor(v, dtype=self.dtype, device=self.device).mean() for k, v in info_dict.items()}

        # Compute GRPO advantages.
        # Partial batch possible.
        assert len(rewards) % self.grpo_group_size == 0
        assert len(rewards) // self.grpo_group_size <= self.grpo_num_groups
        advantages = torch.zeros_like(rewards)
        for i in range(len(rewards) // self.grpo_group_size):
            sl = slice(i * self.grpo_group_size, (i + 1) * self.grpo_group_size)
            group_rewards = rewards[sl]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std(unbiased=False)
            advantages[sl] = (group_rewards - group_mean) / (group_std + 1.0e-8)

        return SDETrajectoryData(
            states=states,
            actions=actions,
            old_log_probs_atoms=old_log_probs_atoms,
            base_outputs=base_outputs,
            rewards=rewards,
            advantages=advantages,
            info_dict=info_dict,
        )

    def ppo_update(self, trajectory: SDETrajectoryData) -> dict[str, float]:
        """
        Perform PPO update using stored trajectory data.

        Loss = policy_loss + kl_regularization + eta_distillation
        """
        batch_size = len(trajectory.rewards)
        times = torch.linspace(SMALL_TIME, BIG_TIME, self.integration_time_steps, device=self.device)
        num_timesteps = len(times) - 1

        total_policy_loss = 0.0
        total_reg_loss = 0.0
        total_distill_loss = 0.0
        total_clip_fraction = 0.0

        opt = self.optimizers()
        opt.zero_grad()

        for t_index in trange(num_timesteps, desc="Perform PPO Update", position=1, leave=False,
                              disable=not self.enable_progress_bar):
            t = times[t_index]
            dt = times[t_index + 1] - times[t_index]
            time = t.repeat(batch_size)

            eps_t = self.epsilon.epsilon(t)
            gamma_t = self.gamma.gamma(t)

            x_t = trajectory.states[t_index]
            action = trajectory.actions[t_index]
            old_log_probs_atoms = trajectory.old_log_probs_atoms[t_index]
            b_base, eta_base = trajectory.base_outputs[t_index]

            # Re-evaluate policy model with gradients.
            policy_output = self.policy_model(x_t, time)
            b_policy = policy_output[DataField.pos.name + "_b"]
            eta_policy = policy_output[DataField.pos.name + "_eta"]

            # The action distribution is N(drift*dt, 2ε·dt·I).
            # We stored the rollout action, so evaluate its log prob under the current drift.
            drift_current = b_policy - (eps_t / gamma_t) * eta_policy
            drift_base = b_base.detach() - (eps_t / gamma_t) * eta_base.detach()

            # Current log prob: log N(action; drift_current * dt, 2ε·dt·I).
            # = -0.5 * [d*log(2π·2ε·dt) + ||action - drift_current*dt||² / (2ε·dt)]
            var = 2.0 * eps_t * dt
            # Action basically stores x_t+dt - x_t.
            # Sum log probs over x, y, z. Shape: (sum(n_atoms),).
            diff = action - drift_current * dt
            current_log_probs_atoms = -0.5 * (
                torch.log(2.0 * torch.pi * var) + (diff ** 2) / var
            ).sum(dim=-1)

            # PPO ratio and clipped surrogate based on position normalization mode.
            if self.position_normalization == self.PositionNormalization.NONE:
                # Sum log probs over all atoms in each structure to get batch-wise log probs.
                old_log_prob = scatter_add(old_log_probs_atoms, x_t.batch)
                current_log_prob = scatter_add(current_log_probs_atoms, x_t.batch)
                ratio = torch.exp(current_log_prob - old_log_prob.detach())
                clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon)
                clip_fraction = ((ratio < 1.0 - self.ppo_clip_epsilon)
                                 | (ratio > 1.0 + self.ppo_clip_epsilon)).float().mean()
                policy_loss = -torch.min(ratio * trajectory.advantages,
                                         clipped_ratio * trajectory.advantages).mean()
            elif self.position_normalization == self.PositionNormalization.PER_ATOM_SURROGATE:
                # Per-atom ratio, averaged per structure.
                ratio_atoms = torch.exp(current_log_probs_atoms - old_log_probs_atoms.detach())
                clipped_ratio_atoms = torch.clamp(ratio_atoms, 1.0 - self.ppo_clip_epsilon,
                                                  1.0 + self.ppo_clip_epsilon)
                clip_mask_atoms = ((ratio_atoms < 1.0 - self.ppo_clip_epsilon)
                                   | (ratio_atoms > 1.0 + self.ppo_clip_epsilon)).float()
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
                clip_fraction = ((ratio < 1.0 - self.ppo_clip_epsilon)
                                 | (ratio > 1.0 + self.ppo_clip_epsilon)).float().mean()
                # Weight advantages by 1 / n_atoms so that large structures do not dominate.
                weighted_advantages = trajectory.advantages / x_t.n_atoms.to(dtype=ratio.dtype)
                policy_loss = -torch.min(ratio * weighted_advantages,
                                         clipped_ratio * weighted_advantages).mean()

            # KL regularization: KL(current || base) for the drift distribution.
            # KL(N(μ1, σ²I) || N(μ2, σ²I)) = ||μ1 - μ2||² / (2σ²)
            # Here μ1 = drift_current * dt, μ2 = drift_base * dt, σ² = 2ε·dt.
            kl_per_dim = ((drift_current - drift_base.detach()) ** 2 * dt * dt) / (2.0 * var)
            kl_per_atom = kl_per_dim.sum(dim=-1)
            if self.position_normalization == self.PositionNormalization.NONE:
                # Sum over atoms to get per-structure KL and take mean over batch.
                reg_loss = scatter_add(kl_per_atom, x_t.batch).mean()
            else:
                # Average over atoms to get per-structure KL and take mean over batch.
                reg_loss = scatter_mean(kl_per_atom, x_t.batch).mean()

            # Eta distillation loss: keep eta close to base model's eta.
            eta_diff = (eta_policy - eta_base.detach()) ** 2
            distill_per_atom = eta_diff.sum(dim=-1)  # Sum over x, y, z.
            distill_per_struct = scatter_mean(distill_per_atom, x_t.batch)
            distill_loss = distill_per_struct.mean()

            # Combined loss.
            timestep_loss = (
                self.relative_costs["pos_policy"] * policy_loss
                + self.relative_costs["pos_regularization"] * reg_loss
                + self.relative_costs["pos_eta_distillation"] * distill_loss
            )

            scaled_loss = timestep_loss / num_timesteps
            self.manual_backward(scaled_loss)

            total_policy_loss += policy_loss.item()
            total_reg_loss += reg_loss.item()
            total_distill_loss += distill_loss.item()
            total_clip_fraction += clip_fraction.item()

        if self.gradient_clip_val is not None:
            self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_val,
                                gradient_clip_algorithm=self.gradient_clip_algorithm)
        opt.step()

        return {
            "pos_policy": total_policy_loss / num_timesteps,
            "pos_regularization": total_reg_loss / num_timesteps,
            "pos_eta_distillation": total_distill_loss / num_timesteps,
            "pos_clip_fraction": total_clip_fraction / num_timesteps,
        }

    @torch.no_grad()
    def integrate(self, x_0: OMGData) -> OMGData:
        """Integrate using SDE dynamics for validation/prediction."""
        base_model = base_modules["model"].model
        assert base_model is not None
        batch_size = len(x_0.n_atoms)
        times = torch.linspace(SMALL_TIME, BIG_TIME, self.integration_time_steps, device=self.device)
        x_t = x_0.clone()

        for t_index in trange(1, len(times), desc="SDE Integration", position=1, leave=False,
                              disable=not self.enable_progress_bar):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            sqrt_dt = torch.sqrt(dt)
            time = t.repeat(batch_size)

            eps_t = self.epsilon.epsilon(t)
            gamma_t = self.gamma.gamma(t)

            # Get output before updating x_t.pos.
            base_output = base_model(x_t, time) if self.integrate_cell else None
            policy_output = self.policy_model(x_t, time)
            b_policy = policy_output[DataField.pos.name + "_b"]
            eta_policy = policy_output[DataField.pos.name + "_eta"]

            drift = b_policy - (eps_t / gamma_t) * eta_policy
            diffusion = torch.sqrt(2.0 * eps_t) * sqrt_dt * torch.randn_like(b_policy)

            x_t.pos = self.pos_corrector.correct(x_t.pos + drift * dt + diffusion)

            if self.integrate_cell:
                assert base_output is not None
                cell_b = base_output[DataField.cell.name + "_b"]
                x_t.cell = self.cell_corrector.correct(x_t.cell + cell_b * dt)

        return x_t

    def training_step(self, batch: OMGData, batch_idx: int) -> None:
        if self.grpo_share_x_0:
            x_0_per_structure = base_modules["model"].sampler.sample_p_0(batch).to(self.device)
            x_0 = Batch.from_data_list(
                [x_0_per_structure[i // self.grpo_group_size]
                 for i in range(self.grpo_group_size * len(batch))]
            ).to(self.device)
        else:
            grpo_batch = Batch.from_data_list(
                [batch[i // self.grpo_group_size] for i in range(self.grpo_group_size * len(batch))]
            ).to(self.device)
            x_0 = base_modules["model"].sampler.sample_p_0(grpo_batch).to(self.device)

        trajectory = self.rollout(x_0)

        all_losses = {}
        for epoch in range(self.ppo_epochs):
            losses = self.ppo_update(trajectory)
            for key, value in losses.items():
                all_losses[key] = all_losses.get(key, 0.0) + value

        for key in all_losses:
            all_losses[key] /= self.ppo_epochs

        rewards = trajectory.rewards
        self.log_dict(all_losses, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch))
        self.log("reward_mean", rewards.mean(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log("reward_std", rewards.std(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log_dict(trajectory.info_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                      batch_size=len(batch))

    def validation_step(self, batch: OMGData, batch_idx: int) -> None:
        x_0 = base_modules["model"].sampler.sample_p_0(batch).to(self.device)
        x_1 = self.integrate(x_0)

        x_1 = x_1.to("cpu")
        structures = []
        for i in range(len(x_1.n_atoms)):
            sl = slice(x_1.ptr[i], x_1.ptr[i + 1])
            structures.append(Structure(
                cell=x_1.cell[i, :, :].detach(),
                atomic_numbers=x_1.species[sl].detach(),
                pos=x_1.pos[sl, :].detach(),
                pos_is_fractional=x_1.pos_is_fractional[i],
            ))

        rewards, info_dict = self.reward.compute(
            structures, Reward.ComputeStage.VAL, enable_progress_bar=self.enable_progress_bar
        )
        rewards = torch.tensor(rewards, dtype=self.dtype, device=self.device)
        info_dict = {f"val_{k}": torch.tensor(v, dtype=self.dtype, device=self.device).mean()
                     for k, v in info_dict.items()}

        self.log("val_reward_mean", rewards.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log("val_reward_std", rewards.std(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(batch))
        self.log_dict(info_dict, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch))

        if self.store_validation_structures_path is not None:
            filename = Path(self.store_validation_structures_path)
            epoch_filename = filename.with_stem(f"{filename.stem}_epoch_{self.current_epoch:04d}")
            if epoch_filename.exists() and batch_idx == 0:
                epoch_filename.unlink()
            xyz_saver(x_1, epoch_filename)

    def predict_step(self, batch: OMGData) -> OMGData:
        x_0 = base_modules["model"].sampler.sample_p_0(batch).to(self.device)
        x_1 = self.integrate(x_0)

        filename = (Path(self.generation_xyz_filename) if self.generation_xyz_filename is not None
                    else Path(f"{strftime('%Y%m%d-%H%M%S')}.xyz"))
        init_filename = filename.with_stem(filename.stem + "_init")
        xyz_saver(x_0.to("cpu"), init_filename)
        xyz_saver(x_1.to("cpu"), filename)

        return x_1
