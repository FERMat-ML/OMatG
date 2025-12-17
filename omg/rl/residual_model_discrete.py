"""
Residual model extensions for discrete flow matching (species).

This module handles residual learning for discrete variables (atomic species)
using rate matrix corrections instead of velocity fields.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from omg.globals import MAX_ATOM_NUM


class DiscreteResidualMixin:
    """
    Mixin for ResidualModel to handle discrete species with rate matrices.

    This extends the standard continuous residual model to support discrete
    flow matching for atomic species.
    """

    def forward_discrete_species(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        return_log_prob: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for discrete species (rate matrix output).

        Instead of outputting velocity fields, this outputs logits over species
        which define a rate matrix correction.

        :param x: Current structure state
        :type x: torch.Tensor
        :param t: Time values
        :type t: torch.Tensor
        :param return_log_prob: Whether to compute log probabilities
        :type return_log_prob: bool

        :return: Tuple of (logits, log_prob, mean_logits)
        :rtype: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
        """
        # Get logits from network (shape: [n_atoms, MAX_ATOM_NUM])
        # These represent the residual rate matrix
        output = self.network(x, t)

        # Extract species logits (should be key like 'species_b' or similar)
        species_key = None
        for key in output.keys():
            if 'species' in key.lower():
                species_key = key
                break

        if species_key is None:
            raise ValueError("No species output found in network")

        mean_logits = output[species_key]  # Shape: [n_atoms, MAX_ATOM_NUM]

        if self.training and return_log_prob:
            # Add Gumbel noise for stochastic policy
            # Gumbel-Softmax allows discrete sampling with gradients
            temperature = self.noise_scale

            # Sample Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(mean_logits) + 1e-20) + 1e-20)
            noisy_logits = (mean_logits + gumbel_noise) / temperature

            # Log probability of the sample
            # For Gumbel-Softmax, this is approximate
            probs = F.softmax(mean_logits, dim=-1)
            sampled_probs = F.softmax(noisy_logits, dim=-1)

            # KL divergence as log prob (negative)
            log_prob = -(sampled_probs * (torch.log(sampled_probs + 1e-20) - torch.log(probs + 1e-20))).sum(dim=-1)

            output_copy = output.copy()
            output_copy[species_key] = noisy_logits

            return output_copy, log_prob, output
        else:
            # Inference: deterministic (no noise)
            return output, None, output


class ResidualRateMatrix(nn.Module):
    """
    Residual model specifically for discrete flow matching with rate matrices.

    This model learns corrections to the base rate matrix for species transitions.

    :param base_architecture: Architecture for the residual network
    :type base_architecture: nn.Module
    :param noise_scale: Temperature for Gumbel-Softmax during training
    :type noise_scale: float
    :param regularization_weight: Weight for regularization loss
    :type regularization_weight: float
    :param mask_index: Index for masked species (default 0)
    :type mask_index: int
    """

    def __init__(
        self,
        base_architecture: nn.Module,
        noise_scale: float = 1.0,  # Temperature for Gumbel-Softmax
        regularization_weight: float = 0.01,
        mask_index: int = 0,
    ):
        """Constructor for ResidualRateMatrix."""
        super().__init__()

        self.network = base_architecture
        self.noise_scale = noise_scale  # Temperature for Gumbel-Softmax
        self.regularization_weight = regularization_weight
        self.mask_index = mask_index

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        return_log_prob: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass for discrete species.

        :param x: Current structure
        :param t: Time values
        :param return_log_prob: Whether to return log probabilities

        :return: Tuple of (output_dict, log_prob, mean_output_dict)
        """
        # Get network output (logits over species)
        output = self.network(x, t)

        # Find species key
        species_key = self._find_species_key(output)

        mean_logits = output[species_key]  # Shape: [n_atoms, MAX_ATOM_NUM]

        if self.training and return_log_prob:
            # Gumbel-Softmax for stochastic discrete sampling
            temperature = self.noise_scale

            # Sample from Gumbel-Softmax
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(mean_logits) + 1e-20) + 1e-20)
            noisy_logits = (mean_logits + gumbel_noise) / temperature

            # Compute log probability
            # Use categorical distribution log prob
            probs = F.softmax(mean_logits, dim=-1)
            log_probs_per_atom = torch.log(probs.max(dim=-1)[0] + 1e-20)
            log_prob = log_probs_per_atom  # Shape: [n_atoms]

            # Create output with noisy logits
            output_noisy = {k: v for k, v in output.items()}
            output_noisy[species_key] = noisy_logits

            return output_noisy, log_prob, output
        else:
            # Inference: deterministic
            return output, None, output

    def _find_species_key(self, output: Dict[str, torch.Tensor]) -> str:
        """Find the species key in output dictionary."""
        for key in output.keys():
            if 'species' in key.lower():
                return key
        raise ValueError("No species output found in network")

    def compute_regularization_loss(
        self,
        mean_output: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute regularization loss for discrete residuals.

        For discrete flow matching, we regularize the deviation from uniform
        distribution (encourages small rate matrix corrections).

        :param mean_output: Mean output from network
        :return: Regularization loss
        """
        species_key = self._find_species_key(mean_output)
        logits = mean_output[species_key]  # Shape: [n_atoms, MAX_ATOM_NUM]

        # Encourage residual to be close to uniform (small correction)
        # KL divergence from uniform distribution
        probs = F.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / MAX_ATOM_NUM

        kl_div = (probs * (torch.log(probs + 1e-20) - torch.log(uniform + 1e-20))).sum(dim=-1).mean()

        return self.regularization_weight * kl_div

    def set_noise_scale(self, noise_scale: float):
        """
        Update the temperature (noise scale) for Gumbel-Softmax.

        :param noise_scale: New temperature value
        """
        self.noise_scale = noise_scale

    def integrate_with_detailed_balance(
        self,
        base_probs: torch.Tensor,
        residual_logits: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor,
        time_step: torch.Tensor,
        noise_param: float = 0.0,
    ) -> torch.Tensor:
        """
        Integrate discrete species using combined base + residual rate matrices
        with detailed balance noise (as in DiscreteFlowMatchingMask).

        :param base_probs: Base model probabilities [n_atoms, MAX_ATOM_NUM]
        :param residual_logits: Residual logits [n_atoms, MAX_ATOM_NUM]
        :param x_t: Current species [n_atoms]
        :param time: Current time
        :param time_step: Integration time step
        :param noise_param: Noise parameter (eta in detailed balance)

        :return: Updated species
        """
        # Combine base and residual
        residual_probs = F.softmax(residual_logits, dim=-1)

        # Simple combination: weighted sum (could use other strategies)
        # You might want to make this more sophisticated
        alpha = 0.1  # Weight for residual (could be learned or adaptive)
        combined_probs = (1 - alpha) * base_probs + alpha * residual_probs

        # Ensure probabilities sum to 1
        combined_probs = combined_probs / combined_probs.sum(dim=-1, keepdim=True)

        # Sample target species from combined distribution
        from torch.distributions import Categorical
        x_1 = Categorical(combined_probs).sample() + 1  # +1 for species indexing

        # Detailed balance dynamics (from DiscreteFlowMatchingMask)
        # Unmask with rate from combined model
        will_unmask = torch.rand(x_t.shape, device=x_t.device) < (
            time_step * (1.0 + noise_param * time) / (1.0 - time + 1e-8)
        )
        will_unmask = will_unmask & (x_t == self.mask_index)

        # Final step: unmask all
        from omg.globals import BIG_TIME
        if abs(time + time_step - BIG_TIME) < 5e-3:
            will_unmask = (x_t == self.mask_index)

        # Re-mask with noise (detailed balance)
        will_mask = torch.rand(x_t.shape, device=x_t.device) < time_step * noise_param
        will_mask = will_mask & (x_t != self.mask_index)

        # Update species
        x_t = x_t.clone()
        x_t[will_unmask] = x_1[will_unmask]

        if abs(time + time_step - BIG_TIME) >= 5e-3:
            x_t[will_mask] = self.mask_index

        return x_t


def combine_rate_matrices(
    base_probs: torch.Tensor,
    residual_logits: torch.Tensor,
    combination_mode: str = 'weighted_sum',
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Combine base and residual rate matrices.

    :param base_probs: Probabilities from base model [n_atoms, MAX_ATOM_NUM]
    :param residual_logits: Logits from residual model [n_atoms, MAX_ATOM_NUM]
    :param combination_mode: How to combine ('weighted_sum', 'product', 'geometric_mean')
    :param alpha: Weight for residual in weighted sum mode

    :return: Combined probabilities
    """
    residual_probs = F.softmax(residual_logits, dim=-1)

    if combination_mode == 'weighted_sum':
        # Linear interpolation
        combined = (1 - alpha) * base_probs + alpha * residual_probs

    elif combination_mode == 'product':
        # Product of experts
        combined = base_probs * residual_probs
        combined = combined / combined.sum(dim=-1, keepdim=True)

    elif combination_mode == 'geometric_mean':
        # Geometric mean
        combined = torch.sqrt(base_probs * residual_probs)
        combined = combined / combined.sum(dim=-1, keepdim=True)

    else:
        raise ValueError(f"Unknown combination mode: {combination_mode}")

    return combined
