import torch
import torch.nn as nn
from typing import Optional, Tuple
from omg.model.encoders.cspnet_full import CSPNetFull


class ResidualModel(nn.Module):
    """
    Residual model that learns small velocity corrections to apply during inference.

    This model outputs a mean residual velocity that can be made stochastic during training
    for policy gradient methods, but remains deterministic during inference.

    :param base_architecture: The architecture to use for the residual model (same type as base model)
    :type base_architecture: Model
    :param noise_scale: Standard deviation of Gaussian noise added to residual velocities during training
    :type noise_scale: float
    :param regularization_weight: Weight for L2 regularization on residual magnitudes
    :type regularization_weight: float
    """

    # TODO: GENERALIZE TO OTHER ARCHITECTURES BESIDES CSPNetFull?
    # TODO: IS REGULARIZATION REALLY NEEDED HERE?
    def __init__(self, base_architecture: CSPNetFull, noise_scale: float = 0.1, regularization_weight: float = 0.01):
        """Constructor for the ResidualModel class."""
        super().__init__()
        self.network = base_architecture
        self.noise_scale = noise_scale
        self.regularization_weight = regularization_weight

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                return_log_prob: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the residual model.

        During training (self.training=True), adds Gaussian noise to enable policy gradients.
        During inference (self.training=False), returns deterministic mean residual.

        :param x: Current structure state
        :type x: torch.Tensor
        :param t: Time values
        :type t: torch.Tensor
        :param return_log_prob: Whether to compute and return log probabilities
        :type return_log_prob: bool

        :return: Tuple of (residual velocities, log probabilities, mean residuals)
            - residual velocities: The actual residual to apply (with noise during training)
            - log probabilities: Log prob of the sampled residuals (None during inference)
            - mean residuals: The deterministic mean output (for regularization)
        :rtype: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
        """
        # Get mean residual velocity from network
        # The network outputs predictions for each data field (pos_b, cell_b, etc.)
        output = self.network(x, t)

        if self.training and return_log_prob:
            # Training mode: add noise for exploration and policy gradients
            noisy_output = output.clone()
            log_probs = {}

            # Add noise to each velocity field and compute log probs
            for key in output.keys():
                if key.endswith('_b'):  # Velocity fields
                    mean = output[key]
                    noise = torch.randn_like(mean)
                    noisy_output[key] = mean + self.noise_scale * noise

                    # Log probability: -0.5 * ||noise||^2
                    # Sum over all dimensions except batch
                    log_prob = -0.5 * (noise ** 2).sum(dim=tuple(range(1, noise.ndim)))
                    log_probs[key] = log_prob

            # Combine log probs (sum log probs = log of product of probs)
            total_log_prob = sum(log_probs.values())

            return noisy_output, total_log_prob, output
        else:
            # Inference mode: deterministic
            return output, None, output

    def compute_regularization_loss(self, mean_residuals: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 regularization loss on residual magnitudes to encourage small residuals.

        :param mean_residuals: Mean residual velocities (output from network)
        :type mean_residuals: torch.Tensor

        :return: Regularization loss
        :rtype: torch.Tensor
        """
        reg_loss = 0.0

        # Regularize each velocity field
        for key in mean_residuals.keys():
            if key.endswith('_b'):  # Velocity fields
                reg_loss = reg_loss + (mean_residuals[key] ** 2).mean()

        return self.regularization_weight * reg_loss

    def set_noise_scale(self, noise_scale: float):
        """
        Update the noise scale (useful for annealing during training).

        TODO: TEST OUT ANNEALING? GRADUALLY DECREASE NOISE SCALE DURING TRAINING

        :param noise_scale: New noise scale value
        :type noise_scale: float
        """
        self.noise_scale = noise_scale
