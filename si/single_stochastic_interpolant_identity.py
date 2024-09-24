from typing import Optional
import torch
from .abstracts import Interpolant, LatentGamma, StochasticInterpolant
from enum import Enum, auto
import torch.nn as nn


class SingleStochasticInterpolantIdentity(StochasticInterpolant):
    """
    Stochastic interpolant x_t = x_0 = x_1, between two points from two distributions p_0 and p_1 at
    times t. Differs from the SingleStochasticInterpolant class insofar as the quantity represented 
    by x_0 and x_1 (such as atom types) must remain constant during interpolation.
    """

    def __init__(self) -> None:
        """Construct stochastic interpolant."""
        super().__init__()


    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.tensor:
        """
        Stochastically interpolate between two points from two distributions p_0 and p_1 at times t.

        :param t:
            Times in [0,1].
        :type t: torch.tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.tensor, must be same as x_0

        :return:
            Stochastically interpolated value.
        :rtype: torch.tensor
        """
        assert torch.equal(x_0, x_1)
        return x_0


    def loss(self, model_prediction: tuple[torch.tensor, torch.tensor], t: torch.tensor, x_0: torch.tensor,
             x_1: torch.tensor) -> torch.tensor:
        """
        Compute the loss for the stochastic interpolant.

        :param model_prediction:
            Model prediction for the velocity field b and the denoiser eta.
        :type model_prediction: tuple[torch.tensor, torch.tensor]
        :param t:
            Times in [0,1].
        :type t: torch.tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.tensor, must be same as x_0

        :return:
            Loss.
        :rtype: torch.tensor
        """

        # Compute ground truth
        assert torch.equal(x_0, x_1)
        return torch.zeros_like(x_0)
