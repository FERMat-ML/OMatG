from typing import Optional
import torch
from .abstracts import Interpolant, LatentGamma, TimeChecker


class StochasticInterpolant(TimeChecker):
    """
    Stochastic interpolant x_t = I(t, x_0, x_1) + gamma(t) * z between two points from two distributions p_0 and p_1 at
    times t based on an interpolant I(t, x_0, x_1), a gamma function gamma(t), and a Gaussian random variable z.

    :param interpolant:
        Interpolant I(t, x_0, x_1) between two points from two distributions p_0 and p_1 at times t.
    :type interpolant: Interpolant
    :param gamma:
        Gamma function gamma(t) in the latent variable gamma(t) * z of a stochastic interpolant.
    :type gamma: LatentGamma
    """

    def __init__(self, interpolant: Interpolant, gamma: Optional[LatentGamma]) -> None:
        """Construct stochastic interpolant."""
        self._interpolant = interpolant
        self._gamma = gamma

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
        :type x_1: torch.tensor

        :return:
            Stochastically interpolated value.
        :rtype: torch.tensor
        """
        assert x_0.shape == x_1.shape
        interpolate = self._interpolant.interpolate(t, x_0, x_1)
        if self._gamma is not None:
            interpolate += self._gamma.gamma(t) * torch.randn_like(t)
        return interpolate

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.tensor:
        """
        Derivative with respect to time of the stochastic interpolants between two points from two distributions p_0
        and p_1 at times t.

        :param t:
            Times in [0,1].
        :type t: torch.tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.tensor

        :return:
            Stochastically interpolated value.
        :rtype: torch.tensor
        """
        assert x_0.shape == x_1.shape
        self._check_t(t)
        interpolate_derivative = self._interpolant.interpolate_derivative(t, x_0, x_1)
        if self._gamma is not None:
            interpolate_derivative += self._gamma.gamma_derivative(t) * torch.randn_like(t)
        return interpolate_derivative
