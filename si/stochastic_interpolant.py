from typing import Optional
import torch
from .abstracts import Interpolant, LatentGamma, TimeChecker
from enum import Enum, auto
import torch.nn as nn

class DE(Enum):
    ODE = auto()
    SDE = auto()

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

    def __init__(self, interpolant: Interpolant, gamma: Optional[LatentGamma], de_type: DE) -> None:
        """Construct stochastic interpolant."""
        self._interpolant = interpolant
        self._gamma = gamma
        if de_type == DE.ODE:
            self.loss = self.__ode_loss
            self.integrate = self.__ode_integrate
        else:
            self.loss = self.__sde_loss
            self.integrate = self.__sde_integrate

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

    def __ode_loss(self, pred: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor):
        """
        Compute loss function for stochastic interpolant under ODE

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
            Loss function value.
        :rtype: torch.tensor 
        """

        # Compute ground truth
        gt = self.interpolate_derivative(t, x_0, x_1)
        loss = nn.MSELoss(gt, pred[0])
        return loss

    def __sde_loss(self, pred: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor):
        """
        Compute loss function for stochastic interpolant under SDE

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
            Tuple of loss function values.
        :rtype: (torch.tensor, torch.tensor)
        """

        # Compute ground truth
        gt_b = self.interpolate_derivative(t, x_0, x_1)
        gt_z = torch.randn_like(t)
        loss_b = nn.MSELoss(gt_b, pred[0])
        loss_z = nn.MSELoss(gt_z, pred[1])
        return loss_b + loss_z

    