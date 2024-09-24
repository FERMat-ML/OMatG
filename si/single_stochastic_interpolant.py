from typing import Optional, Callable
import torch
from .abstracts import Interpolant, LatentGamma, StochasticInterpolant, Epsilon
from enum import Enum, auto
import torch.nn as nn
import numpy as np

# Integrating DE
from scipy.integrate import solve_ivp
import sdeint

class DE(Enum):
    ODE = auto()
    SDE = auto()


class SingleStochasticInterpolant(StochasticInterpolant):
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

    def __init__(self, interpolant: Interpolant, gamma: Optional[LatentGamma], eps: Epsilon, de_type: DE, use_pbc=True) -> None:
        """Construct stochastic interpolant."""
        super().__init__()
        self._interpolant = interpolant
        self._gamma = gamma
        self._eps = eps
        self._de_type = de_type
        if self._de_type == DE.ODE:
            self.loss = self._ode_loss
            self.integrate = self._ode_integrate
        else:
            assert self._de_type == DE.SDE
            self.loss = self._sde_loss
            self.integrate = self._sde_integrate

        # PBC
        if use_pbc:
            self._use_pbc = True

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

    def loss(self, model_prediction: tuple[torch.tensor, torch.tensor], t: torch.tensor, x_0: torch.tensor,
             x_1: torch.tensor) -> torch.tensor:
        """
        Compute the loss for the stochastic interpolant.

        This method is only defined here to define all methods of the abstract base class. The actual loss method is
        either _ode_loss or _sde_loss, which are chosen based on the type of differential equation (self._de_type).

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
        :type x_1: torch.tensor

        :return:
            Loss.
        :rtype: torch.tensor
        """
        raise NotImplementedError

    def _ode_loss(self, pred: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor):
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

    def _sde_loss(self, pred: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor):
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
    
    def _ode_integrate(self, wrapper: Callable[[tuple], tuple], x_t: tuple, tspan: tuple):
        """
        Integrate ODE

        :param wrapper:
            Wrapper function for getting model prediction
        :type Callable
        :param x_t:
            Old timestep x
        :type x_t: tuple
        :param tspan:
            Time span to integrate 
        :type tspan: tuple

        :return:
            new x at new t
        :rtype: torch.tensor
        """

        # Modify wrapper to only use b(t,x)
        ode_wrap = lambda t, x : wrapper(t, x)[0]

        # Integrate with scipy IVP integrator
        x_t_new = solve_ivp(ode_wrap, tspan, x_t)

        # Account for periodic boundaries
        if self._use_pbc:
            x_t_new %= 1.0

        # Return
        return torch.Tensor(x_t_new[:,-1])

    def _sde_integrate(self, wrapper: Callable[[tuple], tuple], x_t: tuple, tspan: tuple):
        """
        Integrate SDE

        :param wrapper:
            Wrapper function for getting model prediction
        :type Callable
        :param x_t
            Old timestep x
        :type x_t: tuple
        """

        # Modify wrapper for use in SDE integrator
        def f(x, t): 
            preds = wrapper(t, x)
            out = preds[0] - (self._eps.epsilon(t) / self._gamma.gamma(t)) * preds[1]
            return out

        def G(x, t):
            out = np.sqrt(2 * self._eps.epsilon(t)) * np.eye(x.shape[-1])
            return out

        # SDE Integrator
        x_t_new = sdeint.itoEuler(f, G, x_t, tspan)

        # Account for periodic boundaries
        if self._use_pbc:
            x_t_new %= 1.0
        
        # Return
        return torch.Tensor(x_t_new[:,-1])