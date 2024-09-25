from enum import Enum, auto
import numpy as np
from scipy.integrate import solve_ivp
import sdeint  # TODO: Find better package.
import torch
import torch.nn as nn
from typing import Optional, Callable
from .abstracts import Corrector, Epsilon, Interpolant, LatentGamma, StochasticInterpolant


class DifferentialEquationType(Enum):
    """
    Enum for the possible types of differential equation that should be used by the stochastic interpolants.
    """

    ODE = auto()
    """
    Ordinary differential equation.
    """
    SDE = auto()
    """
    Stochastic differential equation.
    """


class SingleStochasticInterpolant(StochasticInterpolant):
    """
    Stochastic interpolant x_t = I(t, x_0, x_1) + gamma(t) * z between points x_0 and x_1 from two distributions p_0 and
    p_1 at times t based on an interpolant I(t, x_0, x_1), a gamma function gamma(t), and a Gaussian random variable z.

    The gamma function gamma(t) scaling the random variable z is optional.

    The stochastic interpolant can either use an ordinary differential equation (ODE) or a stochastic differential
    equation during inference. If an SDE is used, one should additionally provide an epsilon function epsilon(t).

    :param interpolant:
        Interpolant I(t, x_0, x_1) between points from two distributions p_0 and p_1 at times t.
    :type interpolant: Interpolant
    :param gamma:
        Optional gamma function gamma(t) in the latent variable gamma(t) * z of a stochastic interpolant.
    :type gamma: Optional[LatentGamma]
    :param epsilon:
        Optional epsilon function epsilon(t) for the stochastic differential equation.
        Should only be provided if the differential equation type is SDE.
    :type epsilon: Optional[Epsilon]
    :param differential_equation_type:
        Type of differential equation to use for inference.
    :type differential_equation_type: DifferentialEquationType
    :param sde_number_time_steps:
        Number of time steps for the integration of the SDE.
        Note that the time span [0, 1] is already subdivided by the StochasticInterpolants class.
        This number of timesteps will be used for the subintervals.
        Should be positive and only be provided if the differential equation type is SDE.
    :type sde_number_time_steps: Optional[int]
    :param corrector:
        Corrector that will be applied to the points x_t during integration (for instance, to enforce periodic boundary
        conditions).
        If None, no correction will be applied.
    :type corrector: Optional[Corrector]

    :raises ValueError:
        If epsilon is provided for ODEs or not provided for SDEs.
        If sde_number_time_steps is provided for ODEs or not provided for SDEs.
        If sde_number_time_steps is not positive.
    """

    def __init__(self, interpolant: Interpolant, gamma: Optional[LatentGamma], epsilon: Optional[Epsilon],
                 differential_equation_type: DifferentialEquationType, sde_number_time_steps: Optional[int] = None,
                 corrector: Optional[Corrector] = None) -> None:
        """Construct stochastic interpolant."""
        super().__init__()
        self._interpolant = interpolant
        self._gamma = gamma
        self._epsilon = epsilon
        self._differential_equation_type = differential_equation_type
        self._sde_number_time_steps = sde_number_time_steps
        self._corrector = corrector if corrector is not None else self.IdentityCorrector()
        if self._differential_equation_type == DifferentialEquationType.ODE:
            self.loss = self._ode_loss
            self.integrate = self._ode_integrate
            if self._epsilon is not None:
                raise ValueError("Epsilon function should not be provided for ODEs.")
            if self._sde_number_time_steps is not None:
                raise ValueError("SDE number of time steps should not be provided for ODEs.")
        else:
            assert self._differential_equation_type == DifferentialEquationType.SDE
            self.loss = self._sde_loss
            self.integrate = self._sde_integrate
            if self._epsilon is None:
                raise ValueError("Epsilon function should be provided for SDEs.")
            if self._sde_number_time_steps is None:
                raise ValueError("SDE number of time steps should be provided SDEs.")
            if not self._sde_number_time_steps > 0:
                raise ValueError("SDE number of time steps should be bigger than zero.")

    class IdentityCorrector(Corrector):
        """
        Corrector that does nothing.
        """

        def __init__(self):
            """Construct identity corrector."""
            super().__init__()

        def correct(self, x: torch.Tensor) -> torch.Tensor:
            """
            Correct the input x.

            :param x:
                Input to correct.
            :type x: torch.Tensor

            :return:
                Corrected input.
            :rtype: torch.Tensor
            """
            return x

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> tuple(torch.Tensor, torch.Tensor):
        """
        Stochastically interpolate between points x_0 and x_1 from two distributions p_0 and p_1 at times t.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor

        :return:
            Stochastically interpolated points x_t.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        assert x_0.shape == x_1.shape
        interpolate = self._interpolant.interpolate(t, x_0, x_1)
        z = torch.randn_like(x_0)
        if self._gamma is not None:
            interpolate += self._gamma.gamma(t) * z
        return (interpolate, z)

    def _interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, z: torch.Tensor, n_atoms: torch.Tensor) -> tuple(torch.Tensor, torch.Tensor):
        """
        Derivative with respect to time of the stochastic interpolant between points x_0 and x_1 from two distributions
        p_0 and p_1 at times t.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor
        :param z:
            Random variable
        :type z: torch.Tensor
        :param n_atoms:
            Number of atoms in crystal from batch
        :type n_atoms: torch.Tensor

        :return:
            Stochastically interpolated value.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        assert x_0.shape == x_1.shape
        self._check_t(t)
        interpolate_derivative = self._interpolant.interpolate_derivative(t, x_0, x_1, n_atoms)
        if self._gamma is not None:
            interpolate_derivative += self._gamma.gamma_derivative(t) * z
        return (interpolate_derivative, z)

    def loss(self, model_prediction: tuple[torch.Tensor, torch.Tensor], t: torch.Tensor, x_0: torch.Tensor,
             x_1: torch.Tensor, n_atoms: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
        p_1 at times t based on the model prediction for the velocity fields b and the denoisers eta.

        This method is only defined here to define all methods of the abstract base class. The actual loss method is
        either _ode_loss or _sde_loss, which are chosen based on the type of differential equation (self._de_type).

        :param model_prediction:
            Model prediction for the velocity field b and the denoiser eta.
        :type model_prediction: tuple[torch.Tensor, torch.Tensor]
        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor
        :param n_atoms:
            Number of atoms in each crystal from batch.
        :type n_atoms: torch.Tensor
        :param z:
            Random variable.
        :type z: torch.Tensor

        :return:
            Loss.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def _ode_loss(self, model_prediction: tuple[torch.Tensor, torch.Tensor], t: torch.Tensor, x_0: torch.Tensor,
                  x_1: torch.Tensor, n_atoms: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the ODE stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
        p_1 at times t based on the model prediction for the velocity fields b and the denoisers eta.

        :param model_prediction:
            Model prediction for the velocity field b and the denoiser eta.
        :type model_prediction: tuple[torch.Tensor, torch.Tensor]
        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor
        :param n_atoms:
            Number of atoms in each crystal from batch.
        :type n_atoms: torch.Tensor
        :param z:
            Random variable.
        :type z: torch.Tensor

        :return:
            Loss.
        :rtype: torch.Tensor
        """
        expected_velocity, _ = self._interpolate_derivative(t, x_0, x_1, n_atoms)
        loss = nn.functional.mse_loss(expected_velocity, model_prediction[0])
        return loss

    def _sde_loss(self, model_prediction: tuple[torch.Tensor, torch.Tensor], t: torch.Tensor, x_0: torch.Tensor,
                  x_1: torch.Tensor, n_atoms: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the SDE stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
        p_1 at times t based on the model prediction for the velocity fields b and the denoisers eta.

        :param model_prediction:
            Model prediction for the velocity field b and the denoiser eta.
        :type model_prediction: tuple[torch.Tensor, torch.Tensor]
        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor
        :param n_atoms:
            Number of atoms in each crystal from batch.
        :type n_atoms: torch.Tensor
        :param z:
            Random variable.
        :type z: torch.Tensor

        :return:
            Loss.
        :rtype: torch.Tensor
        """
        expected_velocity, expected_z = self._interpolate_derivative(t, x_0, x_1, n_atoms)
        loss_b = nn.functional.mse_loss(expected_velocity, model_prediction[0])
        loss_z = nn.functional.mse_loss(expected_z, model_prediction[1])
        return loss_b + loss_z

    def integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  x_t: torch.Tensor, tspan: tuple[float, float]) -> torch.Tensor:
        """
        Integrate the current positions x_t from time tspan[0] to tspan[1] based on the velocity fields b and the
        denoisers eta returned by the model function.

        This method is only defined here to define all methods of the abstract base class. The actual loss method is
        either _ode_integrate or _sde_integrate, which are chosen based on the type of differential equation (self._de_type).

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta given the current times t and positions
            x_t.
        :type model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        :param x_t:
            Current positions.
        :type x_t: torch.Tensor
        :param tspan:
            Time span for integration.
        :type tspan: tuple[float, float]

        :return:
            Integrated position.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def _ode_integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                       x_t: torch.Tensor, tspan: tuple[float, float]) -> torch.Tensor:
        """
        Integrate the ODE for the current positions x_t from time tspan[0] to tspan[1] based on the velocity fields b
        and the denoisers eta returned by the model function.

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta given the current times t and positions
            x_t.
        :type model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        :param x_t:
            Current positions.
        :type x_t: torch.Tensor
        :param tspan:
            Time span for integration.
        :type tspan: tuple[float, float]

        :return:
            Integrated position.
        :rtype: torch.Tensor
        """
        # Modify wrapper to only use b(t,x).
        ode_wrapper = lambda t, x: self._corrector.correct(model_function(t, x)[0])

        # Integrate with scipy IVP integrator
        x_t_new = solve_ivp(ode_wrapper, tspan, x_t)

        return torch.tensor(x_t_new[:, -1])

    def _sde_integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                       x_t: torch.Tensor, tspan: tuple[float, float]) -> torch.Tensor:
        """
        Integrate the SDE for the current positions x_t from time tspan[0] to tspan[1] based on the velocity fields b
        and the denoisers eta returned by the model function.

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta given the current times t and positions
            x_t.
        :type model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        :param x_t:
            Current positions.
        :type x_t: torch.Tensor
        :param tspan:
            Time span for integration.
        :type tspan: tuple[float, float]

        :return:
            Integrated position.
        :rtype: torch.Tensor
        """
        # Modify wrapper for use in SDE integrator
        def f(x, t):
            preds = model_function(t, self._corrector.correct(x))  # Because of the noise, the x should be corrected.
            out = preds[0] - (self._epsilon.epsilon(t) / self._gamma.gamma(t)) * preds[1]
            return self._corrector.correct(out)

        def g(x, t):
            out = np.sqrt(2 * self._epsilon.epsilon(t)) * np.eye(x.shape[-1])
            return out

        # SDE Integrator
        x_t_new = sdeint.itoint(f, g, x_t, torch.linspace(tspan[0], tspan[1], self._sde_number_time_steps))

        # Return
        return torch.tensor(x_t_new[:, -1])
