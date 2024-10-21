from enum import Enum, auto
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import torchsde
from typing import Any, Optional, Callable
from .abstracts import Corrector, Epsilon, Interpolant, LatentGamma, StochasticInterpolant


# Modify wrapper for use in SDE integrator
class SDE(torch.nn.Module):
    def __init__(self, model_func):
        super().__init__()
        self.model_func = model_func

    def f(self, t, x):
        preds = self.model_func(t, self._corrector.correct(x))  # Because of the noise, the x should be corrected.
        out = preds[0] - (self._epsilon.epsilon(t) / self._gamma.gamma(t)) * preds[1]
        return self._corrector.correct(out)

    def g(self, t, x):
        out = torch.sqrt(2 * self._epsilon.epsilon(t)) * np.eye(x.shape[-1])
        return out


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

    The ODE is integrated using the torchdiffeq library and the SDE is integrated using the torchsde library.

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
    :param integrator_kwargs: Optional keyword arguments for the odeint function of torchdiffeq (see
        https://github.com/rtqichen/torchdiffeq/blob/master/README.md) or the sdeint function of torchsde (see
        https://github.com/google-research/torchsde/blob/master/DOCUMENTATION.md#keyword-arguments-of-sdeint).
    :type integrator_kwargs: Optional[dict]

    :raises ValueError:
        If epsilon is provided for ODEs or not provided for SDEs.
        If sde_number_time_steps is provided for ODEs or not provided for SDEs.
        If sde_number_time_steps is not positive.
    """

    def __init__(self, interpolant: Interpolant, gamma: Optional[LatentGamma], epsilon: Optional[Epsilon],
                 differential_equation_type: str, sde_number_time_steps: Optional[int] = None,
                 corrector: Optional[Corrector] = None, integrator_kwargs: Optional[dict[str, Any]] = None) -> None:
        """Construct stochastic interpolant."""
        super().__init__()
        self._interpolant = interpolant
        self._gamma = gamma
        if self._gamma is not None:
            self._use_antithetic = True
        else:
            self._use_antithetic = False
        self._epsilon = epsilon
        self._differential_equation_type = differential_equation_type
        self._sde_number_time_steps = sde_number_time_steps
        self._corrector = corrector if corrector is not None else self.IdentityCorrector()
        try:
            self._differential_equation_type = DifferentialEquationType[differential_equation_type]
        except AttributeError:
            raise ValueError(f"Unknown differential equation type f{differential_equation_type}.")
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
        self._integrator_kwargs = integrator_kwargs if integrator_kwargs is not None else {}

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

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                    batch_pointer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Stochastically interpolated points x_t, random variables z used for interpolation.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        assert x_0.shape == x_1.shape
        interpolate = self._interpolant.interpolate(t, x_0, x_1, batch_pointer)
        if self._gamma is not None:
            z = torch.randn_like(x_0)
            interpolate += self._gamma.gamma(t) * z
        else:
            z = torch.zeros_like(x_0)
        return interpolate, z

    def _interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, z: torch.Tensor,
                                batch_pointer: torch.Tensor) -> torch.Tensor:
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
            Random variable z that was used for the stochastic interpolation to get the model prediction.
        :type z: torch.Tensor
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Stochastically interpolated value.
        :rtype: torch.Tensor
        """
        assert x_0.shape == x_1.shape
        self._check_t(t)
        interpolate_derivative = self._interpolant.interpolate_derivative(t, x_0, x_1, batch_pointer)
        if self._gamma is not None:
            interpolate_derivative += self._gamma.gamma_derivative(t) * z
        return interpolate_derivative

    def loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
             t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
             batch_pointer: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
        p_1 at times t based on the model prediction for the velocity fields b and the denoisers eta.

        This method is only defined here to define all methods of the abstract base class. The actual loss method is
        either _ode_loss or _sde_loss, which are chosen based on the type of differential equation (self._de_type).

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta given the current positions x_t.
        :type model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor
        :param x_t:
            Stochastically interpolated points x_t.
        :type x_t: torch.Tensor
        :param z:
            Random variable z that was used for the stochastic interpolation to get the model prediction.
        :type z: torch.Tensor
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Loss.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def _ode_loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
                  batch_pointer: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the ODE stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
        p_1 at times t based on the model prediction for the velocity fields b and the denoisers eta.

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta given the current positions x_t.
        :type model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor
        :param x_t:
            Stochastically interpolated points x_t.
        :type x_t: torch.Tensor
        :param z:
            Random variable z that was used for the stochastic interpolation to get the model prediction.
        :type z: torch.Tensor
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Loss.
        :rtype: torch.Tensor
        """
        assert x_0.shape == x_1.shape
        x_t_without_gamma = self._interpolant.interpolate(t, x_0, x_1, batch_pointer)
        expected_velocity_without_gamma = self._interpolant.interpolate_derivative(t, x_0, x_1, batch_pointer)
        if self._use_antithetic:
            assert self._gamma is not None
            x_t_p = x_t_without_gamma + self._gamma.gamma(t) * z
            assert torch.equal(x_t, x_t_p)
            x_t_m = x_t_without_gamma - self._gamma.gamma(t) * z
            expected_velocity_p = expected_velocity_without_gamma + self._gamma.gamma_derivative(t) * z
            expected_velocity_m = expected_velocity_without_gamma - self._gamma.gamma_derivative(t) * z
            loss = (nn.functional.mse_loss(expected_velocity_p, model_function(x_t_p)[0])
                    + nn.functional.mse_loss(expected_velocity_m, model_function(x_t_m)[0])) / 2.0
        else:
            assert self._gamma is None
            assert torch.equal(x_t, x_t_without_gamma)
            loss = nn.functional.mse_loss(expected_velocity_without_gamma, model_function(x_t_without_gamma)[0])
        return loss

    def _sde_loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
                  batch_pointer: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the SDE stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
        p_1 at times t based on the model prediction for the velocity fields b and the denoisers eta.

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta given the current positions x_t.
        :type model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor
        :param x_t:
            Stochastically interpolated points x_t.
        :type x_t: torch.Tensor
        :param z:
            Random variable z that was used for the stochastic interpolation to get the model prediction.
        :type z: torch.Tensor
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Loss.
        :rtype: torch.Tensor
        """
        assert x_0.shape == x_1.shape
        x_t_without_gamma = self._interpolant.interpolate(t, x_0, x_1, batch_pointer)
        expected_velocity_without_gamma = self._interpolant.interpolate_derivative(t, x_0, x_1, batch_pointer)
        if self._use_antithetic:
            assert self._gamma is not None
            x_t_p = x_t_without_gamma + self._gamma.gamma(t) * z
            assert torch.equal(x_t, x_t_p)
            x_t_m = x_t_without_gamma - self._gamma.gamma(t) * z
            expected_velocity_p = expected_velocity_without_gamma + self._gamma.gamma_derivative(t) * z
            expected_velocity_m = expected_velocity_without_gamma - self._gamma.gamma_derivative(t) * z
            pred_b_p, pred_z = model_function(x_t_p)
            loss_b = (nn.functional.mse_loss(expected_velocity_p, pred_b_p)
                      + nn.functional.mse_loss(expected_velocity_m, model_function(x_t_m)[0])) / 2.0
        else:
            assert self._gamma is None
            assert torch.equal(x_t, x_t_without_gamma)
            pred_b, pred_z = model_function(x_t_without_gamma)
            loss_b = nn.functional.mse_loss(expected_velocity_without_gamma, pred_b)

        loss_z = nn.functional.mse_loss(z, pred_z)
        return loss_b + loss_z

    def integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  x_t: torch.Tensor, time: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        """
        Integrate the current positions x_t at the given time for the given time step based on the velocity fields b and
        the denoisers eta returned by the model function.

        This method is only defined here to define all methods of the abstract base class. The actual loss method is
        either _ode_integrate or _sde_integrate, which are chosen based on the type of differential equation (self._de_type).

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta given the current times t and positions
            x_t.
        :type model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        :param x_t:
            Current positions.
        :type x_t: torch.Tensor
        :param time:
            Initial time (0-dimensional torch tensor).
        :type time: torch.Tensor
        :param time_step:
            Time step (0-dimensional torch tensor).
        :type time_step: torch.Tensor

        :return:
            Integrated position.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def _ode_integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                       x_t: torch.Tensor, time: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        """
        Integrate the ODE for the current positions x_t at the given time for the given time step based on the velocity
        fields b and the denoisers eta returned by the model function.

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta given the current times t and positions
            x_t.
        :type model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        :param x_t:
            Current positions.
        :type x_t: torch.Tensor
        :param time:
            Initial time (0-dimensional torch tensor).
        :type time: torch.Tensor
        :param time_step:
            Time step (0-dimensional torch tensor).
        :type time_step: torch.Tensor

        :return:
            Integrated position.
        :rtype: torch.Tensor
        """

        # Set up ODE function
        odefunc = lambda t, x: model_function(t, self._corrector.correct(x))[0]

        # Integrate with scipy IVP integrator
        original_shape = x_t.shape
        t_span = torch.tensor([time, time + time_step])
        x_t = torch.reshape(x_t, (-1,))
        x_t_new = odeint(odefunc, x_t, t_span, **self._integrator_kwargs)
        x_t_new = torch.tensor(x_t_new.y[:, -1].reshape(original_shape))

        # Applies corrector to output of integration not the b field itself
        # Can consider only applying corrector after final integration step but useful here for debugging/testing purposes
        return self._corrector.correct(x_t_new)

    def _sde_integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                       x_t: torch.Tensor, time: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        """
        Integrate the SDE for the current positions x_t at the given time for the given time step based on the velocity
        fields b and the denoisers eta returned by the model function.

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta given the current times t and positions
            x_t.
        :type model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        :param x_t:
            Current positions.
        :type x_t: torch.Tensor
        :param time:
            Initial time (0-dimensional torch tensor).
        :type time: torch.Tensor
        :param time_step:
            Time step (0-dimensional torch tensor).
        :type time_step: torch.Tensor

        :return:
            Integrated position.
        :rtype: torch.Tensor
        """

        # SDE Integrator
        sde = SDE(model_func=model_function)
        t_span = torch.tensor([time, time + time_step])
        x_t_new = torchsde.sdeint(sde, x_t, t_span, **self._integrator_kwargs)

        # Return
        return torch.tensor(x_t_new[:, -1])
