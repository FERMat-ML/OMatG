from enum import Enum, auto
import torch
from torch_scatter import scatter_mean
from torchdiffeq import odeint
from torchsde import sdeint
from typing import Any, Optional, Callable
from .abstracts import Corrector, Epsilon, Interpolant, LatentGamma, StochasticInterpolant
from .single_stochastic_interpolant import DifferentialEquationType

class SingleStochasticInterpolantOS(StochasticInterpolant):
    """
    Stochastic interpolant (One sided) x_t = I(t, x_0, x_1) + gamma(t) * z between points x_0 and x_1 from two distributions p_0 and
    p_1 at times t based on an interpolant I(t, x_0, x_1), a gamma function gamma(t), and a Gaussian random variable z.

    No Gamma function accepted for this particular type of interpolant

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
    :param integrator_kwargs: Optional keyword arguments for the odeint function of torchdiffeq (see
        https://github.com/rtqichen/torchdiffeq/blob/master/README.md) or the sdeint function of torchsde (see
        https://github.com/google-research/torchsde/blob/master/DOCUMENTATION.md#keyword-arguments-of-sdeint).
    :type integrator_kwargs: Optional[dict]
    :param correct_center_of_mass:
        TODO: Do we also want to do that during integration?
        Whether to correct the center of mass of the points x_0 and x_1 to zero before computing the loss.
        Defaults to False.
    :type correct_center_of_mass: bool
    :param correct_center_of_mass_motion:
        TODO: Do we also want to do that during integration?
        Whether to correct the center-of-mass motion to zero before computing the loss.
        This might be useful because the translational invariant model cannot predict the center-of-mass motion.
        This is the approach chosen by FlowMM.
        Defaults to False.
    :type correct_center_of_mass_motion: bool
    :param correct_first_atom:
        TODO: Do we want to correct x_t_p and x_t_m in the antithetic case?
        TODO: Do we also want to do that during integration?
        Whether to correct the positions of the first atoms of the points x_0 and x_1 to zero before computing the loss.
        Defaults to False.
    :raises ValueError:
        If epsilon is provided for ODEs or not provided for SDEs.
    """

    def __init__(self, interpolant: Interpolant, gamma: Optional[LatentGamma], epsilon: Optional[Epsilon],
                 differential_equation_type: str, integrator_kwargs: Optional[dict[str, Any]] = None,
                 correct_center_of_mass: bool = False, correct_center_of_mass_motion: bool = False,
                 correct_first_atom: bool = False, velocity_annealing_factor: float = 0.0) -> None:
        """Construct stochastic interpolant."""
        super().__init__()
        self._interpolant = interpolant
        self._gamma = gamma
        assert self._gamma is None
        self._epsilon = epsilon
        self._differential_equation_type = differential_equation_type
        # Corrector that needs to be applied to the points x_t during integration.
        self._corrector = self._interpolant.get_corrector()
        try:
            self._differential_equation_type = DifferentialEquationType[differential_equation_type]
        except AttributeError:
            raise ValueError(f"Unknown differential equation type f{differential_equation_type}.")
        if self._differential_equation_type == DifferentialEquationType.ODE:
            self.loss = self._ode_loss
            self.integrate = self._ode_integrate
            if self._epsilon is not None:
                raise ValueError("Epsilon function should not be provided for ODEs.")

        else:
            assert self._differential_equation_type == DifferentialEquationType.SDE
            self.loss = self._sde_loss
            self.integrate = self._sde_integrate
            if self._epsilon is None:
                raise ValueError("Epsilon function should be provided for SDEs.")
        self._integrator_kwargs = integrator_kwargs if integrator_kwargs is not None else {}
        self._correct_center_of_mass = correct_center_of_mass
        self._correct_center_of_mass_motion = correct_center_of_mass_motion
        self._correct_first_atom = correct_first_atom
        if self._correct_center_of_mass and self._correct_first_atom:
            raise ValueError("Correcting the center of mass and the first atom at the same time is not possible.")
        self._velocity_annealing_factor = velocity_annealing_factor

    @staticmethod
    def _shift_first_atom(x: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        unique_indices = torch.unique(batch_indices)
        first_occurrences = torch.tensor([torch.nonzero(batch_indices == i, as_tuple=True)[0][0]
                                          for i in unique_indices], device=x.device)
        first_atoms = torch.index_select(x[first_occurrences], 0, batch_indices)
        return x - first_atoms

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                    batch_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.
        :type batch_indices: torch.Tensor

        :return:
            Stochastically interpolated points x_t, random variables z used for interpolation.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        assert x_0.shape == x_1.shape
        interpolate = self._interpolant.interpolate(t, x_0, x_1)
        z = x_0
        interpolate = self._corrector.correct(interpolate)
        return interpolate, z

    def _interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                                z: torch.Tensor) -> torch.Tensor:
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

        :return:
            Stochastically interpolated value.
        :rtype: torch.Tensor
        """
        assert x_0.shape == x_1.shape
        self._check_t(t)
        interpolate_derivative = self._interpolant.interpolate_derivative(t, x_0, x_1)
        return interpolate_derivative

    def loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
             t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
             batch_indices: torch.Tensor) -> torch.Tensor:
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
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.
        :type batch_indices: torch.Tensor

        :return:
            Loss.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def _ode_loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
                  batch_indices: torch.Tensor) -> torch.Tensor:
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
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.
        :type batch_indices: torch.Tensor

        :return:
            Loss.
        :rtype: torch.Tensor
        """
        assert x_0.shape == x_1.shape
        pred_z = model_function(x_t)[1]
        loss = (torch.mean(pred_z ** 2) - 2.0 * torch.mean(pred_z * z))
        return loss

    def _sde_loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
                  batch_indices: torch.Tensor) -> torch.Tensor:
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
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.
        :type batch_indices: torch.Tensor

        :return:
            Loss.
        :rtype: torch.Tensor
        """
        return self._ode_loss(model_function=model_function, t=t, x_0=x_0, x_1=x_1, x_t=x_t, z=z, batch_indices=batch_indices)

    def integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  x_t: torch.Tensor, time: torch.Tensor, time_step: torch.Tensor,
                  batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Integrate the current positions x_t at the given time for the given time step based on the velocity fields b and
        the denoisers eta returned by the model function.

        This method is only defined here to define all methods of the abstract base class. The actual loss method is
        either _ode_integrate or _sde_integrate, which are chosen based on the type of differential equation
        (self._de_type).

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
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.
        :type batch_indices: torch.Tensor

        :return:
            Integrated position.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def _ode_integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                       x_t: torch.Tensor, time: torch.Tensor, time_step: torch.Tensor,
                       batch_indices: torch.Tensor) -> torch.Tensor:
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
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.
        :type batch_indices: torch.Tensor

        :return:
            Integrated position.
        :rtype: torch.Tensor
        """
        # OS ODE function
        def odefunc(t, x):
            x_corr = self._corrector.correct(x)
            z = model_function(t, x_corr)[1]
            t1 = (self._interpolant.alpha_dot(t) * z)
            t2 = (self._interpolant.beta_dot(t) / self._interpolant.beta(t)) * (x_corr - self._interpolant.alpha(t) * z)
            return (1.0 + self._velocity_annealing_factor * t) * (t1 + t2)
        t_span = torch.tensor([time, time + time_step], device=x_t.device)
        with torch.no_grad():
            x_t_new = odeint(odefunc, x_t, t_span, **self._integrator_kwargs)[-1]
        return self._corrector.correct(x_t_new)

    # Modify wrapper for use in SDE integrator
    class SDE(torch.nn.Module):
        def __init__(self, model_func, corrector, interpolant, epsilon, original_x_shape):
            super().__init__()
            self._model_func = model_func
            self._corrector = corrector
            self._interpolant = interpolant
            self._epsilon = epsilon
            self._original_x_shape = original_x_shape
            # Required for torchsde.
            self.sde_type = "ito"
            self.noise_type = "diagonal"

        def f(self, t, x):
            # Because of the noise, the x should be corrected when it is passed to the model.
            new_x_shape = x.shape
            x_corr = self._corrector(x)
            z = self._model_func(t, x_corr.reshape(self._original_x_shape))[1]
            t1 = (self._interpolant.alpha_dot(t) * z)
            t2 = (self._interpolant.beta(t) / self._interpolant.beta_dot(t)) * (x_corr - self._interpolant.alpha(t) * z)
            t3 = (self._epsilon(t) / self._interpolant.alpha(t)) * z
            out = t1 + t2 - t3
            return out.reshape(new_x_shape)

        def g(self, t, x):
            return torch.sqrt(2.0 * self._epsilon.epsilon(t)) * torch.ones_like(x)

    def _sde_integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                       x_t: torch.Tensor, time: torch.Tensor, time_step: torch.Tensor,
                       batch_indices: torch.Tensor) -> torch.Tensor:
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
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.
        :type batch_indices: torch.Tensor

        :return:
            Integrated position.
        :rtype: torch.Tensor
        """
        # TODO: Introduce corrections here.
        raise NotImplementedError
        # SDE Integrator
        original_shape = x_t.shape
        sde = self.SDE(model_func=model_function, corrector=self._corrector, gamma=self._gamma, epsilon=self._epsilon,
                       original_x_shape=original_shape)
        t_span = torch.tensor([time, time + time_step])

        with torch.no_grad():
            # Diagonal noise in torchsde expects a tensor of shape (batch_size, state_size).
            # See https://github.com/google-research/torchsde/blob/master/DOCUMENTATION.md.
            # Since every configuration in the batch can have a different number of atoms, such a reshape is generally
            # not possible. Therefore, we keep the first dimension as the batch size and flatten the rest.
            # This should not matter for the integration, as the noise is diagonal. Every noise term is independent and
            # affects only on state-size dimension.
            x_t_new = sdeint(sde, x_t.reshape((original_shape[0], -1)), t_span, **self._integrator_kwargs)

        return self._corrector.correct(x_t_new[-1].reshape(original_shape))

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the stochastic interpolant (for instance, a corrector that considers periodic
        boundary conditions).

        :return:
           Corrector.
        :rtype: Corrector
        """
        return self._corrector
