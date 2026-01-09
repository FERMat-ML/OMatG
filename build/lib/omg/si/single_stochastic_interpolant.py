from enum import Enum, auto
from typing import Any, Dict, Iterable, Optional, Callable
import torch
from torch_scatter import scatter_mean
from torchdiffeq import odeint
from torchsde import sdeint
from .abstracts import Corrector, Epsilon, Interpolant, LatentGamma, StochasticInterpolant
from .interpolants import ScoreBasedDiffusionModelInterpolantVP, ScoreBasedDiffusionModelInterpolantVE


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
    :param integrator_kwargs: Optional keyword arguments for the odeint function of torchdiffeq (see
        https://github.com/rtqichen/torchdiffeq/blob/master/README.md) or the sdeint function of torchsde (see
        https://github.com/google-research/torchsde/blob/master/DOCUMENTATION.md#keyword-arguments-of-sdeint).
    :type integrator_kwargs: Optional[dict]
    :param correct_center_of_mass_motion:
        TODO: Do we also want to do that during integration?
        Whether to correct the center-of-mass motion to zero before computing the loss.
        This might be useful because the translational invariant model cannot predict the center-of-mass motion.
        This is the approach chosen by FlowMM.
        Defaults to False.
    :type correct_center_of_mass_motion: bool
    :param velocity_annealing_factor:
        During inference, the predicted velocity fields b at time are multiplied by (1 + velocity_annealing_factor * t).
        A velocity annealing factor of 0.0 corresponds to no annealing.
        Defaults to 0.0.
    :type velocity_annealing_factor: float

    :raises ValueError:
        If epsilon is provided for ODEs or not provided for SDEs.
    """

    def __init__(self, interpolant: Interpolant, gamma: Optional[LatentGamma], epsilon: Optional[Epsilon],
                 differential_equation_type: str, integrator_kwargs: Optional[dict[str, Any]] = None,
                 correct_center_of_mass_motion: bool = False, velocity_annealing_factor: float = 0.0) -> None:
        """Construct stochastic interpolant."""
        super().__init__()
        self._interpolant = interpolant
        self._gamma = gamma
        if self._gamma is not None:
            self._use_antithetic = self._gamma.requires_antithetic()
        else:
            self._use_antithetic = False
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
            if self._gamma is None:
                raise ValueError("Gamma function should be provided for SDEs.")
        self._integrator_kwargs = integrator_kwargs if integrator_kwargs is not None else {}
        self._correct_center_of_mass_motion = correct_center_of_mass_motion
        self._velocity_annealing_factor = velocity_annealing_factor
        # This also disables PeriodicScoreBasedDiffusionModelInterpolantVP and
        # PeriodicScoreBasedDiffusionModelInterpolantVE.
        if isinstance(self._interpolant,
                      (ScoreBasedDiffusionModelInterpolantVP, ScoreBasedDiffusionModelInterpolantVE)):
            raise ValueError("The interpolant should not be a ScoreBasedDiffusionModelInterpolantVP or "
                             "ScoreBasedDiffusionModelInterpolantVE because it requires antithetic sampling on the "
                             "level of alpha(t).")

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
        # Output is already corrected.
        interpolate = self._interpolant.interpolate(t, x_0, x_1)
        if self._gamma is not None:
            z = torch.randn_like(x_0)
            interpolate = self._corrector.correct(interpolate + self._gamma.gamma(t) * z)
        else:
            z = torch.zeros_like(x_0)
        return interpolate, z

    def _interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                                z: torch.Tensor) -> torch.Tensor:
        """
        Derivative with respect to time of the stochastic interpolant between points x_0 and x_1 from two distributions
        p_0 and p_1 at times t.

        TODO: Remove this unused method.

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
        if self._gamma is not None:
            interpolate_derivative += self._gamma.gamma_derivative(t) * z
        return interpolate_derivative

    def loss_keys(self) -> Iterable[str]:
        """
        Get the keys of the losses returned by the loss function.

        :return:
            Keys of the losses.
        :rtype: Iterable[str]
        """
        if self._differential_equation_type == DifferentialEquationType.ODE:
            yield "loss_b"
        else:
            assert self._differential_equation_type == DifferentialEquationType.SDE
            yield "loss_b"
            yield "loss_z"

    def loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
             t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
             batch_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the losses for the stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
        p_1 at times t based on the model prediction for the velocity fields b and the denoisers eta.

        The loss of the velocity fields is returned with the key 'loss_b'. For the SDE case, an additional loss for the
        denoisers is returned with the keys 'loss_z'.

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
            Losses.
        :rtype: Dict[str, torch.Tensor]
        """
        raise NotImplementedError

    def _ode_loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
                  batch_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the losses for the ODE stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
        p_1 at times t based on the model prediction for the velocity fields b and the denoisers eta.

        The loss of the velocity fields is returned with the key 'loss_b'.

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
            Losses.
        :rtype: Dict[str, torch.Tensor]
        """
        assert x_0.shape == x_1.shape
        if self._use_antithetic:
            assert self._gamma is not None
            x_t_without_gamma = self._interpolant.interpolate(t, x_0, x_1)
            gamma = self._gamma.gamma(t)
            x_t_p = self._corrector.correct(x_t_without_gamma + gamma * z)
            assert torch.equal(x_t, x_t_p)
            x_t_m = self._corrector.correct(x_t_without_gamma - gamma * z)
            expected_velocity_without_gamma = self._interpolant.interpolate_derivative(t, x_0, x_1)
            gamma_derivative = self._gamma.gamma_derivative(t)
            expected_velocity_p = expected_velocity_without_gamma + gamma_derivative * z
            expected_velocity_m = expected_velocity_without_gamma - gamma_derivative * z
            if self._correct_center_of_mass_motion:
                # scatter_mean is used to compute the mean velocity for every configuration.
                # index_select is used to replicate the mean velocity for every atom in the configuration.
                # We don't need to worry about periodic boundary conditions in the corrector here, since the tangent
                # space is Euclidean.
                # After this correction, it is true that
                # expected_velocity_p = corr(expected_velocity_without_gamma) + corr(gamma_derivative * z),
                # expected_velocity_m = corr(expected_velocity_without_gamma) - corr(gamma_derivative * z),
                # where corr is the correction to the center of mass motion.
                mean_velocity_p = torch.index_select(scatter_mean(expected_velocity_p, batch_indices, dim=0),
                                                     0, batch_indices)
                expected_velocity_p = expected_velocity_p - mean_velocity_p
                mean_velocity_m = torch.index_select(scatter_mean(expected_velocity_m, batch_indices, dim=0),
                                                     0, batch_indices)
                expected_velocity_m = expected_velocity_m - mean_velocity_m
            pred_b_p = model_function(x_t_p)[0]
            pred_b_m = model_function(x_t_m)[0]
            loss = (0.5 * torch.mean(pred_b_p ** 2) + 0.5 * torch.mean(pred_b_m ** 2)
                    - torch.mean(pred_b_p * expected_velocity_p) - torch.mean(pred_b_m * expected_velocity_m))
        else:
            expected_velocity = self._interpolant.interpolate_derivative(t, x_0, x_1)
            if self._gamma is not None:
                expected_velocity += self._gamma.gamma_derivative(t) * z
            pred_b = model_function(x_t)[0]
            if self._correct_center_of_mass_motion:
                # scatter_mean is used to compute the mean velocity for every configuration.
                # index_select is used to replicate the mean velocity for every atom in the configuration.
                mean_velocity = torch.index_select(scatter_mean(expected_velocity, batch_indices, dim=0),
                                                   0, batch_indices)
                expected_velocity = expected_velocity - mean_velocity
            loss = (torch.mean(pred_b ** 2) - 2.0 * torch.mean(pred_b * expected_velocity))
        return {"loss_b": loss}

    def _sde_loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
                  batch_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the losses for the SDE stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
        p_1 at times t based on the model prediction for the velocity fields b and the denoisers eta.

       The loss of the velocity fields is returned with the key 'loss_b'. The loss of the denoisers eta is returned with
       the key 'loss_z'.

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
            Losses.
        :rtype: Dict[str, torch.Tensor]
        """
        assert x_0.shape == x_1.shape
        assert self._gamma is not None
        if self._use_antithetic:
            x_t_without_gamma = self._interpolant.interpolate(t, x_0, x_1)
            gamma = self._gamma.gamma(t)
            x_t_p = self._corrector.correct(x_t_without_gamma + gamma * z)
            assert torch.equal(x_t, x_t_p)
            x_t_m = self._corrector.correct(x_t_without_gamma - gamma * z)
            expected_velocity_without_gamma = self._interpolant.interpolate_derivative(t, x_0, x_1)
            gamma_derivative = self._gamma.gamma_derivative(t)
            expected_velocity_p = expected_velocity_without_gamma + gamma_derivative * z
            expected_velocity_m = expected_velocity_without_gamma - gamma_derivative * z
            if self._correct_center_of_mass_motion:
                # scatter_mean is used to compute the mean velocity for every configuration.
                # index_select is used to replicate the mean velocity for every atom in the configuration.
                # We don't need to worry about periodic boundary conditions in the corrector here, since the tangent
                # space is Euclidean.
                # After this correction, it is true that
                # expected_velocity_p = corr(expected_velocity_without_gamma) + corr(gamma_derivative * z),
                # expected_velocity_m = corr(expected_velocity_without_gamma) - corr(gamma_derivative * z),
                # where corr is the correction to the center of mass motion.
                mean_velocity_p = torch.index_select(scatter_mean(expected_velocity_p, batch_indices, dim=0),
                                                     0, batch_indices)
                expected_velocity_p = expected_velocity_p - mean_velocity_p
                mean_velocity_m = torch.index_select(scatter_mean(expected_velocity_m, batch_indices, dim=0),
                                                     0, batch_indices)
                expected_velocity_m = expected_velocity_m - mean_velocity_m
            pred_b_p, pred_z = model_function(x_t_p)
            pred_b_m, _ = model_function(x_t_m)

            loss_b = (0.5 * torch.mean(pred_b_p ** 2) + 0.5 * torch.mean(pred_b_m ** 2)
                      - torch.mean(pred_b_p * expected_velocity_p) - torch.mean(pred_b_m * expected_velocity_m))
        else:
            expected_velocity = (self._interpolant.interpolate_derivative(t, x_0, x_1)
                                 + self._gamma.gamma_derivative(t) * z)
            pred_b, pred_z = model_function(x_t)
            if self._correct_center_of_mass_motion:
                # scatter_mean is used to compute the mean velocity for every configuration.
                # index_select is used to replicate the mean velocity for every atom in the configuration.
                mean_velocity = torch.index_select(scatter_mean(expected_velocity, batch_indices, dim=0),
                                                   0, batch_indices)
                expected_velocity = expected_velocity - mean_velocity
            loss_b = torch.mean(pred_b ** 2) - torch.mean(pred_b * expected_velocity)

        loss_z = torch.mean(pred_z ** 2) - 2.0 * torch.mean(pred_z * z)

        return {"loss_b": loss_b, "loss_z": loss_z}

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
        # Set up ODE function
        odefunc = lambda t, x: ((1.0 + self._velocity_annealing_factor * t)
                                * model_function(t, self._corrector.correct(x))[0])
        t_span = torch.tensor([time, time + time_step], device=x_t.device)
        with torch.no_grad():
            x_t_new = odeint(odefunc, x_t, t_span, **self._integrator_kwargs)[-1]
        return self._corrector.correct(x_t_new)

    # Modify wrapper for use in SDE integrator
    class SDE(torch.nn.Module):
        def __init__(self, model_func, corrector, gamma, epsilon, original_x_shape, velocity_annealing_factor):
            super().__init__()
            self._model_func = model_func
            self._corrector = corrector
            self._gamma = gamma
            self._epsilon = epsilon
            self._original_x_shape = original_x_shape
            self._velocity_annealing_factor = velocity_annealing_factor
            # Required for torchsde.
            self.sde_type = "ito"
            self.noise_type = "diagonal"

        def f(self, t, x):
            # Because of the noise, the x should be corrected when it is passed to the model.
            new_x_shape = x.shape
            preds = self._model_func(t, self._corrector.correct(x.reshape(self._original_x_shape)))
            out = ((1.0 + self._velocity_annealing_factor * t) * preds[0]
                   - (self._epsilon.epsilon(t) / self._gamma.gamma(t)) * preds[1])
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
        # SDE Integrator
        original_shape = x_t.shape
        sde = self.SDE(model_func=model_function, corrector=self._corrector, gamma=self._gamma, epsilon=self._epsilon,
                       original_x_shape=original_shape, velocity_annealing_factor=self._velocity_annealing_factor)
        t_span = torch.tensor([time, time + time_step], device=x_t.device)

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
