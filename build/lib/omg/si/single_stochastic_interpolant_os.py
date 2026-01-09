from typing import Any, Dict, Iterable, Optional, Callable
import torch
from torch_scatter import scatter_mean
from torchdiffeq import odeint
from torchsde import sdeint
from .abstracts import Corrector, Epsilon, Interpolant, StochasticInterpolant
from .interpolants import ScoreBasedDiffusionModelInterpolantVP, ScoreBasedDiffusionModelInterpolantVE
from .single_stochastic_interpolant import DifferentialEquationType


class SingleStochasticInterpolantOS(StochasticInterpolant):
    """
    One-sided stochastic interpolant x_t = alpha(t) * x_0 + beta(t) * x_1 between points x_0 and x_1 from two
    distributions p_0 and p_1 at times t based on interpolating functions alpha(t) and beta(t) with
    alpha(0) = beta(1) = 1 and alpha(1) = beta(0) = 0. In this class, x_0 is assumed to be a Gaussian random variable in
    which case the effect of the latent variable z in the SingleStochasticInterpolant class can be merged with x_0.

    The stochastic interpolant can either use an ordinary differential equation (ODE) or a stochastic differential
    equation during inference. If an SDE is used, one should additionally provide an epsilon function epsilon(t).

    The ODE is integrated using the torchdiffeq library and the SDE is integrated using the torchsde library.

    In principle, one only needs to learn the denoiser for one-sided interpolants because the velocity field can derived
    from it (see Eq. (6.7) in the SI paper). However, the resulting velocity field is singular (though with a finite
    mean) which is why it might still be useful to learn the velocity field.

    :param interpolant:
        Interpolant I(t, x_0, x_1) = alpha(t) * x_0 + beta(t) * x_1 between points from two distributions p_0 and p_1 at
        times t.
    :type interpolant: Interpolant
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
    :param predict_velocity:
        Whether to compute the loss for the velocity field.
        Defaults to True.
    :type predict_velocity: bool
    :param velocity_annealing_factor:
        During inference, the predicted velocity fields b at time are multiplied by (1 + velocity_annealing_factor * t).
        A velocity annealing factor of 0.0 corresponds to no annealing.
        Should only be set if predict_velocity is True.
        Defaults to None.
    :type velocity_annealing_factor: Optional[float]

    :raises ValueError:
        If epsilon is provided for ODEs or not provided for SDEs.
    """

    def __init__(self, interpolant: Interpolant, epsilon: Optional[Epsilon], differential_equation_type: str,
                 integrator_kwargs: Optional[dict[str, Any]] = None, correct_center_of_mass_motion: bool = False,
                 predict_velocity: bool = True, velocity_annealing_factor: Optional[float] = None) -> None:
        """Construct stochastic interpolant."""
        super().__init__()
        self._interpolant = interpolant
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
        self._correct_center_of_mass_motion = correct_center_of_mass_motion
        self._predict_velocity = predict_velocity
        # This is also true for the PeriodicScoreBasedDiffusionModelInterpolantVP and
        # PeriodicScoreBasedDiffusionModelInterpolantVE.
        self._use_antithetic = isinstance(self._interpolant,
                                          (ScoreBasedDiffusionModelInterpolantVP,
                                           ScoreBasedDiffusionModelInterpolantVE))
        self._velocity_annealing_factor = velocity_annealing_factor
        if not self._predict_velocity and self._velocity_annealing_factor is not None:
            raise ValueError("Velocity annealing factor should only be set if predict_velocity is True.")
        if self._predict_velocity and self._velocity_annealing_factor is None:
            self._velocity_annealing_factor = 0.0

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
        # x_0 takes the role of the random variable z.
        return interpolate, x_0.clone()

    def loss_keys(self) -> Iterable[str]:
        """
        Get the keys of the losses returned by the loss function.

        :return:
            Keys of the losses.
        :rtype: Iterable[str]
        """
        if self._predict_velocity:
            yield "loss_b"
            if self._differential_equation_type == DifferentialEquationType.SDE:
                yield "loss_z"
        else:
            yield "loss_z"

    def loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
             t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
             batch_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the losses for the stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
        p_1 at times t based on the model prediction for the velocity fields b and the denoisers eta.

        If predict_velocity was set to True on initialization, the loss of the velocity fields is returned with the key
        'loss_b'. For the SDE case, an additional loss for the denoisers is returned with the keys 'loss_z'. If
        predict_velocity was set to False on initialization, only the loss of the denoiser is returned with the key
        'loss_z'.

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

        If predict_velocity was set to True on initialization, only the loss of the velocity fields is returned with the
        key 'loss_b'. If predict_velocity was set to False on initialization, only the loss of the denoiser is returned
        with the key 'loss_z'.

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
        assert torch.equal(x_0, z)
        if self._predict_velocity:
            if self._use_antithetic:
                x_t_p = self._interpolant.interpolate(t, x_0, x_1)
                assert torch.equal(x_t, x_t_p)
                x_t_m = self._interpolant.interpolate(t, -x_0, x_1)
                expected_velocity_p = self._interpolant.interpolate_derivative(t, x_0, x_1)
                expected_velocity_m = self._interpolant.interpolate_derivative(t, -x_0, x_1)
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
                return {"loss_b": (0.5 * torch.mean(pred_b_p ** 2) + 0.5 * torch.mean(pred_b_m ** 2)
                                   - torch.mean(pred_b_p * expected_velocity_p)
                                   - torch.mean(pred_b_m * expected_velocity_m))}
            else:
                expected_velocity = self._interpolant.interpolate_derivative(t, x_0, x_1)
                pred_b = model_function(x_t)[0]
                if self._correct_center_of_mass_motion:
                    # scatter_mean is used to compute the mean velocity for every configuration.
                    # index_select is used to replicate the mean velocity for every atom in the configuration.
                    mean_velocity = torch.index_select(scatter_mean(expected_velocity, batch_indices, dim=0),
                                                       0, batch_indices)
                    expected_velocity = expected_velocity - mean_velocity
                return {"loss_b": torch.mean(pred_b ** 2) - 2.0 * torch.mean(pred_b * expected_velocity)}
        else:
            assert torch.equal(x_0, z)
            pred_z = model_function(x_t)[1]
            return {"loss_z": torch.mean(pred_z ** 2) - 2.0 * torch.mean(pred_z * z)}

    def _sde_loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
                  batch_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the losses for the SDE stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
        p_1 at times t based on the model prediction for the velocity fields b and the denoisers eta.

        If predict_velocity was set to True on initialization, the loss of the velocity fields is returned with the
        key 'loss_b'. The loss of the denoiser is always returned with the key 'loss_z'.

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
        assert torch.equal(x_0, z)
        pred_b, pred_z = model_function(x_t)
        loss_z = (torch.mean(pred_z ** 2) - 2.0 * torch.mean(pred_z * z))

        if self._predict_velocity:
            if self._use_antithetic:
                x_t_p = self._interpolant.interpolate(t, x_0, x_1)
                assert torch.equal(x_t, x_t_p)
                x_t_m = self._interpolant.interpolate(t, -x_0, x_1)
                expected_velocity_p = self._interpolant.interpolate_derivative(t, x_0, x_1)
                expected_velocity_m = self._interpolant.interpolate_derivative(t, -x_0, x_1)
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
                pred_b_p = pred_b
                pred_b_m = model_function(x_t_m)[0]
                return {"loss_b": (0.5 * torch.mean(pred_b_p ** 2) + 0.5 * torch.mean(pred_b_m ** 2)
                                   - torch.mean(pred_b_p * expected_velocity_p)
                                   - torch.mean(pred_b_m * expected_velocity_m)),
                        "loss_z": loss_z}
            else:
                expected_velocity = self._interpolant.interpolate_derivative(t, x_0, x_1)
                if self._correct_center_of_mass_motion:
                    # scatter_mean is used to compute the mean velocity for every configuration.
                    # index_select is used to replicate the mean velocity for every atom in the configuration.
                    mean_velocity = torch.index_select(scatter_mean(expected_velocity, batch_indices, dim=0),
                                                       0, batch_indices)
                    expected_velocity = expected_velocity - mean_velocity
                return {"loss_b": torch.mean(pred_b ** 2) - 2.0 * torch.mean(pred_b * expected_velocity),
                        "loss_z": loss_z}
        else:
            return {"loss_z": loss_z}

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
        if self._predict_velocity:
            assert self._velocity_annealing_factor is not None
            odefunc = lambda t, x: ((1.0 + self._velocity_annealing_factor * t)
                                    * model_function(t, self._corrector.correct(x))[0])
        else:
            assert self._velocity_annealing_factor is None
            def odefunc(t, x):
                x_corr = self._corrector.correct(x)
                z = model_function(t, x_corr)[1]
                t1 = (self._interpolant.alpha_dot(t) * z)
                t2 = (self._interpolant.beta_dot(t) / self._interpolant.beta(t)
                      * (x_corr - self._interpolant.alpha(t) * z))
                return t1 + t2

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
            x_corr = self._corrector.correct(x.reshape(self._original_x_shape))
            z = self._model_func(t, x_corr)[1]
            t1 = self._interpolant.alpha_dot(t) * z
            t2 = self._interpolant.beta_dot(t) / self._interpolant.beta(t) * (x_corr - self._interpolant.alpha(t) * z)
            t3 = self._epsilon.epsilon(t) / self._interpolant.alpha(t) * z
            out = t1 + t2 - t3
            return out.reshape(new_x_shape)

        def g(self, t, x):
            return torch.sqrt(2.0 * self._epsilon.epsilon(t)) * torch.ones_like(x)

    class SDEPredictVelocity(SDE):
        def __init__(self, model_func, corrector, interpolant, epsilon, original_x_shape, velocity_annealing_factor):
            super().__init__(model_func=model_func, corrector=corrector, interpolant=interpolant, epsilon=epsilon,
                             original_x_shape=original_x_shape)
            self._velocity_annealing_factor = velocity_annealing_factor

        def f(self, t, x):
            # Because of the noise, the x should be corrected when it is passed to the model.
            new_x_shape = x.shape
            preds = self._model_func(t, self._corrector.correct(x.reshape(self._original_x_shape)))
            out = ((1.0 + self._velocity_annealing_factor * t) * preds[0]
                   - (self._epsilon.epsilon(t) / self._interpolant.alpha(t)) * preds[1])
            return out.reshape(new_x_shape)

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
        if self._predict_velocity:
            assert self._velocity_annealing_factor is not None
            sde = self.SDEPredictVelocity(model_func=model_function, corrector=self._corrector,
                                          interpolant=self._interpolant, epsilon=self._epsilon,
                                          original_x_shape=original_shape,
                                          velocity_annealing_factor=self._velocity_annealing_factor)
        else:
            assert self._velocity_annealing_factor is None
            sde = self.SDE(model_func=model_function, corrector=self._corrector, interpolant=self._interpolant,
                           epsilon=self._epsilon, original_x_shape=original_shape)
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
