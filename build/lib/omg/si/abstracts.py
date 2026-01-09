from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable
import torch


class TimeChecker(object):
    """
    Abstract class providing a method to check that all times in a tensor are in [0,1].
    """

    @staticmethod
    def _check_t(t: torch.Tensor) -> torch.Tensor:
        """
        Check that all times are in [0,1].

        The return value can be used in an assert statement.

        :param t:
            Times to check.
        :type t: torch.tensor

        :return:
            Whether all times are in [0,1].
        :rtype: torch.tensor
        """
        return torch.all((0.0 <= t) & (t <= 1.0))


class Corrector(ABC):
    """
    Abstract class for defining a corrector function that corrects the input x (for instance, wrapping back coordinates
    to a specific cell in periodic boundary conditions).
    """

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def unwrap(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """
        Correct the input x_1 based on the reference input x_0 (for instance, return the image of x_1 closest to x_0 in
        periodic boundary conditions).

        :param x_0:
            Reference input.
        :type x_0: torch.Tensor
        :param x_1:
            Input to correct.
        :type x_1: torch.Tensor

        :return:
            Unwrapped x_1 value.
        :rtype: torch.Tensor
        """
        raise NotImplementedError


class Epsilon(ABC, TimeChecker):
    """
    Abstract class for defining an epsilon function epsilon(t).
    """

    @abstractmethod
    def epsilon(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the epsilon function at times t.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Epsilon function epsilon(t).
        :rtype: torch.Tensor
        """
        raise NotImplementedError


class Interpolant(ABC, TimeChecker):
    """
    Abstract class for defining an interpolant I(t, x_0, x_1) = alpha(t) * x_0 + beta(t) * x_1 in a stochastic
    interpolant between points x_0 and x_1 from two distributions p_0 and p_1 at times t.
    """

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """
        Interpolate between points x_0 and x_1 from two distributions p_0 and p_1 at times t.

        In order to possibly allow for periodic boundary conditions, x_1 is first unwrapped based on the corrector of
        this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
        this unwrapping returns the closest image of x_1 to x_0. The interpolant is then computed based on the unwrapped
        x_1 and the alpha and beta functions.

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
            Interpolated value.
        :rtype: torch.Tensor
        """
        assert self._check_t(t)
        x_0prime = self.get_corrector().correct(x_0)
        x_1prime = self.get_corrector().unwrap(x_0prime, x_1)
        x_t = self.alpha(t) * x_0prime + self.beta(t) * x_1prime
        return self.get_corrector().correct(x_t)

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the interpolant between points x_0 and x_1 from two distributions p_0 and p_1 at times
        t with respect to time.

        In order to possibly allow for periodic boundary conditions, x_1 is first unwrapped based on the corrector of
        this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
        this unwrapping returns the closest image of x_1 to x_0. The interpolant derivative is then computed based on
        the unwrapped x_1 and the alpha and beta functions.

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
            Derivative of the interpolant.
        :rtype: torch.Tensor
        """
        assert self._check_t(t)
        x_0prime = self.get_corrector().correct(x_0)
        x_1prime = self.get_corrector().unwrap(x_0prime, x_1)
        return self.alpha_dot(t) * x_0prime + self.beta_dot(t) * x_1prime

    @abstractmethod
    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant (for instance, a corrector that considers periodic boundary
        conditions).

        :return:
            Corrector.
        :rtype: Corrector
        """
        raise NotImplementedError

    @abstractmethod
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Alpha function alpha(t) in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Values of the alpha function at the given times.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the alpha function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the alpha function at the given times.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Beta function beta(t) in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Values of the beta function at the given times.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def beta_dot(self, t: torch.Tensor):
        """
        Time derivative of the beta function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: torch.Tensor
        """
        raise NotImplementedError


class LatentGamma(ABC, TimeChecker):
    """
    Abstract class for defining the gamma function gamma(t) in a latent variable gamma(t) * z of a stochastic
    interpolant between points x_0 and x_1 from two distributions p_0 and p_1 at times t.
    """

    @abstractmethod
    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the gamma function gamma(t) in the latent variable gamma(t) * z at the times t.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Gamma function gamma(t).
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def gamma_derivative(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the gamma function gamma(t) in the latent variable gamma(t) * z with respect to time.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivative of the gamma function.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def requires_antithetic(self) -> bool:
        """
        Whether the gamma function requires antithetic sampling because its derivative diverges as t -> 0  or t -> 1.

        :return:
            Whether the gamma function requires antithetic sampling.
        :rtype: bool
        """
        raise NotImplementedError


class StochasticInterpolant(ABC, TimeChecker):
    """
    Abstract class for defining a stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
    p_1 at times t.
    """

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def loss_keys(self) -> Iterable[str]:
        """
        Get the keys of the losses returned by the loss function.

        :return:
            Keys of the losses.
        :rtype: List[str]
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
             t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
             batch_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the losses for the stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
        p_1 at times t based on the model prediction for the velocity fields b and the denoisers eta.

        Since there can be several losses (say, one for the velocity field b and one for the denoiser eta), this
        function returns a dictionary mapping from a loss label to the loss value.

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
        raise NotImplementedError

    @abstractmethod
    def integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  x_t: torch.Tensor, time: torch.Tensor, time_step: torch.Tensor,
                  batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Integrate the current positions x_t at the given time for the given time step based on the velocity fields b and
        the denoisers eta returned by the model function.

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

    @abstractmethod
    def get_corrector(self) -> Corrector:
        """
       Get the corrector implied by the stochastic interpolant (for instance, a corrector that considers periodic
       boundary conditions).

       :return:
           Corrector.
       :rtype: Corrector
       """
        raise NotImplementedError


class StochasticInterpolantSpecies(StochasticInterpolant, ABC):
    """
    Abstract class for defining a stochastic interpolant between species x_0 and x_1 from two distributions p_0 and
    p_1 at times t.
    """

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the stochastic interpolant.

        The stochastic interpolants for atom species should not define a corrector.

        :return:
            Corrector.
        :rtype: Corrector
        """
        raise RuntimeError("Corrector not defined for StochasticInterpolantSpecies.")

    @abstractmethod
    def uses_masked_species(self) -> bool:
        """
        Whether the stochastic interpolant uses an additional masked species.

        :return:
            Whether the stochastic interpolant uses an additional masked species.
        :rtype: bool
        """
        raise NotImplementedError


class Sigma(ABC, TimeChecker):
    """
    Abstract class for defining a noise schedule sigma(s) for a one-sided variance-exploding interpolant.

    The one-sided variance-exploding interpolant is defined as x_t = sqrt(sigma^2(1-t) - sigma^2(0)) * x_0 + x_1.
    Note the inversion of time, that is, the noise sigma(s) should be smallest at s=0 and largest at s=1.
    """

    @abstractmethod
    def sigma(self, s: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the sigma function at times s.

        :param s:
            Times in [0,1].
        :type s: torch.Tensor

        :return:
            Sigma function sigma(s).
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def sigma_dot(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the sigma function with respect to time.

        :param s:
            Times in [0,1].
        :type s: torch.Tensor

        :return:
            Derivative of the sigma function at the given times.
        :rtype: torch.Tensor
        """
        raise NotImplementedError


class Tau(ABC, TimeChecker):
    """
    Abstract class for defining the tau function tau(t) for a one-sided variance-preserving interpolant.

    The one-sided variance-preserving interpolant is defined as x_t = sqrt(1 - tau^2(t)) * x_0 + tau(t) * x_1.
    """

    @abstractmethod
    def tau(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the tau function at times t.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Tau function tau(t).
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def tau_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the tau function with respect to time.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivative of the tau function at the given times.
        :rtype: torch.Tensor
        """
        raise NotImplementedError
