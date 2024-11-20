from abc import ABC, abstractmethod
from typing import Callable
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

    @abstractmethod
    def compute_center_of_mass(self, x: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute the center of masses of the configurations in the batch with respect to the correction applied by this
        corrector (as, for instance, periodic boundary conditions).

        The first dimension of the input tensor x is the batch dimension. Here, different elements in the batch can
        belong to different configurations. The one-dimensional tensor batch_indices contains the indices of the
        configurations for the first dimension of the input tensor x. The center of mass for every configuration should
        be computed separately.

        The dimensions of the returned tensor should be the same as the dimensions of the input tensor x. This means
        that the center of mass for every configuration should be replicated for every element in the configuration.

        :param x:
            Input whose center of masses will be returned.
        :type x: torch.Tensor
        :param batch_indices:
            The batch indices for the input tensor x.
        :type batch_indices: torch.Tensor

        :return:
            Center of masses.
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
    Abstract class for defining an interpolant I(t, x_0, x_1) in a stochastic interpolant between points x_0 and x_1
    from two distributions p_0 and p_1 at times t.
    """

    @abstractmethod
    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """
        Interpolate between points x_0 and x_1 from two distributions p_0 and p_1 at times t.

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
        raise NotImplementedError

    @abstractmethod
    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the interpolant between points x_0 and x_1 from two distributions p_0 and p_1 at times
        t with respect to time.

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
        raise NotImplementedError

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


class StochasticInterpolant(ABC, TimeChecker):
    """
    Abstract class for defining a stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
    p_1 at times t.
    """

    @abstractmethod
    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
            Stochastically interpolated points x_t, random variables z used for interpolation.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
             t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
             batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
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
    @abstractmethod
    def uses_masked_species(self) -> bool:
        """
        Whether the stochastic interpolant uses an additional masked species.

        :return:
            Whether the stochastic interpolant uses an additional masked species.
        :rtype: bool
        """
        raise NotImplementedError
