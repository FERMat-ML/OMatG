from abc import ABC, abstractmethod
import torch


class TimeChecker(object):
    """
    Abstract class providing a method to check that all times in a tensor are in [0,1].
    """

    @staticmethod
    def _check_t(t: torch.tensor) -> torch.tensor:
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


class Interpolant(ABC, TimeChecker):
    """
    Abstract class for defining an interpolant I(t, x_0, x_1) in a stochastic interpolant between two points from two
    distributions p_0 and p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct interpolant.
        """
        pass

    @abstractmethod
    def interpolate(self, t: torch.tensor, x_0: torch.tensor, x_1: torch.tensor) -> torch.tensor:
        """
        Interpolate between two points from two distributions p_0 and p_1 at times t.

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
            Interpolated value.
        :rtype: torch.tensor
        """
        raise NotImplementedError

    @abstractmethod
    def interpolate_derivative(self, t: torch.tensor, x_0: torch.tensor, x_1: torch.tensor) -> torch.tensor:
        """
        Compute the derivative of the interpolant between two points from two distributions p_0 and p_1 at times t with
        respect to time.

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
            Derivative of the interpolant.
        :rtype: torch.tensor
        """
        raise NotImplementedError


class LatentGamma(ABC, TimeChecker):
    """
    Abstract class for defining the gamma function gamma(t) in a latent variable gamma(t) * z of a stochastic
    interpolant between two points from two distributions p_0 and p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct the gamma function.
        """
        pass

    @abstractmethod
    def gamma(self, t: torch.tensor) -> torch.tensor:
        """
        Evaluate the gamma function gamma(t) in the latent variable gamma(t) * z at the times t.

        :param t:
            Times in [0,1].
        :type t: torch.tensor

        :return:
            Gamma function gamma(t).
        :rtype: torch.tensor
        """
        raise NotImplementedError

    @abstractmethod
    def gamma_derivative(self, t: torch.tensor) -> torch.tensor:
        """
        Compute the derivative of the gamma function gamma(t) in the latent variable gamma(t) * z with respect to time.

        :param t:
            Times in [0,1].
        :type t: torch.tensor

        :return:
            Derivative of the gamma function.
        :rtype: torch.tensor
        """
        raise NotImplementedError


class StochasticInterpolant(ABC, TimeChecker):
    """
    Abstract class for defining a stochastic interpolant between two points from two distributions p_0 and p_1.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
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
        raise NotImplementedError

    def loss(self, model_prediction: tuple[torch.tensor, torch.tensor], t: torch.tensor, x_0: torch.tensor,
             x_1: torch.tensor) -> torch.tensor:
        """
        Compute the loss for the stochastic interpolant.

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
