import torch
from .abstracts import Interpolant


class LinearInterpolant(Interpolant):
    """
    Linear interpolant I(t, x_0, x_1) = (1 - t) * x_0 + t * x_1 between two points from two distributions p_0 and p_1 at
    times t.
    """

    def __init__(self) -> None:
        """
        Construct linear interpolant.
        """
        super().__init__()

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
        assert self._check_t(t)
        return (1.0 - t) * x_0 + t * x_1

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
        self._check_t(t)
        return x_1 - x_0


class TrigonometricInterpolant(Interpolant):
    """
    Trigonometric interpolant I(t, x_0, x_1) = cos(pi / 2 * t) * x_0 + sin(pi / 2 * t) * x_1 between two points from two
    distributions p_0 and p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct trigonometric interpolant
        """
        super().__init__()

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
        assert self._check_t(t)
        return torch.cos(torch.pi * t / 2.0) * x_0 + torch.sin(torch.pi * t / 2.0) * x_1

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
        assert self._check_t(t)
        return (-torch.pi / 2.0 * torch.sin(torch.pi * t / 2.0) * x_0
                + torch.pi / 2.0 * torch.cos(torch.pi * t / 2.0) * x_1)


class EncoderDecoderInterpolant(Interpolant):
    """
    Encoder-decoder interpolant
    I(t, x_0, x_1) = cos^2(pi * t) * 1_[0, 0.5) * x_0 + cos^2(pi * t) * 1_(0.5, 1] * x_1 between two points from two
    distributions p_0 and p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct encoder-decoder interpolant.
        """
        super().__init__()

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
        assert self._check_t(t)
        return (torch.cos(torch.pi * t) ** 2) * torch.where(t < 0.5, x_0, x_1)

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
        assert self._check_t(t)
        return (-2.0 * torch.cos(torch.pi * t) * torch.pi * torch.sin(torch.pi * t)
                * torch.where(t < 0.5, x_0, x_1))


class MirrorInterpolant(Interpolant):
    """
    Mirror interpolant I(t, x_0, x_1) = x_1 between two points from the same distribution p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct mirror interpolant
        """
        super().__init__()

    def interpolate(self, t: torch.tensor, x_0: torch.tensor, x_1: torch.tensor) -> torch.tensor:
        """
        Interpolate between two points from two distributions p_0 and p_1 at times t.

        :param t:
            Times in [0,1].
        :type t: torch.tensor
        :param x_0:
            Points from p_0 = p_1.
        :type x_0: torch.tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.tensor

        :return:
            Interpolated value.
        :rtype: torch.tensor
        """
        assert self._check_t(t)
        return x_1

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
        assert self._check_t(t)
        return torch.zeros_like(x_1)


class ScoreBasedDiffusionModelInterpolant(Interpolant):
    """
    Interpolant I(t, x_0, x_1) = sqrt(1 - t^2) * x_0 + t * x_1 that can be used to reproduce score-based diffusion
    models.
    """

    def __init__(self) -> None:
        """
        Construct VP interpolant
        """
        super().__init__()

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
        assert self._check_t(t)
        return torch.sqrt(1.0 - (t ** 2)) * x_0 + t * x_1

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
        assert self._check_t(t)
        return -t / torch.sqrt(1.0 - (t ** 2)) * x_0 + x_1
