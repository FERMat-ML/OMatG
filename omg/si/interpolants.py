import torch
from .abstracts import Interpolant


class LinearInterpolant(Interpolant):
    """
<<<<<<< HEAD
    Linear interpolant I(t, x_0, x_1) = (1 - t) * x_0 + t * x_1 between two points from two distributions p_0 and p_1 at
    times t.
=======
    Linear interpolant I(t, x_0, x_1) = (1 - t) * x_0 + t * x_1 between points x_0 and x_1 from two distributions p_0
    and p_1 at times t.
>>>>>>> origin/si-dev
    """

    def __init__(self) -> None:
        """
        Construct linear interpolant.
        """
        super().__init__()

<<<<<<< HEAD
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
=======
    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                    batch_pointer: torch.Tensor) -> torch.Tensor:
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
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Interpolated value.
        :rtype: torch.Tensor
>>>>>>> origin/si-dev
        """
        assert self._check_t(t)
        return (1.0 - t) * x_0 + t * x_1

<<<<<<< HEAD
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
=======
    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                               batch_pointer: torch.Tensor) -> torch.Tensor:
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
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Derivative of the interpolant.
        :rtype: torch.Tensor
>>>>>>> origin/si-dev
        """
        self._check_t(t)
        return x_1 - x_0


class TrigonometricInterpolant(Interpolant):
    """
<<<<<<< HEAD
    Trigonometric interpolant I(t, x_0, x_1) = cos(pi / 2 * t) * x_0 + sin(pi / 2 * t) * x_1 between two points from two
    distributions p_0 and p_1 at times t.
=======
    Trigonometric interpolant I(t, x_0, x_1) = cos(pi / 2 * t) * x_0 + sin(pi / 2 * t) * x_1 between points x_0 and x_1
    from two distributions p_0 and p_1 at times t.
>>>>>>> origin/si-dev
    """

    def __init__(self) -> None:
        """
<<<<<<< HEAD
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
=======
        Construct trigonometric interpolant.
        """
        super().__init__()

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                    batch_pointer: torch.Tensor) -> torch.Tensor:
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
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Interpolated value.
        :rtype: torch.Tensor
>>>>>>> origin/si-dev
        """
        assert self._check_t(t)
        return torch.cos(torch.pi * t / 2.0) * x_0 + torch.sin(torch.pi * t / 2.0) * x_1

<<<<<<< HEAD
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
=======
    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                               batch_pointer: torch.Tensor) -> torch.Tensor:
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
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Derivative of the interpolant.
        :rtype: torch.Tensor
>>>>>>> origin/si-dev
        """
        assert self._check_t(t)
        return (-torch.pi / 2.0 * torch.sin(torch.pi * t / 2.0) * x_0
                + torch.pi / 2.0 * torch.cos(torch.pi * t / 2.0) * x_1)


<<<<<<< HEAD
class EncoderDecoderInterpolant(Interpolant):
    """
    Encoder-decoder interpolant
    I(t, x_0, x_1) = cos^2(pi * t) * 1_[0, 0.5) * x_0 + cos^2(pi * t) * 1_(0.5, 1] * x_1 between two points from two
    distributions p_0 and p_1 at times t.
=======
class PeriodicLinearInterpolant(Interpolant):
    """
    Linear interpolant I(t, x_0, x_1) = exp_(x_0)(t * log_(x_0))(x_1)) (see Eqs (11)-(13) in
    https://arxiv.org/pdf/2302.03660) between points x_0 and x_1 from two distributions p_0 and p_1 at times t on a
    periodic manifold.

    The exponential and logarithmic maps are given by:
    exp_x(v) = x + v - floor(x + v)
    log_v(x) = 1 / (2 * pi) * atan2(sin(2 * pi * (x - v)), cos(2 * pi * (x - v)))
    """

    def __init__(self) -> None:
        """
        Construct PeriodicLinearInterpolant.
        """
        super().__init__()

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                    batch_pointer: torch.Tensor) -> torch.Tensor:
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
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Interpolated value.
        :rtype: torch.Tensor
        """
        assert self._check_t(t)
        omega = 2.0 * torch.pi * (x_1 - x_0)
        out = t * torch.atan2(torch.sin(omega), torch.cos(omega)) / (2.0 * torch.pi)
        return out + x_1 - torch.floor(out + x_1)

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                               batch_pointer: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the interpolant between points x_0 and x_1 from two distributions p_0 and p_1 at times
        t with respect to time.

        This function implements -(log_(x_1)(x_0) - 1 / n sum_{i = 1}^n log_(x_1^i)(x_0^i)) where n is the number of
        atoms. The subtraction of the mean is necessary for translational invariance (compare Eq. (15) in
        https://arxiv.org/pdf/2302.03660, the first minus sign will be negated once the output of this function is used
        in a mse loss).

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
            Derivative of the interpolant.
        :rtype: torch.Tensor
        """
        assert self._check_t(t)
        omega = 2.0 * torch.pi * (x_0 - x_1)
        out = torch.atan2(torch.sin(omega), torch.cos(omega)) / (2.0 * torch.pi)

        # Subtract mean w.r.t. number of atoms in each batch.
        for index in range(1, len(batch_pointer)):
            out[batch_pointer[index - 1]:batch_pointer[index]] -= (
                out[batch_pointer[index - 1]:batch_pointer[index]].mean(dim=0))
        return -out


class EncoderDecoderInterpolant(Interpolant):
    """
    Encoder-decoder interpolant
    I(t, x_0, x_1) = cos^2(pi * t) * 1_[0, 0.5) * x_0 + cos^2(pi * t) * 1_(0.5, 1] * x_1 between points x_0 and x_1 from
    two distributions p_0 and p_1 at times t.
>>>>>>> origin/si-dev
    """

    def __init__(self) -> None:
        """
        Construct encoder-decoder interpolant.
        """
        super().__init__()

<<<<<<< HEAD
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
=======
    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                    batch_pointer: torch.Tensor) -> torch.Tensor:
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
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Interpolated value.
        :rtype: torch.Tensor
>>>>>>> origin/si-dev
        """
        assert self._check_t(t)
        return (torch.cos(torch.pi * t) ** 2) * torch.where(t < 0.5, x_0, x_1)

<<<<<<< HEAD
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
=======
    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                               batch_pointer: torch.Tensor) -> torch.Tensor:
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
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Derivative of the interpolant.
        :rtype: torch.Tensor
>>>>>>> origin/si-dev
        """
        assert self._check_t(t)
        return (-2.0 * torch.cos(torch.pi * t) * torch.pi * torch.sin(torch.pi * t)
                * torch.where(t < 0.5, x_0, x_1))


class MirrorInterpolant(Interpolant):
    """
<<<<<<< HEAD
    Mirror interpolant I(t, x_0, x_1) = x_1 between two points from the same distribution p_1 at times t.
=======
    Mirror interpolant I(t, x_0, x_1) = x_1 between points x_0 and x_1 from the same distribution p_1 at times t.
>>>>>>> origin/si-dev
    """

    def __init__(self) -> None:
        """
<<<<<<< HEAD
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
=======
        Construct mirror interpolant.
        """
        super().__init__()

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                    batch_pointer: torch.Tensor) -> torch.Tensor:
        """
        Interpolate between points x_0 and x_1 from two distributions p_0 and p_1 at times t.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Points from p_0 = p_1.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Interpolated value.
        :rtype: torch.Tensor
>>>>>>> origin/si-dev
        """
        assert self._check_t(t)
        return x_1

<<<<<<< HEAD
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
=======
    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                               batch_pointer: torch.Tensor) -> torch.Tensor:
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
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Derivative of the interpolant.
        :rtype: torch.Tensor
>>>>>>> origin/si-dev
        """
        assert self._check_t(t)
        return torch.zeros_like(x_1)


class ScoreBasedDiffusionModelInterpolant(Interpolant):
    """
<<<<<<< HEAD
    Interpolant I(t, x_0, x_1) = sqrt(1 - t^2) * x_0 + t * x_1 that can be used to reproduce score-based diffusion
    models.
=======
    Interpolant I(t, x_0, x_1) = sqrt(1 - t^2) * x_0 + t * x_1 between points x_0 and x_1 from
    two distributions p_0 (assumed to be Gaussian here) and p_1 at times t that can be used to reproduce score-based
    diffusion models.
>>>>>>> origin/si-dev
    """

    def __init__(self) -> None:
        """
        Construct VP interpolant
        """
        super().__init__()

<<<<<<< HEAD
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
=======
    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                    batch_pointer: torch.Tensor) -> torch.Tensor:
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
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Interpolated value.
        :rtype: torch.Tensor
>>>>>>> origin/si-dev
        """
        assert self._check_t(t)
        return torch.sqrt(1.0 - (t ** 2)) * x_0 + t * x_1

<<<<<<< HEAD
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
=======
    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                               batch_pointer: torch.Tensor) -> torch.Tensor:
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
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Derivative of the interpolant.
        :rtype: torch.Tensor
>>>>>>> origin/si-dev
        """
        assert self._check_t(t)
        return -t / torch.sqrt(1.0 - (t ** 2)) * x_0 + x_1
