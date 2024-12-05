import torch
from .abstracts import Interpolant
from .corrector import Corrector, IdentityCorrector, PeriodicBoundaryConditionsCorrector


class LinearInterpolant(Interpolant):
    """
    Linear interpolant I(t, x_0, x_1) = (1 - t) * x_0 + t * x_1 between points x_0 and x_1 from two distributions p_0
    and p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct linear interpolant.
        """
        super().__init__()

    def alpha(self, t:torch.Tensor):
        """
        Alpha term in stochastic interpolant

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Value of alpha.
        :rtype: torch.Tensor
        """
        return (1.0 - t)

    def alpha_dot(self, t: torch.Tensor):
        """
        Derivative of alpha term in stochastic interpolant

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Value of alpha derivative.
        :rtype: torch.Tensor
        """
        return - 1.0

    def beta(self, t:torch.Tensor):
        """
        Alpha term in stochastic interpolant

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Value of beta.
        :rtype: torch.Tensor
        """
        return t

    def beta_dot(self, t:torch.Tensor):
        """
        Derivative of beta term in stochastic interpolant

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Value of beta derivative.
        :rtype: torch.Tensor
        """
        return 1.0

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()


class TrigonometricInterpolant(Interpolant):
    """
    Trigonometric interpolant I(t, x_0, x_1) = cos(pi / 2 * t) * x_0 + sin(pi / 2 * t) * x_1 between points x_0 and x_1
    from two distributions p_0 and p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct trigonometric interpolant.
        """
        super().__init__()

    def alpha(self, t:torch.Tensor):
        """
        Alpha term in stochastic interpolant

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Value of alpha.
        :rtype: torch.Tensor
        """
        return torch.cos((torch.pi * t) / 2)

    def alpha_dot(self, t: torch.Tensor):
        """
        Derivative of alpha term in stochastic interpolant

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Value of alpha derivative.
        :rtype: torch.Tensor
        """
        return - (torch.pi / 2) * torch.sin((torch.pi * t) / 2)

    def beta(self, t:torch.Tensor):
        """
        Alpha term in stochastic interpolant

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Value of beta.
        :rtype: torch.Tensor
        """
        return torch.sin((torch.pi * t) / 2)

    def beta_dot(self, t:torch.Tensor):
        """
        Derivative of beta term in stochastic interpolant

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Value of beta derivative.
        :rtype: torch.Tensor
        """
        return (torch.pi / 2) * torch.cos((torch.pi * t) / 2)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()


class PeriodicLinearInterpolant(Interpolant):
    """
    Linear interpolant I(t, x_0, x_1) = exp_(x_0)(t * log_(x_0))(x_1)) (see Eqs (11)-(13) in
    https://arxiv.org/pdf/2406.04713) between points x_0 and x_1 from two distributions p_0 and p_1 at times t on a
    periodic manifold. The coordinates are assumed to be in [0,1].

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
        return out + x_0 - torch.floor(out + x_0)

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
        omega = 2.0 * torch.pi * (x_1 - x_0)
        out = torch.atan2(torch.sin(omega), torch.cos(omega)) / (2.0 * torch.pi)
        # TODO: This COM issue should be fixed differently.
        # Subtract mean w.r.t. number of atoms in each batch.
        for index in range(1, len(batch_pointer)):
            out[batch_pointer[index - 1]:batch_pointer[index]] -= (
                out[batch_pointer[index - 1]:batch_pointer[index]].mean(dim=0))
        return out

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Corrector that corrects for periodic boundary conditions.
        :rtype: Corrector
        """
        return PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)


class EncoderDecoderInterpolant(Interpolant):
    """
    Encoder-decoder interpolant
    I(t, x_0, x_1) = cos^2(pi * t) * 1_[0, 0.5) * x_0 + cos^2(pi * t) * 1_(0.5, 1] * x_1 between points x_0 and x_1 from
    two distributions p_0 and p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct encoder-decoder interpolant.
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
        return (torch.cos(torch.pi * t) ** 2) * torch.where(t < 0.5, x_0, x_1)

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
        """
        assert self._check_t(t)
        return (-2.0 * torch.cos(torch.pi * t) * torch.pi * torch.sin(torch.pi * t)
                * torch.where(t < 0.5, x_0, x_1))

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()


class MirrorInterpolant(Interpolant):
    """
    Mirror interpolant I(t, x_0, x_1) = x_1 between points x_0 and x_1 from the same distribution p_1 at times t.
    """

    def __init__(self) -> None:
        """
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
        """
        assert self._check_t(t)
        # Return new object here.
        return x_1.clone()

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
        """
        assert self._check_t(t)
        return torch.zeros_like(x_1)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()  # TODO: Sometimes you do want to have a Periodic Corrector here?


class ScoreBasedDiffusionModelInterpolant(Interpolant):
    """
    Interpolant I(t, x_0, x_1) = sqrt(1 - t^2) * x_0 + t * x_1 between points x_0 and x_1 from
    two distributions p_0 (assumed to be Gaussian here) and p_1 at times t that can be used to reproduce score-based
    diffusion models.
    """

    def __init__(self) -> None:
        """
        Construct VP interpolant
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
        return torch.sqrt(1.0 - (t ** 2)) * x_0 + t * x_1

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
        """
        assert self._check_t(t)
        return -t / torch.sqrt(1.0 - (t ** 2)) * x_0 + x_1

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()


class PeriodicScoreBasedDiffusionModelInterpolant(Interpolant):
    """
    Periodic interpolant mimicking score based diffusion model
    I(t, x_0, x_1) = I(t, x_0, x_1) = sqrt(1 - t^2) * x_0 + t * x_1 on a torus.
    between points x_0 and x_1 from two distributions p_0 (assumed to be Gaussian here) 
    and p_1 at times t that can be used to reproduce score-based diffusion models.
    """

    def __init__(self) -> None:
        """
        Construct periodic VP interpolant
        """
        super().__init__()
        self._corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                    batch_pointer: torch.Tensor) -> torch.Tensor:
        """
        Interpolate between points x_0 and x_1 from two distributions p_0 and p_1 at times t on a torus.

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

        # Correct for periodic boundaries by using the geodesic to move x_1 to x_1prime
        # Then wrap the interpolated value back onto the torus
        x_1prime = self._corrector.unwrap(x_0, x_1)
        x_t = torch.sqrt(1.0 - (t ** 2)) * x_0 + t * x_1prime
        return self._corrector.correct(x_t)

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                               batch_pointer: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the interpolant between points x_0 and x_1 on a torus
        from two distributions p_0 and p_1 at times t with respect to time.

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

        # Correct for periodic boundaries by using the geodesic to move x_1 to x_1prime
        # Unlike the interpolation, the derivative does not need to be corrected
        x_1prime = self._corrector.unwrap(x_0, x_1)
        out = -t / torch.sqrt(1.0 - (t ** 2)) * x_0 + x_1prime
        # TODO: This COM issue should be fixed differently.
        # Subtract mean w.r.t. number of atoms in each batch.
        for index in range(1, len(batch_pointer)):
            out[batch_pointer[index - 1]:batch_pointer[index]] -= (
                out[batch_pointer[index - 1]:batch_pointer[index]].mean(dim=0))
        return out

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Corrector that corrects for periodic boundary conditions.
        :rtype: Corrector
        """
        return self._corrector


class PeriodicTrigonometricInterpolant(Interpolant):
    """
    Periodic Trigonometric interpolant 
    I(t, x_0, x_1) = cos(pi / 2 * t) * x_0 + sin(pi / 2 * t) * x_1 
    between points x_0 and x_1 on a torus
    from two distributions p_0 and p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct periodic trigonometric interpolant.
        """
        super().__init__()
        self._corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                    batch_pointer: torch.Tensor) -> torch.Tensor:
        """
        Interpolate between points x_0 and x_1 from two distributions p_0 and p_1 at times t on a torus.

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
        x_1prime = self._corrector.unwrap(x_0, x_1)
        x_t = torch.cos(torch.pi * t / 2.0) * x_0 + torch.sin(torch.pi * t / 2.0) * x_1prime
        return self._corrector.correct(x_t)

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                               batch_pointer: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the interpolant between points x_0 and x_1 on a torus
        from two distributions p_0 and p_1 at times t with respect to time.

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
        x_1prime = self._corrector.unwrap(x_0, x_1)
        out = (-torch.pi / 2.0 * torch.sin(torch.pi * t / 2.0) * x_0
               + torch.pi / 2.0 * torch.cos(torch.pi * t / 2.0) * x_1prime)
        # TODO: This COM issue should be fixed differently.
        # Subtract mean w.r.t. number of atoms in each batch.
        for index in range(1, len(batch_pointer)):
            out[batch_pointer[index - 1]:batch_pointer[index]] -= (
                out[batch_pointer[index - 1]:batch_pointer[index]].mean(dim=0))
        return out

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Corrector that corrects for periodic boundary conditions.
        :rtype: Corrector
        """
        return self._corrector


class PeriodicEncoderDecoderInterpolant(Interpolant):
    """
    Periodic Encoder-decoder interpolant
    I(t, x_0, x_1) = cos^2(pi * t) * 1_[0, 0.5) * x_0 + cos^2(pi * t) * 1_(0.5, 1] * x_1 
    between points x_0 and x_1 on a torus 
    from two distributions p_0 and p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct periodic encoder-decoder interpolant.
        """
        super().__init__()
        self._corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                    batch_pointer: torch.Tensor) -> torch.Tensor:
        """
        Interpolate between points x_0 and x_1 from two distributions p_0 and p_1 at times t on a torus.

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
        x_1prime = self._corrector.unwrap(x_0, x_1)
        x_t = (torch.cos(torch.pi * t) ** 2) * torch.where(t < 0.5, x_0, x_1prime)
        return self._corrector.correct(x_t)

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                               batch_pointer: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the interpolant between points x_0 and x_1 on a torus
        from two distributions p_0 and p_1 at times t with respect to time.

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
        x_1prime = self._corrector.unwrap(x_0, x_1)
        out = (-2.0 * torch.cos(torch.pi * t) * torch.pi * torch.sin(torch.pi * t)
               * torch.where(t < 0.5, x_0, x_1prime))
        # TODO: This COM issue should be fixed differently.
        # Subtract mean w.r.t. number of atoms in each batch.
        for index in range(1, len(batch_pointer)):
            out[batch_pointer[index - 1]:batch_pointer[index]] -= (
                out[batch_pointer[index - 1]:batch_pointer[index]].mean(dim=0))
        return out

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Corrector that corrects for periodic boundary conditions.
        :rtype: Corrector
        """
        return self._corrector
