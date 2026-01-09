import torch
from .abstracts import LatentGamma


class LatentGammaSqrt(LatentGamma):
    """
    Gamma function gamma(t) = sqrt(a * t * (1 - t)) in the latent variable gamma(t) * z of a stochastic interpolant.

    :param a:
        Constant a > 0.
    :type a: float
    """

    def __init__(self, a: float) -> None:
        """Construct gamma function."""
        super().__init__()
        if a <= 0.0:
            raise ValueError("Constant a must be positive.")
        self._a = a

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
        self._check_t(t)
        return torch.sqrt(self._a * t * (1.0 - t))

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
        self._check_t(t)
        return self._a * (1.0 - 2.0 * t) / (2.0 * torch.sqrt(self._a * t * (1.0 - t)))

    def requires_antithetic(self) -> bool:
        """
        Whether the gamma function requires antithetic sampling because its derivative diverges as t -> 0  or t -> 1.

        :return:
            Whether the gamma function requires antithetic sampling.
        :rtype: bool
        """
        return True


class LatentGammaEncoderDecoder(LatentGamma):
    """
    Gamma function
    gamma(t) = sqrt(a) * sin^2(pi * (t - switch_time * t)^p / ((switch_time - switch_time * t)^p + (t - switch_time * t)^p))
    in the latent variable gamma(t) * z of a stochastic interpolant.

    For a=1, p=1, and switch_time=0.5, this gamma function becomes gamma(t) = sin^2(pi * t), which was considered in the
    stochastic interpolants paper.

    Note that the time derivatives are only bounded for p>=0.5.

    :param a:
        Constant a > 0.
        Defaults to 1.0.
    :type a: float
    :param switch_time:
        Time in (0, 1) at which to switch from x_0 to x_1.
        Defaults to 0.5.
    :type switch_time: float
    :param power:
        Power p in the interpolant.
        Defaults to 1.0.
    :type power: float

    :raises ValueError:
        If switch_time is not in (0,1), power is less than 0.5, or a is smaller than or equal to zero.
    """

    def __init__(self, a: float = 1.0, switch_time: float = 0.5, power: float = 1.0) -> None:
        """Construct gamma function."""
        super().__init__()
        if a <= 0.0:
            raise ValueError("Constant a must be positive.")
        if switch_time <= 0.0 or switch_time >= 1.0:
            raise ValueError("Switch time must be in (0,1).")
        if power < 0.5:
            raise ValueError("Power must be at least 0.5.")
        self._sqrt_a = a ** 0.5
        self._switch_time = switch_time
        self._power = power

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
        self._check_t(t)
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power + a
        return self._sqrt_a * torch.sin(torch.pi * a / b) ** 2

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
        self._check_t(t)
        # In principle, one should be careful with floating point precision here as t->0 and t->1, especially for
        # self._power=1/2. However, time does not get arbitrarily close to 0 or 1 in practice. We assert this here so
        # this gets updated if omg.globals.SMALL_TIME or omg.globals.BIG_TIME change.
        assert torch.all((1.0e-3 <= t) & (t <= 1.0 - 1.0e-3))
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power
        c = torch.sin(2.0 * torch.pi * a / (a + b))
        return -self._sqrt_a * self._power * torch.pi * a * b * c / (t * (t - 1.0) * ((a + b) ** 2))

    def requires_antithetic(self) -> bool:
        """
        Whether the gamma function requires antithetic sampling because its derivative diverges as t -> 0 or t -> 1.

        :return:
            Whether the gamma function requires antithetic sampling.
        :rtype: bool
        """
        return False
