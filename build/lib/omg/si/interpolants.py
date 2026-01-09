from math import exp
import torch
from .abstracts import Interpolant, Sigma, Tau
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
        return 1.0 - t

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
        return -torch.ones_like(t)

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
        return t.clone()

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the beta function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: torch.Tensor
        """
        return torch.ones_like(t)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()


class PeriodicLinearInterpolant(LinearInterpolant):
    """
    Linear interpolant I(t, x_0, x_1) = (1 - t) * x_0 + t * x_1 between points x_0 and x_1 from two distributions p_0
    and p_1 at times t with periodic boundary conditions. The coordinates are assumed to be in [0,1].
    """

    def __init__(self) -> None:
        """
        Construct PeriodicLinearInterpolant.
        """
        super().__init__()
        self._corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Corrector that corrects for periodic boundary conditions.
        :rtype: Corrector
        """
        return PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)


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
        return torch.cos(torch.pi * t / 2.0)

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
        return -(torch.pi / 2.0) * torch.sin(torch.pi * t / 2.0)

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
        return torch.sin(torch.pi * t / 2.0)

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the beta function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: torch.Tensor
        """
        return (torch.pi / 2.0) * torch.cos(torch.pi * t / 2.0)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()


class PeriodicTrigonometricInterpolant(TrigonometricInterpolant):
    """
    Trigonometric interpolant I(t, x_0, x_1) = cos(pi / 2 * t) * x_0 + sin(pi / 2 * t) * x_1 between points x_0 and x_1
    from two distributions p_0 and p_1 at times t with periodic boundary conditions. The coordinates are assumed to be
    in [0,1].
    """

    def __init__(self) -> None:
        """
        Construct periodic trigonometric interpolant.
        """
        super().__init__()
        self._corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Corrector that corrects for periodic boundary conditions.
        :rtype: Corrector
        """
        return self._corrector


class EncoderDecoderInterpolant(Interpolant):
    """
    Encoder-decoder interpolant
    I(t, x_0, x_1) = cos^2(pi * (t - switch_time * t)^p / ((switch_time - switch_time * t)^p + (t - switch_time * t)^p)) * 1_[0, switch_time) * x_0
                   + cos^2(pi * (t - switch_time * t)^p / ((switch_time - switch_time * t)^p + (t - switch_time * t)^p)) * 1_(1-switch_time, 1] * x_1
    between points x_0 and x_1 from two distributions p_0 and p_1 at times t.

    For p=1 and switch_time=0.5, this interpolant becomes
    I(t, x_0, x_1) = cos^2(pi * t) * 1_[0, switch_time) * x_0 + cos^2(pi * t) * 1_(1-switch_time, 1] * x_1,
    which was considered in the stochastic interpolants paper.

    Note that the time derivatives are only bounded for p>=0.5.

    :param switch_time:
        Time in (0, 1) at which to switch from x_0 to x_1.
        Defaults to 0.5.
    :type switch_time: float
    :param power:
        Power p in the interpolant.
        Defaults to 1.0.
    :type power: float

    :raises ValueError:
        If switch_time is not in (0,1) or power is less than 0.5.
    """

    def __init__(self, switch_time: float = 0.5, power: float = 1.0) -> None:
        """
        Construct encoder-decoder interpolant.
        """
        super().__init__()
        if switch_time <= 0.0 or switch_time >= 1.0:
            raise ValueError("Switch time must be in (0,1).")
        if power < 0.5:
            raise ValueError("Power must be at least 0.5.")
        self._switch_time = switch_time
        self._power = power

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
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power + a
        return torch.where(t <= self._switch_time, torch.cos(torch.pi * a / b) ** 2, 0.0)

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
        # In principle, one should be careful with floating point precision here as t->0 and t->1, especially for
        # self._power=1/2. However, time does not get arbitrarily close to 0 or 1 in practice. We assert this here so
        # this gets updated if omg.globals.SMALL_TIME or omg.globals.BIG_TIME change.
        assert torch.all((1.0e-3 <= t) & (t <= 1.0 - 1.0e-3))
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power
        c = torch.sin(2.0 * torch.pi * a / (a + b))
        return torch.where(
            t <= self._switch_time,
            self._power * torch.pi * a * b * c / (t * (t - 1.0) * ((a + b) ** 2)),
            0.0)

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
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power + a
        return torch.where(t > self._switch_time, torch.cos(torch.pi * a / b) ** 2, 0.0)

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the beta function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: torch.Tensor
        """
        # In principle, one should be careful with floating point precision here as t->0 and t->1, especially for
        # self._power=1/2. However, time does not get arbitrarily close to 0 or 1 in practice. We assert this here so
        # this gets updated if omg.globals.SMALL_TIME or omg.globals.BIG_TIME change.
        assert torch.all((1.0e-3 <= t) & (t <= 1.0 - 1.0e-3))
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power
        c = torch.sin(2.0 * torch.pi * a / (a + b))
        return torch.where(
            t > self._switch_time,
            self._power * torch.pi * a * b * c / (t * (t - 1.0) * ((a + b) ** 2)),
            0.0)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()


class PeriodicEncoderDecoderInterpolant(EncoderDecoderInterpolant):
    """
    Encoder-decoder interpolant
    I(t, x_0, x_1) = cos^2(pi * (t - switch_time * t)^p / ((switch_time - switch_time * t)^p + (t - switch_time * t)^p)) * 1_[0, switch_time) * x_0
                   + cos^2(pi * (t - switch_time * t)^p / ((switch_time - switch_time * t)^p + (t - switch_time * t)^p)) * 1_(1-switch_time, 1] * x_1
    between points x_0 and x_1 from two distributions p_0 and p_1 at times t with periodic boundary conditions. The
    coordinates are assumed to be in [0,1].

    For p=1 and switch_time=0.5, this interpolant becomes
    I(t, x_0, x_1) = cos^2(pi * t) * 1_[0, switch_time) * x_0 + cos^2(pi * t) * 1_(1-switch_time, 1] * x_1,
    which was considered in the stochastic interpolants paper.

    Note that the time derivatives are only bounded for p>=0.5.

    :param switch_time:
        Time in (0, 1) at which to switch from x_0 to x_1.
        Defaults to 0.5.
    :type switch_time: float
    :param power:
        Power p in the interpolant.
        Defaults to 1.0.
    :type power: float

    :raises ValueError:
        If switch_time is not in (0,1) or power is less than 0.5.
    """

    def __init__(self, switch_time: float = 0.5, power: float = 1.0) -> None:
        """
        Construct periodic encoder-decoder interpolant.
        """
        super().__init__(switch_time=switch_time, power=power)
        self._corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Corrector that corrects for periodic boundary conditions.
        :rtype: Corrector
        """
        return self._corrector


class MirrorInterpolant(Interpolant):
    """
    Mirror interpolant I(t, x_0, x_1) = x_1 between points x_0 and x_1 from the same distribution p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct mirror interpolant.
        """
        super().__init__()

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
        return torch.zeros_like(t)

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
        return torch.zeros_like(t)

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
        return torch.ones_like(t)

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the beta function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: torch.Tensor
        """
        return torch.zeros_like(t)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()


class PeriodicMirrorInterpolant(MirrorInterpolant):
    """
    Mirror interpolant I(t, x_0, x_1) = x_1 between points x_0 and x_1 from the same distribution p_1 at times t with
    periodic boundary conditions. The coordinates are assumed to be in [0,1].
    """

    def __init__(self) -> None:
        """
        Construct periodic mirror interpolant.
        """
        super().__init__()
        self._corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Corrector that corrects for periodic boundary conditions.
        :rtype: Corrector
        """
        return self._corrector


class ScoreBasedDiffusionModelInterpolantVP(Interpolant):
    """
    Interpolant I(t, x_0, x_1) = sqrt(1 - tau^2(t)) * x_0 + tau(t) * x_1 between points x_0 and x_1 from two
    distributions p_0 (assumed to be Gaussian here) and p_1 at times t that can be used to reproduce variance-preserving
    score-based diffusion models.

    :param tau:
        Tau function that defines the noise schedule.
    :type tau: Tau
    """

    def __init__(self, tau: Tau) -> None:
        """
        Construct VP interpolant.
        """
        super().__init__()
        self._tau = tau
        self._one_over_e = 1.0 / exp(1.0)

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
        return torch.sqrt(1.0 - self._tau.tau(t) ** 2)

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
        tau = self._tau.tau(t)
        t_sqrt = torch.sqrt(1.0 - tau ** 2)
        tau_dot = self._tau.tau_dot(t)
        return -tau * tau_dot / t_sqrt

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
        return self._tau.tau(t)

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the beta function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: torch.Tensor
        """
        return self._tau.tau_dot(t)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()


class PeriodicScoreBasedDiffusionModelInterpolantVP(ScoreBasedDiffusionModelInterpolantVP):
    """
    Interpolant I(t, x_0, x_1) = sqrt(1 - tau^2(t)) * x_0 + tau(t) * x_1 between points x_0 and x_1 from two
    distributions p_0 (assumed to be Gaussian here) and p_1 at times t with periodic boundary conditions that can be
    used to reproduce variance-preserving score-based diffusion models.

    :param tau:
        Tau function that defines the noise schedule.
    :type tau: Tau
    """

    def __init__(self, tau: Tau) -> None:
        """
        Construct periodic VP interpolant.
        """
        super().__init__(tau=tau)
        self._corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Corrector that corrects for periodic boundary conditions.
        :rtype: Corrector
        """
        return self._corrector


class ScoreBasedDiffusionModelInterpolantVE(Interpolant):
    """
    Interpolant I(t, x_0, x_1) = sqrt(sigma_(1-t)^2 - sigma(0)^2) * x_0 + x_1 between points x_0 and x_1 from two
    distributions p_0 (assumed to be Gaussian here) and p_1 at times t that can be used to reproduce variance-exploding
    score-based diffusion models.

    :param sigma:
        Sigma function that defines the noise schedule.
    :type sigma: Sigma
    """

    def __init__(self, sigma: Sigma) -> None:
        """
        Construct VP interpolant.
        """
        super().__init__()
        self._sigma = sigma

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
        return torch.sqrt(self._sigma.sigma(1.0 - t) ** 2 - self._sigma.sigma(torch.zeros_like(t)) ** 2)

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
        sigma = self._sigma.sigma(1.0 - t)
        alpha = torch.sqrt(sigma ** 2 - self._sigma.sigma(torch.zeros_like(t)) ** 2)
        derivative = self._sigma.sigma_dot(1.0 - t)
        return - sigma * derivative / alpha

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
        return torch.ones_like(t)

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the beta function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: torch.Tensor
        """
        return torch.zeros_like(t)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()


class PeriodicScoreBasedDiffusionModelInterpolantVE(ScoreBasedDiffusionModelInterpolantVE):
    """
    Interpolant I(t, x_0, x_1) = sqrt(sigma_(1-t)^2 - sigma(0)^2) * x_0 + x_1 between points x_0 and x_1 from two
    distributions p_0 (assumed to be Gaussian here) and p_1 at times t with periodic boundary conditions that can be
    used to reproduce variance-exploding score-based diffusion models.

    :param sigma:
        Sigma function that defines the noise schedule.
    :type sigma: Sigma
    """

    def __init__(self, sigma: Sigma) -> None:
        """
        Construct periodic VP interpolant.
        """
        super().__init__(sigma=sigma)
        self._corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Corrector that corrects for periodic boundary conditions.
        :rtype: Corrector
        """
        return self._corrector
