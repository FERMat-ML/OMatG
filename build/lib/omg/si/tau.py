from math import exp, pi, sin
import torch
from .abstracts import Tau


class TauConstantSchedule(Tau):
    """
    Tau function tau(t) = t corresponding to a constant noise schedule beta(s) = 2 in a variance-preserving interpolant.
    """

    def __init__(self) -> None:
        """
        Construct constant tau function.
        """
        super().__init__()

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
        self._check_t(t)
        return t.clone()

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
        self._check_t(t)
        return torch.ones_like(t)


class TauLinearSchedule(Tau):
    """
    Tau function tau(t) = exp(1/2 * beta_min * log(t) - 1/4 * (beta_max - beta_min) * log^2(t)) corresponding to a
    linear noise schedule beta(s) = beta_min + (beta_max - beta_min) * s in a variance-preserving interpolant.

    :param beta_min:
        The minimum noise level.
    :type beta_min: float
    :param beta_max:
        The maximum noise level.
    :type beta_max: float

    :raises ValueError:
        If beta_min is not positive or if beta_max is not positive.
    :raises ValueError:
        If beta_max is not greater than beta_min.
    """

    def __init__(self, beta_min: float, beta_max: float) -> None:
        """
        Construct linear tau function.
        """
        super().__init__()
        if beta_min <= 0.0:
            raise ValueError("beta_min must be positive.")
        if beta_max <= 0.0:
            raise ValueError("beta_max must be positive.")
        if beta_max <= beta_min:
            raise ValueError("beta_max must be greater than beta_min.")
        self._beta_min = beta_min
        self._beta_max = beta_max

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
        self._check_t(t)
        log_t = torch.log(t)
        return torch.exp(0.5 * self._beta_min * log_t - 0.25 * (self._beta_max - self._beta_min) * log_t ** 2)

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
        self._check_t(t)
        log_t = torch.log(t)
        exp_factor = torch.exp(0.5 * self._beta_min * log_t - 0.25 * (self._beta_max - self._beta_min) * log_t ** 2)
        return exp_factor * (0.5 * self._beta_min / t - 0.5 * (self._beta_max - self._beta_min) * log_t / t)


class TauCosineSchedule(Tau):
    """
    Tau function tau(t) = csc(pi / (2 + 2 * d)) * sin((pi + pi * log(t))/ (2 + 2 * d)) corresponding to a
    cosine noise schedule beta(s) = pi / (1 + d) tan(pi / 2 * (s + d) / (1 + d)) in a variance-preserving interpolant.

    :param offset:
        The offset d in the cosine noise schedule.
    :type offset: float

    :raises ValueError:
        If offset is not zero or positive.
    """

    def __init__(self, offset: float) -> None:
        """
        Construct linear tau function.
        """
        super().__init__()
        if offset < 0.0:
            raise ValueError("offset must be non-negative.")
        self._offset = offset
        self._offset_factor = pi / (2.0 + 2.0 * self._offset)
        self._csc_prefactor = 1.0 / sin(self._offset_factor)
        self._one_over_e = 1.0 / exp(1.0)

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
        self._check_t(t)
        return torch.where(
            t > self._one_over_e,
            self._csc_prefactor * torch.sin(self._offset_factor * (1.0 + torch.log(t))),
            0.0)

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
        self._check_t(t)
        return torch.where(
            t > self._one_over_e,
            self._csc_prefactor * self._offset_factor * torch.cos(self._offset_factor * (1.0 + torch.log(t))) / t,
            0.0)
