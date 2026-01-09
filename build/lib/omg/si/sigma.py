from math import log
import torch
from .abstracts import Sigma


class GeometricSigma(Sigma):
    """
    Noise schedule sigma(s) = sigma_min (sigma_max / sigma_min)^s for a one-sided variance-exploding interpolant.

    The schedule implements a geometric progression of noise in time.

    :param sigma_min:
        The minimum noise level at s=0.
    :type sigma_min: float
    :param sigma_max:
        The maximum noise level at s=1.
    :type sigma_max: float

    :raises ValueError:
        If sigma_min is not positive or if sigma_max is not positive.
    :raises ValueError:
        If sigma_max is not greater than sigma_min.
    """

    def __init__(self, sigma_min: float, sigma_max: float) -> None:
        """
        Construct geometric sigma function.
        """
        super().__init__()
        if sigma_min <= 0.0:
            raise ValueError("sigma_min must be positive.")
        if sigma_max <= 0.0:
            raise ValueError("sigma_max must be positive.")
        if sigma_max <= sigma_min:
            raise ValueError("sigma_max must be greater than sigma_min.")
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._ratio = sigma_max / sigma_min
        self._log_ratio = log(self._ratio)

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
        self._check_t(s)
        return self._sigma_min * self._ratio ** s

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
        self._check_t(s)
        return self._sigma_min * self._log_ratio * self._ratio ** s
