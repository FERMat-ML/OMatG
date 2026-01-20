import warnings
import torch
from .abstracts import Epsilon


class ConstantEpsilon(Epsilon):
    """
    Epsilon function epsilon(t) = c that remains constant in time.

    :param c:
        Constant value of epsilon function.
    :type c: float
    """

    def __init__(self, c: float) -> None:
        """
        Construct constant epsilon function.
        """
        super().__init__()
        self._c = c

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
        self._check_t(t)
        return torch.full_like(t, self._c)


class VanishingEpsilon(Epsilon):
    """
    Epsilon function epsilon(t) = c * sqrt(1 - t).

    :params c:
        Constant in front of square root
    :type c: float
    """

    def __init__(self, c: float = 1.0, sigma: float = 0.01, mu: float = 0.075) -> None:
        """
        Construct epsilon. Some product of Fermi functions that 
        vanishes at endpoints
        """
        super().__init__()
        self._c = c
        self._sigma = sigma
        self._mu = mu

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
        self._check_t(t)
        f1 = 1 / (1 + torch.exp(- (t - self._mu) / self._sigma))
        f2 = 1 / (1 + torch.exp(- (1 - self._mu - t) / self._sigma))
        return self._c * f1 * f2


class ConstantNoiseSchedule(ConstantEpsilon):
    def __init__(self, noise_scale: float) -> None:
        """
        Construct constant epsilon function. This is basically converting between SI and FlowGRPO notation.
        """
        if not noise_scale >= 0.0:
            raise ValueError("Noise scale must be non-negative.")
        if noise_scale == 0.0:
            warnings.warn("Using zero noise scale in ConstantSigma will lead to no exploration at all times.")
        self._noise_scale = noise_scale
        c = noise_scale * noise_scale / 2.0
        super().__init__(c=c)


class SqrtNoiseSchedule(Epsilon):
    """
        Noise schedule that returns noise scales following a square root schedule a*sqrt((1-t)/t).

        This is the schedule used by Flow-GRPO (see https://arxiv.org/abs/2505.05470 and note their definition of t going
        from 1 to 0).

        :param noise_scale:
            The noise scale a, i.e., the noise at time 0.5.
        :type noise_scale: float

        :raises ValueError:
            If noise_scale is not positive.
        """

    def __init__(self, noise_scale: float) -> None:
        """Constructor of the SqrtNoiseSchedule class."""
        super().__init__()
        if not noise_scale >= 0.0:
            raise ValueError("Noise scale must be non-negative.")
        if noise_scale == 0.0:
            warnings.warn("Using zero noise scale in SqrtNoiseSchedule will lead to no exploration at all times.")
        self._noise_scale = noise_scale

    def epsilon(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get the noise level at times t.

        :param t:
            Times in [0, 1].
        :type t: torch.Tensor

        :return:
            Noise levels sigma(t).
        :rtype: torch.Tensor
        """
        self._check_t(t)
        return self._noise_scale * self._noise_scale / 2.0 * (1.0 - t) / t
