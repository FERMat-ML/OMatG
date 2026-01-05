import torch
from omg_tf.abstracts import NoiseSchedule


class ConstantNoiseSchedule(NoiseSchedule):
    """
    Noise schedule that returns a constant noise scale.

    :param noise_scale:
        The constant noise scale.
    :type noise_scale: float

    :raises ValueError:
        If noise_scale is not positive.
    """

    def __init__(self, noise_scale: float) -> None:
        """Constructor of the ConstantNoiseSchedule class."""
        super().__init__()
        if not noise_scale > 0.0:
            raise ValueError("Noise scale must be positive.")
        self._noise_scale = noise_scale

    def noise(self, t: torch.Tensor) -> torch.Tensor:
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
        return torch.full_like(t, self._noise_scale)


class SqrtNoiseSchedule(NoiseSchedule):
    """
    Noise schedule that returns noise scales following a square root schedule a*sqrt((1-t)/t).

    This is the schedule used by Flow-GRPO (see https://arxiv.org/abs/2505.05470 and note their definition of t going
    from 1 to 0).

    :param noise_scale:
        The noise scale a.
    :type noise_scale: float

    :raises ValueError:
        If noise_scale is not positive.
    """

    def __init__(self, noise_scale: float) -> None:
        """Constructor of the SqrtNoiseSchedule class."""
        super().__init__()
        if not noise_scale > 0.0:
            raise ValueError("Noise scale must be positive.")
        self._noise_scale = noise_scale

    def noise(self, t: torch.Tensor) -> torch.Tensor:
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
        return self._noise_scale * torch.sqrt((1 - t) / t)
