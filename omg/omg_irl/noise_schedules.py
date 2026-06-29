from abc import ABC, abstractmethod
import warnings
import torch
import torch.nn as nn
from omg.si.abstracts import TimeChecker


class NoiseSchedule(ABC, TimeChecker):
    """
    Abstract base class for defining a noise schedule sigma(t).

    This noise schedule is very similar to the epsilon functions in the omg.si module. However, they are used
    differently in the SDE integrations in the omg.si module and the omg.omg_irl module. They are connected by
    sigma^2(t) = 2 * epsilon(t). Moreover, the noise schedules in the omg.omg_irl module can optionally be
    learnable, which is not the case for the epsilon functions in the omg.si module.
    """

    def __init__(self) -> None:
        """Constructor of the NoiseSchedule class."""
        super().__init__()

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def learnable(self) -> bool:
        """
        Return whether the noise schedule has learnable parameters.

        :return:
            True if the noise schedule has learnable parameters, False otherwise.
        :rtype: bool
        """
        raise NotImplementedError


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
        if not noise_scale >= 0.0:
            raise ValueError("Noise scale must be non-negative.")
        if noise_scale == 0.0:
            warnings.warn("Using zero noise scale in ConstantNoiseSchedule will lead to no exploration at all times.")
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

    def learnable(self) -> bool:
        """
        Return whether the noise schedule has learnable parameters.

        :return:
            True if the noise schedule has learnable parameters, False otherwise.
        :rtype: bool
        """
        return False


class SqrtNoiseSchedule(NoiseSchedule):
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

    def learnable(self) -> bool:
        """
        Return whether the noise schedule has learnable parameters.

        :return:
            True if the noise schedule has learnable parameters, False otherwise.
        :rtype: bool
        """
        return False


class MLPNoiseSchedule(nn.Module, NoiseSchedule):
    """
    Noise schedule that uses an MLP to predict noise scales from time t.

    The MLP is initialized such that the initial noise scale at all times is equal to init_noise_scale.
    Positivity of the predicted noise scales is ensured via a softplus activation at the output.

    :param initial_noise_scale:
        Initial noise scale.
    :type initial_noise_scale: float
    :param hidden_dimension:
        Hidden dimension of the MLP.
    :type hidden_dimension: int
    :param number_hidden_layers:
        Number of hidden layers of the MLP.
    :type number_hidden_layers: int

    :raises ValueError:
        If init_noise_scale is not positive.
        If hidden_dimension is not positive.
        If number_hidden_layers is negative.
    """

    def __init__(self, initial_noise_scale: float, hidden_dimension: int = 64, number_hidden_layers: int = 2) -> None:
        """Constructor of the MLPNoiseSchedule class."""
        super().__init__()
        if not initial_noise_scale > 0.0:
            raise ValueError("Initial noise scale must be positive.")
        if not hidden_dimension > 0:
            raise ValueError("Hidden dimension must be positive.")
        if not number_hidden_layers >= 0:
            raise ValueError("Number of hidden layers must be non-negative.")
        self._initial_noise_scale = initial_noise_scale
        self._hidden_dimension = hidden_dimension
        self._number_hidden_layers = number_hidden_layers
        layers: list[nn.Module] = [nn.Linear(1, hidden_dimension), nn.SiLU()]
        for _ in range(number_hidden_layers):
            layers.append(nn.Linear(hidden_dimension, hidden_dimension))
            layers.append(nn.SiLU())
        final_layer = nn.Linear(hidden_dimension, 1)
        # Initialize final layer to output zero at the beginning.
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        self._mlp = nn.Sequential(*layers)

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
        # Ensure positive output via softplus activation.
        output = nn.functional.softplus(self._mlp(t.unsqueeze(-1))).squeeze(-1)
        # Make sure that the output is zero if model output is zero, softplus(0) = log(2).
        output -= torch.log(torch.tensor(2.0, device=t.device))
        return self._initial_noise_scale + output

    def learnable(self) -> bool:
        """
        Return whether the noise schedule has learnable parameters.

        :return:
            True if the noise schedule has learnable parameters, False otherwise.
        :rtype: bool
        """
        return True
