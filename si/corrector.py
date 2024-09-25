import torch
from .abstracts import Corrector


class PeriodicBoundaryConditionsCorrector(Corrector):
    """
    Corrector function that wraps back coordinates to the interval [min, max] with periodic boundary conditions.

    :param min:
        Minimum value of the interval.
    :type min: float
    :param max:
        Maximum value of the interval.
    :type max: float

    :raises ValueError:
        If the minimum value is greater than the maximum value.
    """

    def __init__(self, min: float, max: float) -> None:
        """
        Construct corrector function.
        """
        super().__init__()
        self._min = min
        self._max = max
        if self._min >= self._max:
            raise ValueError("Minimum value must be less than maximum value.")

    def correct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Correct the input x.

        :param x:
            Input to correct.
        :type x: torch.Tensor

        :return:
            Corrected input.
        :rtype: torch.Tensor
        """
        return torch.fmod(x - self._min, self._max - self._min) + self._min
