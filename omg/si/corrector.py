import torch
from .abstracts import Corrector


class IdentityCorrector(Corrector):
    """
    Corrector that does nothing.
    """

    def __init__(self):
        """Construct identity corrector."""
        super().__init__()

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
        return x


class PeriodicBoundaryConditionsCorrector(Corrector):
    """
    Corrector function that wraps back coordinates to the interval [min, max] with periodic boundary conditions.

    :param min_value:
        Minimum value of the interval.
    :type min_value: float
    :param max_value:
        Maximum value of the interval.
    :type max_value: float

    :raises ValueError:
        If the minimum value is greater than the maximum value.
    """

    def __init__(self, min_value: float, max_value: float) -> None:
        """
        Construct corrector function.
        """
        super().__init__()
        self._min_value = min_value
        self._max_value = max_value
        if self._min_value >= self._max_value:
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
        return torch.remainder(x - self._min_value, self._max_value - self._min_value) + self._min_value
