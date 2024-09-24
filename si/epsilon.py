import torch
from .abstracts import Epsilon

class ConstantEpsilon(Epsilon):
    def __init__(self, c: float) -> None:
        """
        Epsilon function that remains constant in time
        """
        super().__init__()
        self._c = c

    def epsilon(self, t: torch.tensor) -> torch.tensor:
        """
        Evaluate epsilon function epsilon(t) at time

        :param t:
            Times in [0,1].
        :type t: torch.tensor

        :return: 
            Epsilon function epsilon(t)
        :rtype: torch.tensor
        """
        self.check_t(t)
        return self._c