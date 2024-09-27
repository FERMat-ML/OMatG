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
    Epsilon function epsilon(t) = c * sqrt(1 - t)

    :params c:
        Constant in front of square root
    :type c: float
    """

    def __init__(self, c: float) -> None:
        """
        Construct epsilon
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
        ans = self._c * torch.sqrt(1 - t)
        return torch.full_like(t, ans)