from typing import Callable
import torch
from .abstracts import StochasticInterpolant


class SingleStochasticInterpolantIdentity(StochasticInterpolant):
    """
    Stochastic interpolant x_t = x_0 = x_1, between points x_0 and x_1 from two distributions p_0 and p_1 at
    times t. Differs from the SingleStochasticInterpolant class insofar as the quantity represented 
    by x_0 and x_1 (such as atom types) must be equal during interpolation.
    """

    def __init__(self) -> None:
        """Construct stochastic interpolant."""
        super().__init__()

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """
        Stochastically interpolate between points x_0 and x_1 from two distributions p_0 and p_1 at times t.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor, must be same as x_0.

        :return:
            Stochastically interpolated value.
        :rtype: torch.Tensor
        """
        assert torch.equal(x_0, x_1)
        return x_0

    def loss(self, model_prediction: tuple[torch.Tensor, torch.Tensor], t: torch.Tensor, x_0: torch.Tensor,
             x_1: torch.Tensor, n_atoms: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
        p_1 at times t based on the model prediction for the velocity fields b and the denoisers eta.

        :param model_prediction:
            Model prediction for the velocity fields b and the denoisers eta.
        :type model_prediction: tuple[torch.Tensor, torch.Tensor]
        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor
        :param n_atoms:
            Number of atoms in each crystal from batch
        :type n_atoms: torch.Tensor

        :return:
            Loss.
        :rtype: torch.Tensor
        """
        assert torch.equal(x_0, x_1)
        return torch.tensor(0.0)

    def integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  x_t: torch.Tensor, tspan: tuple[float, float]) -> torch.Tensor:
        """
        Integrate the current positions x_t from time tspan[0] to tspan[1] based on the velocity fields b and the
        denoisers eta returned by the model function.

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta given the current times t and positions
            x_t.
        :type model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        :param x_t:
            Current positions.
        :type x_t: torch.Tensor
        :param tspan:
            Time span for integration.
        :type tspan: tuple[float, float]

        :return:
            Integrated position.
        :rtype: torch.Tensor
        """
        return x_t
