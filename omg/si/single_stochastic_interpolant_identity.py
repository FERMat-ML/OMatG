from typing import Callable, Dict, Iterable
import torch
from .abstracts import Corrector, StochasticInterpolantSpecies
from .corrector import IdentityCorrector


class SingleStochasticInterpolantIdentity(StochasticInterpolantSpecies):
    """
    Stochastic interpolant x_t = x_0 = x_1, between points x_0 and x_1 from two distributions p_0 and p_1 at
    times t. Differs from the SingleStochasticInterpolant class insofar as the quantity represented 
    by x_0 and x_1 (such as atom types) must be equal during interpolation.
    """

    def __init__(self) -> None:
        """Construct stochastic interpolant."""
        super().__init__()

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                    batch_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.
        :type batch_indices: torch.Tensor

        :return:
            Stochastically interpolated points x_t, random variables z used for interpolation.
        :rtype: tuple[torch.Tensor, torch.Tensor]

        """
        assert torch.equal(x_0, x_1)
        # Always return new object.
        return x_0.clone(), torch.zeros_like(x_0)

    def loss_keys(self) -> Iterable[str]:
        """
        Get the keys of the losses returned by the loss function.

        :return:
            Keys of the losses.
        :rtype: Iterable[str]
        """
        yield "loss"

    def loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
             t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
             batch_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the losses for the stochastic interpolant between points x_0 and x_1 from two distributions p_0 and
        p_1 at times t based on the model prediction for the velocity fields b and the denoisers eta.

        This class always returns a zero loss with the key 'loss'.

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta given the current positions x_t.
        :type model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor
        :param x_t:
            Stochastically interpolated points x_t.
        :type x_t: torch.Tensor
        :param z:
            Random variable z that was used for the stochastic interpolation to get the model prediction.
        :type z: torch.Tensor
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.
        :type batch_indices: torch.Tensor

        :return:
            Losses.
        :rtype: Dict[str, torch.Tensor]
        """
        assert torch.equal(x_0, x_1)
        return {"loss": torch.tensor(0.0, device=x_0.device)}

    def integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  x_t: torch.Tensor, time: torch.Tensor, time_step: torch.Tensor,
                  batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Integrate the current positions x_t at the given time for the given time step based on the velocity fields b and
        the denoisers eta returned by the model function.

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta given the current times t and positions
            x_t.
        :type model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        :param x_t:
            Current positions.
        :type x_t: torch.Tensor
        :param time:
            Initial time (0-dimensional torch tensor).
        :type time: torch.Tensor
        :param time_step:
            Time step (0-dimensional torch tensor).
        :type time_step: torch.Tensor
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.
        :type batch_indices: torch.Tensor

        :return:
            Integrated position.
        :rtype: torch.Tensor
        """
        # Always return new object.
        return x_t.clone()

    def get_corrector(self) -> Corrector:
        """
       Get the corrector implied by the stochastic interpolant (for instance, a corrector that considers periodic
       boundary conditions).

       :return:
           Corrector.
       :rtype: Corrector
       """
        return IdentityCorrector()

    def uses_masked_species(self) -> bool:
        """
        Return whether the stochastic interpolant uses masked species.

        :return:
            Whether the stochastic interpolant uses masked species.
        :rtype: bool
        """
        # Dataset does not contain masked species.
        return False
