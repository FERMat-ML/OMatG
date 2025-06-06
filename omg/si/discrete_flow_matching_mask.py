from typing import Callable, Dict, Iterable
import torch
from torch.distributions import Categorical
import torch.nn.functional as functional
from omg.globals import MAX_ATOM_NUM, BIG_TIME
from .abstracts import StochasticInterpolantSpecies


class DiscreteFlowMatchingMask(StochasticInterpolantSpecies):
    """
    Class for discrete flow matching (DFM) between categorical distributions based on https://arxiv.org/pdf/2402.04997.

    This class is currently designed for masking base distributions p_0 for the points x_0.

    :param noise:
        Parameter scaling the noise that should be added during integration.
    :type noise: float

    :raises ValueError:
        If the number of integration steps is less than or equal to 0 or the noise is less than 0.
    """

    def __init__(self, noise: float = 0.0) -> None:
        """
        Construct DiscreteFlowMatchingMask class.
        """
        super().__init__()
        if noise < 0.0:
            raise ValueError("Noise parameter must be greater than or equal to 0.")
        self._mask_index = 0  # Real atoms start at index 1.
        self._noise = noise

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                    batch_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate between points x_0 and x_1 from two distributions p_0 and p_1 at times t using discrete flow
        matching.

        The base points x_0 are expected to be entirely in the masked state.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.
        :type batch_indices: torch.Tensor

        :return:
            Interpolated points x_t, random variables z used for interpolation.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        assert x_0.shape == x_1.shape
        assert torch.all(x_0 == self._mask_index)  # Every atom should be masked in the initial state.
        assert torch.all(x_1 != self._mask_index)  # No atom should be masked in the final state.
        # Mask atoms based on t, see Eq. (6) in https://arxiv.org/pdf/2402.04997.
        x_t = x_0.clone()
        mask = torch.rand_like(x_0, dtype=t.dtype) < t
        x_t[mask] = x_1[mask]
        return x_t, torch.zeros_like(x_t)

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
        Compute the cross-entropy loss for the discrete flow matching between points x_0 and x_1 from two distributions
        p_0 and p_1 at times t based on the model prediction for the probability distributions over the species.

        In contrast to the other stochastic interpolants in this module, the model prediction is not the velocity fields
        b and denoisers eta. Instead, only one model prediction is required. The second model prediction is ignored
        in this class.

        This class returns a single loss with the key 'loss'.

        Given that x_0 is of shape (sum(n_atoms),) containing the species of every atom in the batch, the model
        prediction returns a tensor of shape (sum(n_atoms), MAX_ATOM_NUM) containing the probability distribution
        over the species (excluding the mask with token 0) of every atom in the batch.

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
        :type x_1: torch.Tensor:
        :param x_t:
            Stochastically interpolated points x_t:
        :type t: torch.Tensor
        :param z:
            Random variable z that was used for the stochastic interpolation to get the model prediction.
        :type z: torch.Tensor
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.
        :type batch_indices: torch.Tensor

        :return:
            Cross-entropy loss.
        :rtype: Dict[str, torch.Tensor]
        """
        assert x_0.shape == x_1.shape
        assert torch.all(x_0 == self._mask_index)  # Every atom should be masked in the initial state.
        assert torch.all(x_1 != self._mask_index)  # No atom should not be masked in the final state.
        # model_prediction[0][a_i, j] is the probability of atom a_i being of species j + 1.
        # In order to compute the cross-entropy loss, we need to correct for the shift of the species in x_1.
        pred = model_function(x_t)[0] 
        assert pred.shape == (x_0.shape[0], MAX_ATOM_NUM)
        return {"loss": functional.cross_entropy(pred, x_1 - 1)}

    def integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  x_t: torch.Tensor, time: torch.Tensor, time_step: torch.Tensor,
                  batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Integrate the current positions x_t at the given time for the given time step based on the probability
        distributions over the species.

        In contrast to the other stochastic interpolants in this module, the model prediction is not the velocity fields
        b and denoisers eta. Instead, only one model prediction is required. The second model prediction is ignored
        in this class.

        Given that x_0 is of shape (sum(n_atoms),) containing the species of every atom in the batch, the model
        prediction returns a tensor of shape (sum(n_atoms), MAX_ATOM_NUM) containing the probability distribution
        over the species (excluding the mask with token 0) of every atom in the batch.

        Following https://arxiv.org/pdf/2402.04997, we construct a rate matrix R by
        R(i,j|x_1) = ReLU((dp(j|x_1)/dt) - (dp(x_t|x_1)) / (S * p(x_t|x_1)),
        and add noise via "detailed balance" rate matrices R_db (see appendix)
        R_db(i,j|x_1) = eta * delta(i,x_1) + eta * delta(j, x_1) * (St + 1 - t) / (1 - t).
        We adapt the code in Listing 8 in https://arxiv.org/pdf/2402.04997.

        In addition, we discretize time instead of using a continuous time Markov process.

        TODO: I think we should implement a continuous-time Markov process in the future.

        TODO: We should introduce an abstract class based on the abstract functions in Listing 8. Then this should work
        for arbitrary base distributions.

        :param model_function:
            Model function returning the probability distributions over the species given the current times t and
            positions x_t.
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
        x_1_probs = functional.softmax(model_function(time, x_t)[0], dim=-1)  # Shape (sum(n_atoms), MAX_ATOM_NUM).
        x_1_probs = x_1_probs.reshape((-1, MAX_ATOM_NUM))
        # Sample from distribution for every of the sum(n_atoms) elements.
        # Shift the atom type by one to get the real species.
        x_1 = Categorical(x_1_probs).sample() + 1  # Shape (sum(n_atoms),).
        # Shape (sum(n_atoms),).
        will_unmask = torch.rand(x_t.shape, device=x_t.device) < (time_step * (1.0 + self._noise * time) / (1.0 - time))
        # Only unmasks currently masked positions.
        will_unmask = will_unmask * (x_t == self._mask_index)  # Shape (sum(n_atoms),).
        if abs(time + time_step - BIG_TIME) < 5e-3:  # Unmask everyone on the final step.
            will_unmask = (x_t == self._mask_index)
        will_mask = torch.rand(x_t.shape, device=x_t.device) < time_step * self._noise  # Shape (sum(n_atoms),).
        # Only re-mask currently unmasked positions.
        will_mask = will_mask * (x_t != self._mask_index)  # Shape (sum(n_atoms),).
        x_t[will_unmask] = x_1[will_unmask]
        if abs(time + time_step) < BIG_TIME - 5e-3:  # Don't re-mask on the final step.
            x_t[will_mask] = self._mask_index
        return x_t

    def uses_masked_species(self) -> bool:
        """
        Whether the stochastic interpolant uses an additional masked species.

        :return:
            Whether the stochastic interpolant uses an additional masked species.
        :rtype: bool
        """
        return True
