from typing import Callable
import torch
from torch.distributions import Categorical
import torch.nn.functional as functional
from omg.globals import MAX_ATOM_NUM
from .abstracts import StochasticInterpolant


class DiscreteFlowMatchingMask(StochasticInterpolant):
    """
    Class for discrete flow matching (DFM) between categorical distributions based on https://arxiv.org/pdf/2402.04997.

    This class is currently designed for masking base distributions p_0 for the points x_0.

    :param number_integration_steps:
        Number of integration steps.
    :type number_integration_steps: int
    :param noise:
        Parameter scaling the noise that should be added during integration.
    :type noise: float

    :raises ValueError:
        If the number of integration steps is less than or equal to 0 or the noise is less than 0.
    """

    def __init__(self, number_integration_steps: int, noise: float = 0.0) -> None:
        """
        Construct DiscreteFlowMatchingMask class.
        """
        super().__init__()
        if number_integration_steps <= 0:
            raise ValueError("Number of integration steps must be greater than 0.")
        if noise < 0.0:
            raise ValueError("Noise parameter must be greater than or equal to 0.")
        # self.S = MAX_ATOM_NUM
        self._mask_index = MAX_ATOM_NUM
        self._number_integration_steps = number_integration_steps
        self._noise = noise

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
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

        :return:
            Interpolated points x_t.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        assert x_0.shape == x_1.shape
        assert torch.all(x_0 == self._mask_index)
        # Mask atoms based on t, see Eq. (6) in https://arxiv.org/pdf/2402.04997.
        mask = torch.rand_like(x_1) < t
        x_1[mask] = self._mask_index
        return x_1

    def loss(self, model_prediction: tuple[torch.Tensor, torch.Tensor], t: torch.Tensor, x_0: torch.Tensor,
             x_1: torch.Tensor, n_atoms: torch.Tensor, z: torch.tensor) -> torch.Tensor:
        """
        Compute the cross-entropy loss for the discrete flow matching between points x_0 and x_1 from two distributions
        p_0 and p_1 at times t based on the model prediction for the probability distributions over the species.

        In contrast to the other stochastic interpolants in this module, the model prediction is not the velocity fields
        b and denoisers eta. Instead, only one model prediction is required. The second model prediction is ignored
        in this class.

        Given that x_0 is of shape (sum(n_atoms),) containing the species of every atom in the batch, the model
        prediction returns a tensor of shape (sum(n_atoms), MAX_ATOM_NUM + 1) containing the probability distribution
        over the species (including the mask) of every atom in the batch.

        :param model_prediction:
            Model prediction for the probablity distributions over the species.
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
            Number of atoms in each crystal from batch.
        :type n_atoms: torch.Tensor
        :param z:
            Random variable.
        :type z: torch.Tensor

        :return:
            Cross-entropy loss.
        :rtype: torch.Tensor
        """
        assert x_0.shape == x_1.shape
        assert model_prediction[0].shape == (x_0.shape[0], MAX_ATOM_NUM + 1)
        return functional.cross_entropy(model_prediction[0], x_1)

    def integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  x_t: torch.Tensor, tspan: tuple[float, float]) -> torch.Tensor:
        """
        Integrate the current positions x_t from time tspan[0] to tspan[1] based on the probability distributions over
        the species.

        In contrast to the other stochastic interpolants in this module, the model prediction is not the velocity fields
        b and denoisers eta. Instead, only one model prediction is required. The second model prediction is ignored
        in this class.

        Given that x_0 is of shape (sum(n_atoms),) containing the species of every atom in the batch, the model
        prediction returns a tensor of shape (sum(n_atoms), MAX_ATOM_NUM + 1) containing the probability distribution
        over the species (including the mask) of every atom in the batch.

        Following https://arxiv.org/pdf/2402.04997, we construct a rate matrix R by
        R(i,j|x_1) = ReLU((dp(j|x_1)/dt) - (dp(x_t|x_1)) / (S * p(x_t|x_1)),
        and add noise via "detailed balance" rate matrices R_db (see appendix)
        R_db(i,j|x_1) = eta * delta(i,x_1) + eta * delta(j, x_1) * (St + 1 - t) / (1 - t).

        In addition, we discretize time instead of using a continuous time Markov process.

        TODO: I think we should implement a continuous-time Markov process in the future.

        :param model_function:
            Model function returning the probability distributions over the species given the current times t and
            positions x_t.
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
        # Iterate time
        dt = (tspan[-1] - tspan[0]) / self._number_integration_steps
        eps = torch.finfo(torch.float64).eps
        for t in torch.arange(0, 1, dt):

            # Predict x1 for the flattened sequence
            x_1 = model_function(t, x_t)[0]
            x_1_hot = functional.one_hot(x_1, num_classes=self.S)
            M_hot = functional.one_hot(torch.tensor([self._mask_index]), num_classes=self.S)
            dpt = x_1_hot - M_hot
            dpt_xt = dpt.gather(-1, x_t[:, None]).squeeze(-1)

            # Compute pt: linear interpolation based on t
            # TODO: consider adding functionality to use other types of interpolants
            pt = (t * x_1_hot) + (1 - t) * M_hot
            pt_xt = pt.gather(-1, x_t[:, None]).squeeze(-1)

            # Compute the rate R
            R = functional.relu(dpt - dpt_xt[:, None]) / (self.S * pt_xt[:, None])
            R[(pt_xt == 0.0)[:, None].repeat(1, self.S)] = 0.0
            R[pt == 0.0] = 0.0

            # Add noise if present
            if self._noise > 0.0:
                R_db = torch.zeros_like(R)
                R_db[x_t == x_1] = 1
                R_db[x_1 != x_t] = ((self.S * t) + 1 - t) / (1 - t + eps)
                R_db *= self._noise

            # Compute step probabilities and sample
            R += R_db
            step_probs = (R * dt).clamp(max=1.0)
            step_probs.scatter_(-1, x_t[:, None], 0.0)
            step_probs.scatter_(-1, x_t[:, None], 1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)

            # Sample the next x_t
            x_t = Categorical(step_probs).sample()

        # Return x_t
        return x_t
