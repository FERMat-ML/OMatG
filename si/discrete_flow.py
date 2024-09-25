from typing import Callable
import torch
from .abstracts import StochasticInterpolant
from ..globals import MAX_ATOM_NUM
import torch.nn.functional as F
from torch.distributions import Categorical


class DiscreteFlowMatchingMask(StochasticInterpolant):
    """
    Class for discrete flow matching (DFM) between categorical distributions.
    Currently designed for masking base distributions
    """

    def __init__(self, S:int, mask:int, n_int:int, noise:float = 0) -> None:
        """
        Construct DFM
        :param S:
            Number of classes (possible atom types)
        :type S: int
        :param mask:
            Masking token
        :type mask: int
        :param n_int:
            Number of integration timesteps
        :type n_int: int    
        :param noise:
            Noise to add
        :type noise: float
        """
        super().__init__()
        self.S = MAX_ATOM_NUM
        self.mask = -42     # The answer to life, the universe, and everything
        self.nsteps = n_int
        self.noise = noise

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """
        Mask token based on the current time
        
        :param t:
            Times in [0,1]
        :type t: torch.Tensor
        :param x_0: torch.Tensor
            Points from p_0
        :type x_0: torch.Tensor
        :param x_1: 
            Points from p_1
        :type x_1: torch.Tensor

        :return:
            Masked point
        :rtype: torch.Tensor
        """

        # Mask atoms based on t
        assert x_0.shape == x_1.shape
        mask = torch.rand_like(x_1) < t
        x_1[mask] = self.mask

        # Return masked sequence
        return x_1

    def loss(self, model_prediction: tuple[torch.Tensor, torch.Tensor], t: torch.Tensor, x_0: torch.Tensor,
             x_1: torch.Tensor) -> torch.Tensor:
        """
        Cross entropy loss function for discrete flow matching

        :param model_prediction:
            Predicted final composition.
        :type model_prediction: tuple[torch.Tensor, torch.Tensor]
        :param t:  
            Times in [0,1]
        :type t: torch.Tensor
        :param x_0: 
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor

        :return:
            Loss.
        :rtype: torch.Tensor
        """

        # Return loss
        return F.cross_entropy(model_prediction, x_1)

    def integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  x_t: torch.Tensor, tspan: tuple[float, float]) -> torch.Tensor:
        """
        Integrate the current positions x_t from time tspan[0] to tspan[1] based on the predicted x_1.
        Implementation is based on https://arxiv.org/pdf/2402.04997. We construct rate matrix R by:
            R(i,j|x_1) = ReLU((dp(j|x_1)/dt) - (dp(x_t|x_1)) / (S * p(x_t|x_1))
        and add noise via "detailed balance" rate matrices R_db (see appendix):
            R_db(i,j|x_1) = eta * delta(i,x_1) + eta * delta(j, x_1) * (St + 1 - t) / (1 - t) 

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
        # Iterate time
        dt = (tspan[-1] - tspan[0]) / self.nsteps
        for t in torch.arange(0, 1, dt):

            # Predict x1 for the flattened sequence
            x_1 = model_function(t, x_t)[0]
            x_1_hot = F.one_hot(x_1, num_classes=self.S)  
            M_hot = F.one_hot(torch.tensor([self.mask]), num_classes=self.S)
            dpt = x_1_hot - M_hot 
            dpt_xt = dpt.gather(-1, x_t[:, None]).squeeze(-1)

            # Compute pt: linear interpolation based on t
            pt = (t * x_1_hot) + (1 - t) * M_hot
            pt_xt = pt.gather(-1, x_t[:, None]).squeeze(-1)

            # Compute the rate R
            R = F.relu(dpt - dpt_xt[:, None]) / (self.S * pt_xt[:, None])
            R[(pt_xt == 0.0)[:, None].repeat(1, self.S)] = 0.0
            R[pt == 0.0] = 0.0

            # Add noise if present
            if self.noise > 0.0:
                R_db = torch.zeros_like(R)
                R_db[x_t == x_1] = 1
                R_db[x_1 != x_t] = ((self.S * t) + 1 - t) / (1 - t)
                R_db *= self.noise

            # Compute step probabilities and sample
            step_probs = (R * dt).clamp(max=1.0)
            step_probs.scatter_(-1, x_t[:, None], 0.0)
            step_probs.scatter_(-1, x_t[:, None], 1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)

            # Sample the next x_t
            x_t = Categorical(step_probs).sample()

        # Return x_t
        return x_t