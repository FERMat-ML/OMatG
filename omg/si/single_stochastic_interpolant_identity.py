<<<<<<< HEAD
from typing import Optional
import torch
from .abstracts import Interpolant, LatentGamma, StochasticInterpolant
from enum import Enum, auto
import torch.nn as nn


class DE(Enum):
    ODE = auto()
    SDE = auto()

class SingleStochasticInterpolantIdentity(StochasticInterpolant):
    """
    Stochastic interpolant x_t = I(t, x_0, x_1) + gamma(t) * z between two points from two distributions p_0 and p_1 at
    times t based on an interpolant I(t, x_0, x_1), a gamma function gamma(t), and a Gaussian random variable z. Differs 
    from the SingleStochasticInterpolant class insofar as the quantity represented by x_0 and x_1 (such as atom types) must
    remain constant. 

    :param interpolant:
        Interpolant I(t, x_0, x_1) between two points from two distributions p_0 and p_1 at times t.
    :type interpolant: Interpolant
    :param gamma:
        Gamma function gamma(t) in the latent variable gamma(t) * z of a stochastic interpolant.
    :type gamma: LatentGamma
    """

    def __init__(self, interpolant: Interpolant, gamma: Optional[LatentGamma], de_type: DE) -> None:
        """Construct stochastic interpolant."""
        super().__init__()
        self._interpolant = interpolant
        self._gamma = gamma
        self._de_type = de_type
        if self._de_type == DE.ODE:
            self.loss = self._ode_loss
            self.integrate = self._ode_integrate
        else:
            assert self._de_type == DE.SDE
            self.loss = self._sde_loss
            self.integrate = self._sde_integrate


    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.tensor:
        """
        Stochastically interpolate between two points from two distributions p_0 and p_1 at times t.

        :param t:
            Times in [0,1].
        :type t: torch.tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.tensor, must be same as x_0

        :return:
            Stochastically interpolated value.
        :rtype: torch.tensor
        """
        assert torch.equal(x_0, x_1)
        interpolate = self._interpolant.interpolate(t, x_0, x_1)
        if self._gamma is not None:
            interpolate += self._gamma.gamma(t) * torch.randn_like(t)
        return interpolate


    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.tensor:
        """
        Derivative with respect to time of the stochastic interpolants between two points from two distributions p_0
        and p_1 at times t.

        :param t:
            Times in [0,1].
        :type t: torch.tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.tensor, must be same as x_0

        :return:
            Stochastically interpolated value.
        :rtype: torch.tensor
        """ 
        assert torch.equal(x_0, x_1)
        self._check_t(t)
        interpolate_derivative = self._interpolant.interpolate_derivative(t, x_0, x_1)
        if self._gamma is not None:
            interpolate_derivative += self._gamma.gamma_derivative(t) * torch.randn_like(t)
        return interpolate_derivative


    def loss(self, model_prediction: tuple[torch.tensor, torch.tensor], t: torch.tensor, x_0: torch.tensor,
             x_1: torch.tensor) -> torch.tensor:
        """
        Compute the loss for the stochastic interpolant.

        This method is only defined here to define all methods of the abstract base class. The actual loss method is
        either _ode_loss or _sde_loss, which are chosen based on the type of differential equation (self._de_type).

        :param model_prediction:
            Model prediction for the velocity field b and the denoiser eta.
        :type model_prediction: tuple[torch.tensor, torch.tensor]
        :param t:
            Times in [0,1].
        :type t: torch.tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.tensor, must be same as x_0

        :return:
            Loss.
        :rtype: torch.tensor
        """
        raise NotImplementedError

    def _ode_loss(self, pred: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor):
        """
        Compute loss function for stochastic interpolant under ODE

        :param t:
            Times in [0,1].
        :type t: torch.tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.tensor, must be same as x_0

        :return:
            Loss function value.
        :rtype: torch.tensor 
        """

        # Compute ground truth
        assert torch.equal(x_0, x_1)
        gt = self.interpolate_derivative(t, x_0, x_1)
        loss = nn.MSELoss(gt, pred[0])
        return loss

    def _sde_loss(self, pred: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor):
        """
        Compute loss function for stochastic interpolant under SDE

        :param t:
            Times in [0,1].
        :type t: torch.tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.tensor, must be same as x_0

        :return:
            Tuple of loss function values.
        :rtype: (torch.tensor, torch.tensor)
        """

        # Compute ground truth
        assert torch.equal(x_0, x_1)
        gt_b = self.interpolate_derivative(t, x_0, x_1)
        gt_z = torch.randn_like(t)
        loss_b = nn.MSELoss(gt_b, pred[0])
        loss_z = nn.MSELoss(gt_z, pred[1])
        return loss_b + loss_z



class SingleStochasticInterpolant(StochasticInterpolant):
    """
    Stochastic interpolant x_t = I(t, x_0, x_1) + gamma(t) * z between two points from two distributions p_0 and p_1 at
    times t based on an interpolant I(t, x_0, x_1), a gamma function gamma(t), and a Gaussian random variable z.

    :param interpolant:
        Interpolant I(t, x_0, x_1) between two points from two distributions p_0 and p_1 at times t.
    :type interpolant: Interpolant
    :param gamma:
        Gamma function gamma(t) in the latent variable gamma(t) * z of a stochastic interpolant.
    :type gamma: LatentGamma
    """

    def __init__(self, interpolant: Interpolant, gamma: Optional[LatentGamma], de_type: DE) -> None:
        """Construct stochastic interpolant."""
        super().__init__()
        self._interpolant = interpolant
        self._gamma = gamma
        self._de_type = de_type
        if self._de_type == DE.ODE:
            self.loss = self._ode_loss
            self.integrate = self._ode_integrate
        else:
            assert self._de_type == DE.SDE
            self.loss = self._sde_loss
            self.integrate = self._sde_integrate

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.tensor:
        """
        Stochastically interpolate between two points from two distributions p_0 and p_1 at times t.

        :param t:
            Times in [0,1].
        :type t: torch.tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.tensor

        :return:
            Stochastically interpolated value.
        :rtype: torch.tensor
        """
        assert x_0.shape == x_1.shape
        interpolate = self._interpolant.interpolate(t, x_0, x_1)
        if self._gamma is not None:
            interpolate += self._gamma.gamma(t) * torch.randn_like(t)
        return interpolate

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.tensor:
        """
        Derivative with respect to time of the stochastic interpolants between two points from two distributions p_0
        and p_1 at times t.

        :param t:
            Times in [0,1].
        :type t: torch.tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.tensor

        :return:
            Stochastically interpolated value.
        :rtype: torch.tensor
        """
        assert x_0.shape == x_1.shape
        self._check_t(t)
        interpolate_derivative = self._interpolant.interpolate_derivative(t, x_0, x_1)
        if self._gamma is not None:
            interpolate_derivative += self._gamma.gamma_derivative(t) * torch.randn_like(t)
        return interpolate_derivative

    def loss(self, model_prediction: tuple[torch.tensor, torch.tensor], t: torch.tensor, x_0: torch.tensor,
             x_1: torch.tensor) -> torch.tensor:
        """
        Compute the loss for the stochastic interpolant.

        This method is only defined here to define all methods of the abstract base class. The actual loss method is
        either _ode_loss or _sde_loss, which are chosen based on the type of differential equation (self._de_type).

        :param model_prediction:
            Model prediction for the velocity field b and the denoiser eta.
        :type model_prediction: tuple[torch.tensor, torch.tensor]
        :param t:
            Times in [0,1].
        :type t: torch.tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.tensor

        :return:
            Loss.
        :rtype: torch.tensor
        """
        raise NotImplementedError

    def _ode_loss(self, pred: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor):
        """
        Compute loss function for stochastic interpolant under ODE

        :param t:
            Times in [0,1].
        :type t: torch.tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.tensor

        :return:
            Loss function value.
        :rtype: torch.tensor 
        """

        # Compute ground truth
        gt = self.interpolate_derivative(t, x_0, x_1)
        loss = nn.MSELoss(gt, pred[0])
        return loss

    def _sde_loss(self, pred: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor):
        """
        Compute loss function for stochastic interpolant under SDE

        :param t:
            Times in [0,1].
        :type t: torch.tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.tensor

        :return:
            Tuple of loss function values.
        :rtype: (torch.tensor, torch.tensor)
        """

        # Compute ground truth
        gt_b = self.interpolate_derivative(t, x_0, x_1)
        gt_z = torch.randn_like(t)
        loss_b = nn.MSELoss(gt_b, pred[0])
        loss_z = nn.MSELoss(gt_z, pred[1])
        return loss_b + loss_z
=======
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

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                    batch_pointer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

        :return:
            Stochastically interpolated points x_t, random variables z used for interpolation.
        :rtype: tuple[torch.Tensor, torch.Tensor]

        """
        assert torch.equal(x_0, x_1)
        return x_0, torch.zeros_like(x_0)

    def loss(self, model_prediction: tuple[torch.Tensor, torch.Tensor], t: torch.Tensor, x_0: torch.Tensor,
             x_1: torch.Tensor, z: torch.Tensor, batch_pointer: torch.Tensor) -> torch.Tensor:
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
        :param z:
            Random variable z that was used for the stochastic interpolation to get the model prediction.
        :type z: torch.Tensor
        :param batch_pointer:
            Tensor of length batch_size + 1 containing the indices to the first atom in every batch plus the total
            number of atoms in the batch.
        :type batch_pointer: torch.Tensor

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
>>>>>>> origin/si-dev
