import torch
from typing import Callable, Sequence
from .abstracts import StochasticInterpolant


class StochasticInterpolants(object):
    """
    Collection of several stochastic interpolants between points x_0 and x_1 from two distributions p_0 and
    p_1 at times t for different coordinate types x (like atom species, fractional coordinates, and lattice vectors).

    The number of stochastic interpolants must be equal to the number of costs, and it must match the number of
    different coordinate types in the collection of points x_0 and x_1.

    The loss returned by every stochastic interpolant is scaled by the corresponding cost factor.

    :param stochastic_interpolants:
        Sequence of stochastic interpolants for the different coordinate types.
    :type stochastic_interpolants: Sequence[StochasticInterpolant]
    :param costs:
        Cost factors for the different coordinate types that are used to scale their respective losses.
    :type costs: Sequence[float]
    :param integration_time_steps:
        Number of integration time steps for the integration of the collection of stochastic interpolants.

    :raises ValueError:
        If the number of stochastic interpolants and costs are not equal.
        If the number of integration time steps is not positive.
    """

    def __init__(self, stochastic_interpolants: Sequence[StochasticInterpolant], costs: Sequence[float],
                 integration_time_steps: int) -> None:
        """Constructor of the StochasticInterpolants class."""
        super().__init__()
        if not len(stochastic_interpolants) == len(costs):
            raise ValueError("The number of stochastic interpolants and costs must be equal.")
        if not integration_time_steps > 0:
            raise ValueError("The number of integration time steps must be positive.")
        if not all(cost >= 0.0 for cost in costs):
            raise ValueError("All cost factors must be non-negative.")
        if not abs(sum(costs) - 1.0) < 1e-10:
            raise ValueError("The sum of all cost factors must be approximately equal to 1.")
        self._stochastic_interpolants = stochastic_interpolants
        self._costs = costs
        self._integration_time_steps = integration_time_steps

    def interpolate(self, t: torch.Tensor, x_0: tuple[torch.Tensor, ...],
                    x_1: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """
        Stochastically interpolate between the collection of points x_0 and x_1 from the collection of two distributions
        p_0 and p_1 at times t.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Collection of points from the collection of distributions p_0.
        :type x_0: tuple[torch.Tensor, ...]
        :param x_1:
            Collection of points from the collection of distributions p_1.
        :type x_1: tuple[torch.Tensor, ...]

        :return:
            Collection of stochastically interpolated points x_t.
        :rtype: torch.Tensor
        """
        assert len(x_0) == len(self._stochastic_interpolants)
        assert len(x_1) == len(self._stochastic_interpolants)
        return tuple(stochastic_interpolant.interpolate(t, x_0_i, x_1_i)
                     for stochastic_interpolant, x_0_i, x_1_i in zip(self._stochastic_interpolants, x_0, x_1))

    def loss(self, model_prediction: tuple[tuple[torch.Tensor, torch.Tensor], ...],
             t: torch.Tensor, x_0: tuple[torch.Tensor, ...], x_1: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Compute the loss for the collection of stochastic interpolants between the collection of points x_0 and x_1 from
        a collection of distributions p_0 and p_1 at times t based on the collection of model predictions for the
        velocity fields b and the denoisers eta.

        The loss returned by every stochastic interpolant is scaled by the corresponding cost factor.

        :param model_prediction:
            Collection of model predictions for the velocity fields b and the denoisers eta.
        :type model_prediction: tuple[tuple[torch.Tensor, torch.Tensor], ...]
        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Collection of points from the collection of distributions p_0.
        :type x_0: tuple[torch.Tensor, ...]
        :param x_1:
            Collection of points from the collection of distributions p_1.
        :type x_1: tuple[torch.Tensor, ...]

        :return:
            Loss.
        :rtype: torch.Tensor
        """
        assert len(model_prediction) == len(self._stochastic_interpolants)
        assert len(x_0) == len(self._stochastic_interpolants)
        assert len(x_1) == len(self._stochastic_interpolants)
        # noinspection PyTypeChecker
        return sum(cost * stochastic_interpolant.loss(model_prediction_i, t, x_0_i, x_1_i)
                   for cost, stochastic_interpolant, model_prediction_i, x_0_i, x_1_i
                   in zip(self._costs, self._stochastic_interpolants, model_prediction, x_0, x_1))

    def integrate(self, x_0: tuple[torch.Tensor, ...],
                  model_function: Callable[[torch.Tensor, tuple[torch.Tensor, ...]],
                  tuple[tuple[torch.Tensor, torch.Tensor], ...]]) -> tuple[torch.Tensor, ...]:
        """
        Integrate the collection of points x_0 from the collection of distributions p_0 from time 0 to 1 based on the
        model that provides the collection of velocity fields b and denoisers eta.

        In principle, every stochastic interpolant could be integrated independently. However, the model function
        expects the updated positions of all stochastic interpolants at the same time. In this version, the integration
        is discretized in time. Every stochastic interpolant is integrated independently until the next time step based
        on the collection of points x_0 at the last timestep.

        :param x_0:
            Collection of initial points from the distributions p_0.
        :type x_0: tuple[torch.Tensor, ...]
        :param model_function:
            Model function returning the collection of velocity fields b and the denoisers eta given the current times t
            and collection of points x_t.
        :type model_function: Callable[[torch.Tensor, tuple[torch.Tensor, ...]], tuple[tuple[torch.Tensor, torch.Tensor], ...]]

        :return:
            Collection of integrated points x_1.
        :rtype: tuple[torch.Tensor, ...]
        """
        assert len(x_0) == len(self._stochastic_interpolants)
        times = torch.linspace(0.0, 1.0, self._integration_time_steps)
        x_t = [x_i.clone() for x_i in x_0]
        new_x_t = [x_i.clone() for x_i in x_0]
        for t_index in range(1, len(times)):
            tspan = (times[t_index - 1], times[t_index])
            for stochastic_interpolant_index, stochastic_interpolant in enumerate(self._stochastic_interpolants):
                model_prediction_fn = (
                    lambda t, x: model_function(t, tuple(x_i if i != stochastic_interpolant_index else x
                                                         for i, x_i in enumerate(x_t)))[stochastic_interpolant_index])
                new_xi = stochastic_interpolant.integrate(model_prediction_fn, x_t[stochastic_interpolant_index], tspan)
                new_x_t[stochastic_interpolant_index] = new_xi
            x_t = new_x_t
        return tuple(x_t)
