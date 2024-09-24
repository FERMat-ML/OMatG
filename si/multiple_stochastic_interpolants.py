import torch
from typing import Sequence
from .abstracts import StochasticInterpolant


class MultipleStochasticInterpolants(StochasticInterpolant):
    def __init__(self, stochastic_interpolants: Sequence[StochasticInterpolant], costs: Sequence[float],
                 integration_time_steps: int) -> None:
        super().__init__()
        if not len(stochastic_interpolants) == len(costs):
            raise ValueError("The number of stochastic interpolants and costs must be equal.")
        if not integration_time_steps > 0:
            raise ValueError("The number of integration time steps must be positive.")
        self._stochastic_interpolants = stochastic_interpolants
        self._costs = costs
        self._integration_time_steps = integration_time_steps

    def interpolate(self, t: torch.Tensor, x_0: tuple[torch.Tensor, ...],
                    x_1: tuple[torch.Tensor, ...]) -> tuple[torch.tensor, ...]:
        assert len(x_0) == len(self._stochastic_interpolants)
        assert len(x_1) == len(self._stochastic_interpolants)
        return tuple(stochastic_interpolant.interpolate(t, (x_0_i,), (x_1_i,))
                     for stochastic_interpolant, x_0_i, x_1_i in zip(self._stochastic_interpolants, x_0, x_1))

    def loss(self, model_prediction: tuple[tuple[torch.tensor, torch.tensor], ...],
             t: torch.tensor, x_0: tuple[torch.tensor, ...], x_1: tuple[torch.tensor, ...]) -> torch.tensor:
        assert len(model_prediction) == len(self._stochastic_interpolants)
        assert len(x_0) == len(self._stochastic_interpolants)
        assert len(x_1) == len(self._stochastic_interpolants)
        # noinspection PyTypeChecker
        return sum(cost * stochastic_interpolant.loss((model_prediction_i,), t, (x_0_i,), (x_1_i,))
                   for cost, stochastic_interpolant, model_prediction_i, x_0_i, x_1_i
                   in zip(self._costs, self._stochastic_interpolants, model_prediction, x_0, x_1))

    def integrate(self, x_0: tuple[torch.tensor, ...], model) -> tuple[torch.tensor, ...]:
        assert len(x_0) == len(self._stochastic_interpolants)
        times = torch.linspace(0.0, 1.0, self._integration_time_steps)
        x_t = [x_i.clone() for x_i in x_0]
        new_x_t = [x_i.clone() for x_i in x_0]
        for t_index in range(1, len(times)):
            tspan = (times[t_index - 1], times[t_index])
            for stochastic_interpolant_index, stochastic_interpolant in enumerate(self._stochastic_interpolants):
                model_prediction_fn = lambda t, x: model(t, tuple(x_i if i != stochastic_interpolant_index else x
                                                                  for i, x_i in enumerate(x_0)))
                new_xi = stochastic_interpolant.integrate(model_prediction_fn, x_t[stochastic_interpolant_index], tspan)
                new_x_t[stochastic_interpolant_index] = new_xi
            x_t = new_x_t
        return tuple(x_t)
