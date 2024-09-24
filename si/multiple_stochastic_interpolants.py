from typing import Sequence
import torch
from .abstracts import StochasticInterpolant


class MultipleStochasticInterpolants(StochasticInterpolant):
    def __init__(self, stochastic_interpolants: Sequence[StochasticInterpolant], costs: Sequence[float]) -> None:
        super().__init__()
        if not len(stochastic_interpolants) == len(costs):
            raise ValueError("The number of stochastic interpolants and costs must be equal.")
        self._stochastic_interpolants = stochastic_interpolants
        self._costs = costs

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
