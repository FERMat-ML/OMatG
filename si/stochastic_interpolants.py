import torch
from torch_geometric.data import Data
from typing import Callable, Sequence
from omg.globals import DATA_FIELDS
from .abstracts import StochasticInterpolant


class StochasticInterpolants(object):
    """
    Collection of several stochastic interpolants between points x_0 and x_1 from two distributions p_0 and
    p_1 at times t for different coordinate types x (like atom species, fractional coordinates, and lattice vectors).

    Every stochastic interpolant is associated with a data field and a cost factor. The possible data fields are defined
    in omg.globals.DATA_FIELDS. Data is transmitted using the torch_geometric.data.Data class which allows for accessing
    the data with a dictionary-like interface.

    The loss returned by every stochastic interpolant is scaled by the corresponding cost factor.

    :param stochastic_interpolants:
        Sequence of stochastic interpolants for the different coordinate types.
    :type stochastic_interpolants: Sequence[StochasticInterpolant]
    :param data_fields:
        Sequence of data fields for the different stochastic interpolants.
    :type data_fields: Sequence[str]
    :param costs:
        Cost factors for the different coordinate types that are used to scale their respective losses.
    :type costs: Sequence[float]
    :param integration_time_steps:
        Number of integration time steps for the integration of the collection of stochastic interpolants.

    :raises ValueError:
        If the number of stochastic interpolants and costs are not equal.
        If the number of stochastic interpolants and data fields are not equal.
        If the number of integration time steps is not positive.
    """

    def __init__(self, stochastic_interpolants: Sequence[StochasticInterpolant], data_fields: Sequence[str],
                 costs: Sequence[float], integration_time_steps: int) -> None:
        """Constructor of the StochasticInterpolants class."""
        super().__init__()
        if not len(stochastic_interpolants) == len(data_fields):
            raise ValueError("The number of stochastic interpolants and data fields must be equal.")
        if not all(data_field in DATA_FIELDS for data_field in data_fields):
            raise ValueError(f"All data fields must be in {DATA_FIELDS}.")
        if not len(costs) == len(stochastic_interpolants):
            raise ValueError("The number of stochastic interpolants and costs must be equal.")
        if not all(cost >= 0.0 for cost in costs):
            raise ValueError("All cost factors must be non-negative.")
        if not abs(sum(costs) - 1.0) < 1e-10:
            raise ValueError("The sum of all cost factors must be approximately equal to 1.")
        if not integration_time_steps > 0:
            raise ValueError("The number of integration time steps must be positive.")
        self._stochastic_interpolants = stochastic_interpolants
        self._data_fields = data_fields
        self._costs = costs
        self._integration_time_steps = integration_time_steps

    def interpolate(self, t: torch.Tensor, x_0: Data, x_1: Data) -> Data:
        """
        Stochastically interpolate between the collection of points x_0 and x_1 from the collection of two distributions
        p_0 and p_1 at times t.

        TODO: t will be of length batch size, we need to figure that out!

        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Collection of points from the collection of distributions p_0 stored in a torch_geometric.data.Data object.
        :type x_0: torch_geometric.data.Data
        :param x_1:
            Collection of points from the collection of distributions p_1 stored in a torch_geometric.data.Data object.
        :type x_1: torch_geometric.data.Data

        :return:
            Collection of stochastically interpolated points x_t stored in a torch_geometric.data.Data object.
        :rtype: torch_geometric.data.Data
        """
        x_0_dict = x_0.to_dict()
        x_1_dict = x_1.to_dict()
        x_t = x_0.clone(*self._data_fields)
        x_t_dict = x_t.to_dict()
        for stochastic_interpolant, data_field in zip(self._stochastic_interpolants, self._data_fields):
            assert data_field in x_0_dict
            assert data_field in x_1_dict
            assert data_field in x_t_dict
            x_t_dict[data_field] = stochastic_interpolant.interpolate(t, x_0_dict[data_field], x_1_dict[data_field])
        return x_t

    def loss(self, model_prediction: Data, t: torch.Tensor, x_0: Data, x_1: Data) -> torch.Tensor:
        """
        Compute the loss for the collection of stochastic interpolants between the collection of points x_0 and x_1 from
        a collection of distributions p_0 and p_1 at times t based on the collection of model predictions for the
        velocity fields b and the denoisers eta.

        This function expects that the velocity b and denoiser eta corresponding to the data field data_field are stored
        with the keys data_field_b and data_field_eta in the model prediction.

        The loss returned by every stochastic interpolant is scaled by the corresponding cost factor.

        :param model_prediction:
            Collection of model predictions for the velocity fields b and the denoisers eta stored in a
            torch_geometric.data.Data object.
        :type model_prediction: torch_geometric.data.Data
        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Collection of points from the collection of distributions p_0 stored in a torch_geometric.data.Data object.
        :type x_0: torch_geometric.data.Data
        :param x_1:
            Collection of points from the collection of distributions p_1 stored in a torch_geometric.data.Data object.
        :type x_1: torch_geometric.data.Data

        :return:
            Loss.
        :rtype: torch.Tensor
        """
        x_0_dict = x_0.to_dict()
        x_1_dict = x_1.to_dict()
        assert torch.equal(x_0.ptr, x_1.ptr)
        total_loss = torch.tensor(0.0)
        for cost, stochastic_interpolant, data_field in zip(self._costs, self._stochastic_interpolants,
                                                            self._data_fields):
            b_data_field = data_field + "_b"
            eta_data_field = data_field + "_eta"
            assert data_field in x_0_dict
            assert data_field in x_1_dict
            assert b_data_field in model_prediction
            assert eta_data_field in model_prediction
            total_loss += cost * stochastic_interpolant.loss(
                (model_prediction[b_data_field], model_prediction[eta_data_field]),
                t, x_0_dict[data_field], x_1_dict[data_field], x_0.ptr)
        return total_loss

    def integrate(self, x_0: Data, model_function: Callable[[torch.tensor, Data], Data]) -> Data:
        """
        Integrate the collection of points x_0 from the collection of distributions p_0 from time 0 to 1 based on the
        model that provides the collection of velocity fields b and denoisers eta.

        In principle, every stochastic interpolant could be integrated independently. However, the model function
        expects the updated positions of all stochastic interpolants at the same time. In this version, the integration
        is discretized in time. Every stochastic interpolant is integrated independently until the next time step based
        on the collection of points x_0 at the last timestep.

        :param x_0:
            Collection of points from the collection of distributions p_0 stored in a torch_geometric.data.Data object.
        :type x_0: torch_geometric.data.Data
        :param model_function:
            Model function returning the collection of velocity fields b and the denoisers eta stored in a
            torch_geometric.data.Data object given the current times t and collection of points x_t stored in a
            torch_geometric.data.Data object.
        :type model_function: Callable[[torch.Tensor, torch_geometric.data.Data], torch_geometric.data.Data]

        :return:
            Collection of integrated points x_1 stored in a torch_geometric.data.Data object.
        :rtype: torch_geometric.data.Data
        """
        times = torch.linspace(0.0, 1.0, self._integration_time_steps)
        x_t = x_0.clone(*self._data_fields)
        new_x_t = x_0.clone(*self._data_fields)
        x_t_dict = x_t.to_dict()
        new_x_t_dict = new_x_t.to_dict()
        assert all(data_field in x_t_dict for data_field in self._data_fields)
        assert all(data_field in new_x_t_dict for data_field in self._data_fields)

        for t_index in range(1, len(times)):
            tspan = (times[t_index - 1], times[t_index])
            for stochastic_interpolant, data_field in zip(self._stochastic_interpolants, self._data_fields):
                b_data_field = data_field + "_b"
                eta_data_field = data_field + "_eta"
                x_int = x_t.clone(*self._data_fields)
                x_int_dict = x_int.to_dict()

                def model_prediction_fn(t, x):
                    x_int_dict[data_field] = x
                    return model_function(t, x_int)[b_data_field], model_function(t, x_int)[eta_data_field]

                new_x_t_dict[data_field] = stochastic_interpolant.integrate(model_prediction_fn, x_t_dict[data_field],
                                                                            tspan)
            x_t = new_x_t.clone(*self._data_fields)
        return x_t
