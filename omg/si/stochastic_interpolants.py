import torch
from torch_geometric.data import Data
from typing import Callable, Sequence
from omg.globals import reshape_t, DataField, SMALL_TIME, BIG_TIME
from .abstracts import StochasticInterpolant


class StochasticInterpolants(object):
    """
    Collection of several stochastic interpolants between points x_0 and x_1 from two distributions p_0 and
    p_1 at times t for different coordinate types x (like atom species, fractional coordinates, and lattice vectors).

    Every stochastic interpolant is associated with a data field and a cost factor. The possible data fields are defined
    in the omg.globals.DataField enumeration. Data is transmitted using the torch_geometric.data.Data class which allows
    for accessing the data with a dictionary-like interface.

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
    """  # noqa: E501

    def __init__(self, stochastic_interpolants: Sequence[StochasticInterpolant], data_fields: Sequence[str],
                 costs: Sequence[float], integration_time_steps: int) -> None:
        """Constructor of the StochasticInterpolants class."""
        super().__init__()
        if not len(stochastic_interpolants) == len(data_fields):
            raise ValueError("The number of stochastic interpolants and data fields must be equal.")
        try:
            self._data_fields = [DataField[data_field.lower()] for data_field in data_fields]
        except AttributeError:
            raise ValueError(f"All data fields must be in {[d.name for d in DataField]}.")
        if not len(costs) == len(stochastic_interpolants):
            raise ValueError("The number of stochastic interpolants and costs must be equal.")
        if not all(cost >= 0.0 for cost in costs):
            raise ValueError("All cost factors must be non-negative.")
        if not abs(sum(costs) - 1.0) < 1e-10:
            raise ValueError("The sum of all cost factors must be approximately equal to 1.")
        if not integration_time_steps > 0:
            raise ValueError("The number of integration time steps must be positive.")
        self._stochastic_interpolants = stochastic_interpolants
        self._costs = costs
        self._integration_time_steps = integration_time_steps

    def _interpolate(self, t: torch.Tensor, x_0: Data, x_1: Data) -> Data:
        """
        Stochastically interpolate between the collection of points x_0 and x_1 from the collection of two distributions
        p_0 and p_1 at times t.

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
        assert torch.equal(x_0.ptr, x_1.ptr)
        assert torch.equal(x_0.n_atoms, x_1.n_atoms)
        n_atoms = x_0.n_atoms
        x_t = x_0.clone(*[data_field.name for data_field in self._data_fields])
        x_t_dict = x_t.to_dict()
        for stochastic_interpolant, data_field in zip(self._stochastic_interpolants, self._data_fields):
            assert data_field.name in x_0_dict
            assert data_field.name in x_1_dict
            assert data_field.name in x_t_dict
            reshaped_t = reshape_t(t, n_atoms, data_field)
            assert reshaped_t.shape == x_0_dict[data_field.name].shape
            interpolated_x_t, z = stochastic_interpolant.interpolate(reshaped_t, x_0_dict[data_field.name],
                                                                     x_1_dict[data_field.name], x_0.ptr)
            x_t_dict[data_field.name] = interpolated_x_t
            assert data_field.name + "_z" not in x_t_dict["property"]
            x_t_dict["property"][data_field.name + "_z"] = z
        return x_t

    def loss_from_interpolation(self, model_function: Callable[[Data, torch.tensor], Data], t: torch.Tensor, x_0: Data,
                                x_1: Data) -> torch.Tensor:
        """
        Compute the loss for the collection of stochastic interpolants between the collection of points x_0 and x_1 from
        a collection of distributions p_0 and p_1 at times t based on the collection of model predictions for the
        velocity fields b and the denoisers eta.

        This function expects that the velocity b and denoiser eta corresponding to the data field data_field are stored
        with the keys data_field_b and data_field_eta in the model prediction.

        The loss returned by every stochastic interpolant is scaled by the corresponding cost factor.

        :param model_function:
            Model function returning the collection of velocity fields b and the denoisers eta stored in a
            torch_geometric.data.Data object given the current collection of points x_t stored in a
            torch_geometric.data.Data object and times t.
        :type model_function: Callable[[torch_geometric.data.Data, torch.Tensor], torch_geometric.data.Data]
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
        x_t = self._interpolate(t, x_0, x_1)
        x_0_dict = x_0.to_dict()
        x_1_dict = x_1.to_dict()
        x_t_dict = x_t.to_dict()
        assert torch.equal(x_0.ptr, x_1.ptr)
        assert torch.equal(x_0.n_atoms, x_1.n_atoms)
        n_atoms = x_0.n_atoms
        total_loss = torch.tensor(0.0)
        for cost, stochastic_interpolant, data_field in zip(self._costs, self._stochastic_interpolants,
                                                            self._data_fields):
            b_data_field = data_field.name + "_b"
            eta_data_field = data_field.name + "_eta"
            assert data_field.name in x_0_dict
            assert data_field.name in x_1_dict
            assert data_field.name in x_t_dict
            reshaped_t = reshape_t(t, n_atoms, data_field)
            assert reshaped_t.shape == x_0_dict[data_field.name].shape
            assert reshaped_t.shape == x_1_dict[data_field.name].shape
            assert reshaped_t.shape == x_t_dict[data_field.name].shape

            x_t_clone = x_t.clone(*[data_field.name for data_field in self._data_fields])
            x_t_clone_dict = x_t_clone.to_dict()

            def model_prediction_fn(x):
                x_t_clone_dict[data_field.name] = x
                return model_function(x_t_clone, t)[b_data_field], model_function(x_t_clone, t)[eta_data_field]

            assert data_field.name + "_z" in x_t_dict["property"]
            total_loss += cost * stochastic_interpolant.loss(
                model_prediction_fn, reshaped_t, x_0_dict[data_field.name], x_1_dict[data_field.name],
                x_t_dict[data_field.name], x_t_dict["property"][data_field.name + "_z"], x_0.ptr)
        return total_loss

    def integrate(self, x_0: Data, model_function: Callable[[Data, torch.Tensor], Data]) -> Data:
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
            torch_geometric.data.Data object given the current collection of points x_t stored in a
            torch_geometric.data.Data object and times t.
        :type model_function: Callable[[torch_geometric.data.Data, torch.Tensor], torch_geometric.data.Data]

        :return:
            Collection of integrated points x_1 stored in a torch_geometric.data.Data object.
        :rtype: torch_geometric.data.Data
        """
        times = torch.linspace(SMALL_TIME, BIG_TIME, self._integration_time_steps)
        x_t = x_0.clone(*[data_field.name for data_field in self._data_fields])
        new_x_t = x_0.clone(*[data_field.name for data_field in self._data_fields])
        x_t_dict = x_t.to_dict()
        new_x_t_dict = new_x_t.to_dict()
        assert all(data_field.name in x_t_dict for data_field in self._data_fields)
        assert all(data_field.name in new_x_t_dict for data_field in self._data_fields)

        for t_index in range(1, len(times)):
            tspan = (float(times[t_index - 1]), float(times[t_index]))
            for stochastic_interpolant, data_field in zip(self._stochastic_interpolants, self._data_fields):
                b_data_field = data_field.name + "_b"
                eta_data_field = data_field.name + "_eta"
                x_int = x_t.clone(*[data_field.name for data_field in self._data_fields])
                x_int_dict = x_int.to_dict()

                def model_prediction_fn(t, x):
                    x_int_dict[data_field.name] = x
                    return model_function(x_int, t)[b_data_field], model_function(x_int, t)[eta_data_field]

                new_x_t_dict[data_field.name] = stochastic_interpolant.integrate(model_prediction_fn,
                                                                                 x_t_dict[data_field.name],
                                                                                 tspan)
            x_t = new_x_t.clone(*[data_field.name for data_field in self._data_fields])
        return x_t
