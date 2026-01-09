from typing import Callable, List, Tuple, Union, Sequence
from tqdm import trange
import torch
from torch_geometric.data import Data
from omg.globals import SMALL_TIME, BIG_TIME
from omg.utils import reshape_t, DataField
from .abstracts import StochasticInterpolant


class StochasticInterpolants(object):
    """
    Collection of several stochastic interpolants between points x_0 and x_1 from two distributions p_0 and
    p_1 at times t for different coordinate types x (like atom species, fractional coordinates, and lattice vectors).

    Every stochastic interpolant is associated with a data field and a cost factor. The possible data fields are defined
    in the omg.utils.DataField enumeration. Data is transmitted using the torch_geometric.data.Data class which allows
    for accessing the data with a dictionary-like interface.

    The loss returned by every stochastic interpolant is scaled by the corresponding cost factor.

    :param stochastic_interpolants:
        Sequence of stochastic interpolants for the different coordinate types.
    :type stochastic_interpolants: Sequence[StochasticInterpolant]
    :param data_fields:
        Sequence of data fields for the different stochastic interpolants.
    :type data_fields: Sequence[str]
    :param integration_time_steps:
        Number of integration time steps for the integration of the collection of stochastic interpolants.

    :raises ValueError:
        If the number of stochastic interpolants and costs are not equal.
        If the number of stochastic interpolants and data fields are not equal.
        If the number of integration time steps is not positive.
    """

    def __init__(self, stochastic_interpolants: Sequence[StochasticInterpolant], data_fields: Sequence[str],
                 integration_time_steps: int, enable_progress_bar: bool = True) -> None:
        """Constructor of the StochasticInterpolants class."""
        super().__init__()
        if not len(stochastic_interpolants) == len(data_fields):
            raise ValueError("The number of stochastic interpolants and data fields must be equal.")
        try:
            self._data_fields = [DataField[data_field.lower()] for data_field in data_fields]
        except AttributeError:
            raise ValueError(f"All data fields must be in {[d.name for d in DataField]}.")

        if not integration_time_steps > 0:
            raise ValueError("The number of integration time steps must be positive.")
        self._stochastic_interpolants = stochastic_interpolants
        self._integration_time_steps = integration_time_steps
        self._enable_progress_bar = enable_progress_bar

    def __len__(self) -> int:
        """
        Return the number of stochastic interpolants handled by this class.

        :return:
            Number of stochastic interpolants.
        :rtype: int
        """
        return len(self._stochastic_interpolants)

    def loss_keys(self) -> List[str]:
        """
        Return the keys of the losses returned by this class.

        The keys of the losses are constructed by concatenating the data field name and the loss key of the
        corresponding stochastic interpolant.

        :return:
            Keys of the losses.
        :rtype: List[str]
        """
        loss_keys = []
        for df, si in zip(self._data_fields, self._stochastic_interpolants):
            for key in si.loss_keys():
                full_key = f"{df.name}_{key}"
                if full_key in loss_keys:
                    raise ValueError(f"Key {full_key} is already used as a loss key.")
                loss_keys.append(full_key)
        return loss_keys

    def _interpolate(self, t: torch.Tensor, x_0: Data, x_1: Data) -> tuple[Data, Data]:
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
            Collection of stochastically interpolated points x_t stored in a torch_geometric.data.Data object,
            and the collection of z values stored in a torch_geometric.data.Data object.
        :rtype: tuple[torch_geometric.data.Data, torch_geometric.data.Data]
        """
        x_0_dict = x_0.to_dict()
        x_1_dict = x_1.to_dict()
        assert torch.equal(x_0.batch, x_1.batch)
        assert torch.equal(x_0.n_atoms, x_1.n_atoms)
        n_atoms = x_0.n_atoms
        x_t = x_0.clone()
        x_t_dict = x_t.to_dict()
        z_data = {}
        for stochastic_interpolant, data_field in zip(self._stochastic_interpolants, self._data_fields):
            assert data_field.name in x_0_dict
            assert data_field.name in x_1_dict
            assert data_field.name in x_t_dict
            reshaped_t = reshape_t(t, n_atoms, data_field)
            assert reshaped_t.shape == x_0_dict[data_field.name].shape
            # Cell data requires different batch indices.
            interpolated_x_t, z = stochastic_interpolant.interpolate(
                reshaped_t, x_0_dict[data_field.name], x_1_dict[data_field.name],
                x_0.batch if data_field != DataField.cell else torch.arange(len(x_0.n_atoms)))
            # Assignment does not update x_t.
            x_t_dict[data_field.name].copy_(interpolated_x_t)
            assert data_field.name not in z_data
            z_data[data_field.name] = z
        return x_t, Data.from_dict(z_data)

    def losses(self, model_function: Callable[[Data, torch.tensor], Data], t: torch.Tensor, x_0: Data,
               x_1: Data) -> dict[str, torch.Tensor]:
        """
        Compute the losses for the collection of stochastic interpolants between the collection of points x_0 and x_1
        from a collection of distributions p_0 and p_1 at times t based on the collection of model predictions for the
        velocity fields b and the denoisers eta.

        This function expects that the velocity b and denoiser eta corresponding to the data field data_field are stored
        with the keys data_field_b and data_field_eta in the model prediction.

        The losses are returned as a dictionary with the data field names as keys and the corresponding losses as
        values.

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
            The losses for the collection of stochastic interpolants.
        :rtype: dict[str, torch.Tensor]
        """
        # Interpolate everything first so that we can pass all interpolated to the model function.
        x_t, z = self._interpolate(t, x_0, x_1)

        x_0_dict = x_0.to_dict()
        x_1_dict = x_1.to_dict()
        x_t_dict = x_t.to_dict()
        z_dict = z.to_dict()
        assert torch.equal(x_0.batch, x_1.batch)
        assert torch.equal(x_0.n_atoms, x_1.n_atoms)
        n_atoms = x_0.n_atoms
        losses = {}
        for stochastic_interpolant, data_field in zip(self._stochastic_interpolants, self._data_fields):
            b_data_field = data_field.name + "_b"
            eta_data_field = data_field.name + "_eta"
            assert data_field.name in x_0_dict
            assert data_field.name in x_1_dict
            assert data_field.name in x_t_dict
            reshaped_t = reshape_t(t, n_atoms, data_field)
            assert reshaped_t.shape == x_0_dict[data_field.name].shape
            assert reshaped_t.shape == x_1_dict[data_field.name].shape
            assert reshaped_t.shape == x_t_dict[data_field.name].shape

            def model_prediction_fn(x):
                # Clone x_t inside the function so that this function can be called several time.
                # If cloned outside, torch will complain that one of the variables needed for gradient computation has
                # been modified by an inplace operation.
                x_t_clone = x_t.clone()
                x_t_clone_dict = x_t_clone.to_dict()
                x_t_clone_dict[data_field.name].copy_(x)
                # TODO: Cache return of model function.
                model_result = model_function(x_t_clone, t)
                return model_result[b_data_field], model_result[eta_data_field]

            assert data_field.name in z_dict
            assert "loss_" + data_field.name not in losses
            # Cell data requires different batch indices.
            l = stochastic_interpolant.loss(
                model_prediction_fn, reshaped_t, x_0_dict[data_field.name], x_1_dict[data_field.name],
                x_t_dict[data_field.name], z[data_field.name],
                x_0.batch if data_field != DataField.cell else torch.arange(len(x_0.n_atoms)))
            for l_key, l_value in l.items():
                assert l_key not in losses
                losses[f"{data_field.name}_{l_key}"] = l_value
        return losses

    def integrate(self, x_0: Data, model_function: Callable[[Data, torch.Tensor], Data],
                  save_intermediate: bool = False) -> Union[Data, Tuple[Data, List[Data]]]:
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
        :save_intermediate:
            If True, the intermediate points of the integration are saved and returned.
        :type save_intermediate: bool

        :return:
            Collection of integrated points x_1 stored in a torch_geometric.data.Data object.
            If save_intermediate is True, furthermore a list of the intermediate points in Data objects is returned.
        :rtype: torch_geometric.data.Data
        """
        times = torch.linspace(SMALL_TIME, BIG_TIME, self._integration_time_steps, device=x_0.pos.device)
        x_t = x_0.clone(*[data_field.name for data_field in self._data_fields])
        new_x_t = x_0.clone(*[data_field.name for data_field in self._data_fields])
        x_t_dict = x_t.to_dict()
        new_x_t_dict = new_x_t.to_dict()
        assert all(data_field.name in x_t_dict for data_field in self._data_fields)
        assert all(data_field.name in new_x_t_dict for data_field in self._data_fields)

        if save_intermediate:
            inter_list = [x_t]
        else:
            inter_list = None
        for t_index in trange(1, len(times), desc='Integrating', disable=not self._enable_progress_bar):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            for stochastic_interpolant, data_field in zip(self._stochastic_interpolants, self._data_fields):
                b_data_field = data_field.name + "_b"
                eta_data_field = data_field.name + "_eta"
                x_int = x_t.clone(*[data_field.name for data_field in self._data_fields])
                x_int_dict = x_int.to_dict()

                def model_prediction_fn(time, x):
                    # The model expects the time to be repeated for every element in the batch.
                    # The time argument, however, is just a zero-dimensional tensor.
                    time = time.repeat(len(x_int_dict['n_atoms']))
                    x_int_dict[data_field.name].copy_(x)
                    model_result = model_function(x_int, time)
                    return model_result[b_data_field], model_result[eta_data_field]

                # Do not use x_int_dict[data_field.name] here because it will be implicitly updated in the
                # model_prediction_fn, which leads to unpredictable bugs.
                # Cell data requires different batch indices.
                new_x_t_dict[data_field.name].copy_(stochastic_interpolant.integrate(
                    model_prediction_fn, x_t_dict[data_field.name], t, dt,
                    x_0.batch if data_field != DataField.cell else torch.arange(len(x_0.n_atoms))))

            x_t = new_x_t.clone(*[data_field.name for data_field in self._data_fields])
            x_t_dict = x_t.to_dict()
            if save_intermediate:
                inter_list.append(x_t)
        if save_intermediate:
            return x_t, inter_list
        else:
            return x_t

    def get_stochastic_interpolant(self, data_field: str) -> StochasticInterpolant:
        """
        Return the stochastic interpolant associated with the data field.

        :param data_field:
            Data field for which the stochastic interpolant is requested.
        :type data_field: str

        :return:
            Stochastic interpolant associated with the data field.
        :rtype: StochasticInterpolant
        """
        try:
            df = DataField[data_field.lower()]
        except AttributeError:
            raise ValueError(f"Data field must be in {[d.name for d in DataField]}.")
        index = self._data_fields.index(df)
        return self._stochastic_interpolants[index]
