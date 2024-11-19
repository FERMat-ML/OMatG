import torch
from torch_scatter import scatter_mean
from .abstracts import Corrector


class IdentityCorrector(Corrector):
    """
    Corrector that does nothing.
    """

    def __init__(self):
        """Construct identity corrector."""
        super().__init__()

    def correct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Correct the input x.

        :param x:
            Input to correct.
        :type x: torch.Tensor

        :return:
            Corrected input.
        :rtype: torch.Tensor
        """
        return x

    def unwrap(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """
        Correct the input x_1 based on the reference input x_0.

        This method just returns x_1.

        :param x_0:
            Reference input.
        :type x_0: torch.Tensor
        :param x_1:
            Input to correct.
        :type x_1: torch.Tensor

        :return:
            Unwrapped x_1 value.
        :rtype: torch.Tensor
        """
        return x_1.clone()

    def compute_center_of_mass(self, x: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute the center of masses of the configurations in the batch with respect to the correction applied by this
        corrector. This method just computes the standard center of mass.

        The first dimension of the input tensor x is the batch dimension. Here, different elements in the batch can
        belong to different configurations. The one-dimensional tensor batch_indices contains the indices of the
        configurations for the first dimension of the input tensor x. The center of mass for every configuration should
        be computed separately.

        The dimensions of the returned tensor are the same as the dimensions of the input tensor x. This means that the
        center of mass for every configuration is replicated for every element in the configuration.

        :param x:
            Input whose center of masses will be returned.
        :type x: torch.Tensor
        :param batch_indices:
            The batch indices for the input tensor x.
        :type batch_indices: torch.Tensor

        :return:
            Center of masses.
        :rtype: torch.Tensor
        """
        x_com = scatter_mean(x, batch_indices, dim=0)
        return torch.index_select(x_com, 0, batch_indices)


class PeriodicBoundaryConditionsCorrector(Corrector):
    """
    Corrector function that wraps back coordinates to the interval [min, max] with periodic boundary conditions.

    :param min_value:
        Minimum value of the interval.
    :type min_value: float
    :param max_value:
        Maximum value of the interval.
    :type max_value: float

    :raises ValueError:
        If the minimum value is greater than the maximum value.
    """

    def __init__(self, min_value: float, max_value: float) -> None:
        """
        Construct corrector function.
        """
        super().__init__()
        self._min_value = min_value
        self._max_value = max_value
        if self._min_value >= self._max_value:
            raise ValueError("Minimum value must be less than maximum value.")

    def correct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Correct the input x.

        :param x:
            Input to correct.
        :type x: torch.Tensor

        :return:
            Corrected input.
        :rtype: torch.Tensor
        """
        return torch.remainder(x - self._min_value, self._max_value - self._min_value) + self._min_value
    
    def unwrap(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """
        Correct the input x_1 based on the reference input x_0.

        This method returns the image of x_1 closest to x_0 in periodic boundary conditions.

        :param x_0:
            Reference input.
        :type x_0: torch.Tensor
        :param x_1:
            Input to correct.
        :type x_1: torch.Tensor

        :return:
            Unwrapped x_1 value.
        :rtype: torch.Tensor
        """
        separation_vector = x_1 - x_0
        length_over_two = (self._max_value - self._min_value) / 2.0
        # Shortest separation lies in interval [-L/2, L/2].
        shortest_separation_vector = torch.remainder(separation_vector + length_over_two,
                                                     self._max_value - self._min_value) - length_over_two
        return x_0 + shortest_separation_vector

    def compute_center_of_mass(self, x: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute the center of masses of the configurations in the batch with respect to periodic boundary conditions.

        The first dimension of the input tensor x is the batch dimension. Here, different elements in the batch can
        belong to different configurations. The one-dimensional tensor batch_indices contains the indices of the
        configurations for the first dimension of the input tensor x. The center of mass for every configuration should
        be computed separately.

        The dimensions of the returned tensor are the same as the dimensions of the input tensor x. This means that the
        center of mass for every configuration is replicated for every element in the configuration.

        This function uses the approach detailed in https://en.wikipedia.org/wiki/Center_of_mass.

        :param x:
            Input whose center of masses will be returned.
        :type x: torch.Tensor
        :param batch_indices:
            The batch indices for the input tensor x.
        :type batch_indices: torch.Tensor

        :return:
            Center of masses.
        :rtype: torch.Tensor
        """
        assert torch.all(x >= self._min_value)
        assert torch.all(x < self._max_value)
        thetas = (x - self._min_value) / (self._max_value - self._min_value) * 2.0 * torch.pi
        x_coordinates = torch.cos(thetas)
        y_coordinates = torch.sin(thetas)
        x_mean = scatter_mean(x_coordinates, batch_indices, dim=0)
        y_mean = scatter_mean(y_coordinates, batch_indices, dim=0)
        mean_theta = torch.atan2(-y_mean, -x_mean) + torch.pi  # Get angle with respect to (1, 0) in [0, 2pi).
        x_com = mean_theta / (2.0 * torch.pi) * (self._max_value - self._min_value) + self._min_value
        return torch.index_select(x_com, 0, batch_indices)
