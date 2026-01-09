import numpy as np
import torch
from .abstracts import PositionDistribution


class MirrorPosition(PositionDistribution):
    """
    Base distribution that just mirrors the given positions of a single structure.

    The returned positions will have the same coordinate system (fractional or Cartesian) as the input positions.
    """

    def __init__(self) -> None:
        """Constructor of the MirrorPosition class."""
        super().__init__()

    def __call__(self, pos: torch.Tensor, pos_is_fractional: bool) -> tuple[np.ndarray, bool]:
        """
        Sample positions from the base distribution given the atomic positions of a single structure.

        This function just returns a clone of the input positions.

        :param pos:
            A tensor of shape (number_atoms, 3) containing the positions of the atoms in the structure.
        :type pos: torch.Tensor
        :param pos_is_fractional:
            Whether the input positions are in fractional coordinates.
        :type pos_is_fractional: bool

        :return:
            (A sample of positions from the base distribution in a tensor of shape (number_atoms, 3),
             Whether the sampled positions are in fractional coordinates.)
        :rtype: tuple[np.ndarray, bool]
        """
        return pos.detach().clone().cpu().numpy(), pos_is_fractional


class NormalPositionDistribution(PositionDistribution):
    """
    Base distribution that samples fractional positions from a normal distribution with given mean and standard
    deviation.

    :param loc:
        Mean of the normal distribution.
        Defaults to 0.0.
    :type loc: float
    :param scale:
        Standard deviation of the normal distribution.
        Defaults to 1.0.
    :type scale: float

    :raises ValueError:
        If the scale is less than or equal to 0.
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        """Constructor of the NormalPositionDistribution class."""
        super().__init__()
        if scale <= 0.0:
            raise ValueError("Scale must be greater than 0.")
        self._loc = loc
        self._scale = scale

    def __call__(self, pos: torch.Tensor, pos_is_fractional: bool) -> tuple[np.ndarray, bool]:
        """
        Sample fractional positions from the base distribution given the atomic positions of a single structure.

        :param pos:
            A tensor of shape (number_atoms, 3) containing the positions of the atoms in the structure.
        :type pos: torch.Tensor
        :param pos_is_fractional:
            Whether the input positions are in fractional coordinates.
        :type pos_is_fractional: bool

        :return:
            (A sample of positions from the base distribution in a tensor of shape (number_atoms, 3),
             Whether the sampled positions are in fractional coordinates.)
        :rtype: tuple[np.ndarray, bool]
        """
        return np.random.normal(loc=self._loc, scale=self._scale, size=pos.shape), True


class UniformPositionDistribution(PositionDistribution):
    """
    Position base distribution that samples fractional coordinates between 0 and 1 using a uniform distribution.
    """

    def __init__(self) -> None:
        """Constructor of the UniformPositionDistribution class."""
        super().__init__()

    def __call__(self, pos: torch.Tensor, pos_is_fractional: bool) -> tuple[np.ndarray, bool]:
        """
        Sample fractional positions from the base distribution given the atomic positions of a single structure.

        :param pos:
            A tensor of shape (number_atoms, 3) containing the positions of the atoms in the structure.
        :type pos: torch.Tensor
        :param pos_is_fractional:
            Whether the input positions are in fractional coordinates.
        :type pos_is_fractional: bool

        :return:
            (A sample of positions from the base distribution in a tensor of shape (number_atoms, 3),
             Whether the sampled positions are in fractional coordinates.)
        :rtype: tuple[np.ndarray, bool]
        """
        return np.random.uniform(low=0.0, high=1.0, size=pos.shape), True


class SobolSequence(PositionDistribution):
    """
    Position base distribution that samples fractional coordinates between 0 and 1 using a scrambled Sobol sequence.
    """

    def __init__(self) -> None:
        """Constructor of the SobolSequence class."""
        super().__init__()
        # Three dimensions for x, y, z.
        self._sampler = torch.quasirandom.SobolEngine(dimension=3, scramble=True)

    def __call__(self, pos: torch.Tensor, pos_is_fractional: bool) -> tuple[np.ndarray, bool]:
        """
        Sample fractional positions from the base distribution given the atomic positions of a single structure.

        :param pos:
            A tensor of shape (number_atoms, 3) containing the positions of the atoms in the structure.
        :type pos: torch.Tensor
        :param pos_is_fractional:
            Whether the input positions are in fractional coordinates.
        :type pos_is_fractional: bool

        :return:
            (A sample of positions from the base distribution in a tensor of shape (number_atoms, 3),
             Whether the sampled positions are in fractional coordinates.)
        :rtype: tuple[np.ndarray, bool]
        """
        assert len(pos.shape) == 2
        assert pos.shape[1] == 3
        return self._sampler.draw(pos.shape[0]).detach().cpu().numpy(), True
