import numpy as np
import torch
from omg.globals import MAX_ATOM_NUM
from .abstracts import SpeciesDistribution


class MirrorSpecies(SpeciesDistribution):
    """
    Base distribution that just mirrors the given species of a single structure.
    """

    def __init__(self) -> None:
        """Constructor of the MirrorSpecies class."""
        super().__init__()

    def __call__(self, species: torch.Tensor) -> np.ndarray:
        """
        Sample species from the base distribution given the species of a single structure.

        This function just returns a clone of the input species.

        :param species:
            The atomic numbers of all atoms in the structure in a tensor of shape (number_atoms, ).
        :type species: torch.Tensor

        :return:
            A sample of species from the base distribution in a tensor of shape (number_atoms, ).
        :rtype: np.ndarray
        """
        return species.detach().clone().cpu().numpy()


class UniformSpeciesDistribution(SpeciesDistribution):
    """
    Base distribution for species that samples integers uniformly from a given range of atomic numbers.

    :param low:
        Lower bound (inclusive) of the uniform distribution.
        Defaults to 1.
    :type low: int
    :param high:
        Upper bound (exclusive) of the uniform distribution.
        Defaults to MAX_ATOM_NUM + 1.
    :type high: int

    :raises ValueError:
        If high is less than or equal to low.
    """

    def __init__(self, low: int = 1, high: int = MAX_ATOM_NUM + 1) -> None:
        """Constructor of the UniformSpeciesDistribution class."""
        super().__init__()
        if high <= low:
            raise ValueError("High must be greater than low.")
        self._low = low
        self._high = high

    def __call__(self, species: torch.Tensor) -> np.ndarray:
        """
        Sample species from the base distribution given the species of a single structure.

        This method returns a tensor of the same length as the input species tensor,
        where each entry is sampled uniformly from the specified range on initialization.

        :param species:
            The atomic numbers of all atoms in the structure in a tensor of shape (number_atoms, ).
        :type species: torch.Tensor

        :return:
            A sample of species from the base distribution in a tensor of shape (number_atoms, ).
        :rtype: np.ndarray
        """
        return np.random.randint(low=self._low, high=self._high, size=len(species), dtype=np.int64)


class MaskSpeciesDistribution(SpeciesDistribution):
    """
    Base distribution for species that always returns the same masking token for all atoms.

    :param token:
        The masking token to use as the masking atomic number for all atoms.
        Default is 0.
    :type token: int
    """

    def __init__(self, token: int = 0) -> None:
        """Constructor of the MaskSpeciesDistribution class."""
        super().__init__()
        self._token = token

    def __call__(self, species: torch.Tensor) -> np.ndarray:
        """
        Sample species from the base distribution given the species of a single structure.

        This method returns a tensor of the same length as the input species tensor,
        where all entries are set to the masking token.

        :param species:
            The atomic numbers of all atoms in the structure in a tensor of shape (number_atoms, ).
        :type species: torch.Tensor

        :return:
            A sample of species from the base distribution in a tensor of shape (number_atoms, ).
        :rtype: np.ndarray
        """
        # noinspection PyTypeChecker
        return np.ones(len(species), dtype=np.int64) * self._token
