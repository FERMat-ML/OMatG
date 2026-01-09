from abc import ABC, abstractmethod
from omg.datamodule import OMGData


class Sampler(ABC):
    """
    Abstract base class for all samplers that sample structures from the base distribution that will be connected to
    structures from the training data via stochastic interpolants.
    """

    def __init__(self):
        """Constructor for the Sampler class."""
        pass

    @abstractmethod
    def sample_p_0(self, x_1: OMGData) -> OMGData:
        """
        Sample the base distribution given a batch of structures from the training data.

        The returned OMGData represents a batch of structures sampled from the base distribution.

        The number of atoms in each sampled structure is the same as in the corresponding structure in the input OMGData
        x_1. For the positions, they choose the same representation (cartesian or fractional) as in the input OMGData
        x_1.

        :param x_1:
            A batch of structures from the training distribution.
        :type x_1: OMGData

        :return:
            A batch of structures sampled from the base distribution.
        :rtype: OMGData
        """
        raise NotImplementedError
