from abc import ABC, abstractmethod
from typing import Optional


class Sampler(ABC):
    """
    This is an abstract sample class that defines the interface for all samplers.
    """

    def __init__(self):
        pass

    @abstractmethod
    def sample_p_0(self, x_1: Optional["OMGDataBatch"]) -> "OMGDataBatch":
        """
        Sample initial configuration.

        TODO: Document what exactly is stored OMGDataBatch.
        TODO: Do we even need the possibility for None?

        The sample will always have the format
        [species, positions, cell]
        """
        pass

