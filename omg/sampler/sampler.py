from abc import ABC, abstractmethod


class Sampler(ABC):
    """
    This is an abstract sample class that defines the interface for all samplers.
    """

    def __init__(self):
        pass

    @abstractmethod
    def sample_p_0(self):
        """
        Sample initial configuration.

        The sample will always have the format
        [species, positions, cell]
        """
        pass

