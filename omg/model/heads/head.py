from abc import abstractmethod, ABC
import torch.nn as nn


class Head(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, t, prop=None):
        # need to define standard way of performing message passing
        raise NotImplementedError

    @abstractmethod
    def enable_masked_species(self) -> None:
        """
        Enable a masked species (with token 0) in the head.
        """
        raise NotImplementedError
