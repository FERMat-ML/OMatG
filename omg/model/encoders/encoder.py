from abc import ABC, abstractmethod
import torch.nn as nn

class Encoder(ABC, nn.Module):

    def __init__(self) -> None:
        super().__init__()
        pass
         
    @abstractmethod           # TODO: what will x be???
    def _convert_inputs(self, x , **kwargs):
        """
        Takes in OMG data standard and converts to whatever standard the encoder expects.
        Output should be a dict that can be passed as kwargs into _forward method
        """
        raise NotImplementedError

    @abstractmethod
    def _forward(self, **kwargs):      
        """
        Forward process for encoder
        """
        raise NotImplementedError

    @abstractmethod # TODO: What will be our expected output
    def _convert_outputs(self, x, **kwargs):
        """
        Takes output from encoder and returns expected OMG format.
        """
        raise NotImplementedError

                    # TODO: what will x be here
    def forward(self, x, t, prop=None, **kwargs):
        """
        Strings methods together
        """
        x = self._convert_inputs(x, **kwargs)
        x = self._forward(*x, t, prop, **kwargs)
        x = self._convert_outputs(x, **kwargs)
        return x
