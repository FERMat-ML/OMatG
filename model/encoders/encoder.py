# abstract encoder implementation
import torch.nn as nn

class Encoder(nn.module):

    def __init__(self):
        # super

    def convert_inputs(self,):
        raise NotImplementedError

    def _forward(self,x):      
        raise NotImplementedError

    def convert_outputs(self,):
        raise NotImplementedError

    def forward(self,x):
        x = self.convert_inputs(x)
        x = self._forward(x)
        x = self.convert_outputs(x)
        return x
