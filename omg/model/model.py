import torch.nn as nn

class Model(nn.Module):
    def __init__(self, encoder, head):
        self.encoder = encoder
        self.head = head

    def forward(self,x, t, prop=None, **kwargs): # t should probably be embedded beforehand
        x = self.encoder(x, **kwargs) 
        x = self.head(x, t, prop)
        return x

