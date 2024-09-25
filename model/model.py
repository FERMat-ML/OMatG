import torch.nn as nn

class Model(nn.module):
    def __init__(self, encoder, head):
        self.encoder = encoder
        self.head = head

    def forward(self,x, t, prop=None): # t should probably be embedded beforehand
        x = self.encoder(x)
        x = self.head(x, t, prop)
        return x

