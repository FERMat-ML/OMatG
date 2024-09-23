import torch.nn as nn

class Model(nn.module):
    def __init__(self, encoder, head):
        self.encoder = encoder
        self.head = head

    def forward(self,x, t): # t should probably be embedded
        x = self.encoder(x)
        x = self.head(x, t)
        return x

        
