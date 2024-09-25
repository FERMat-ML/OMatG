import torch.nn as nn

class Model(nn.Module):
    def __init__(self, encoder, head, time_embedder, prop_embedder=None):
        self.encoder = encoder
        self.head = head
        self.time_embedder = time_embedder
        self.prop_embedder = prop_embedder

    def forward(self, x, t, prop=None, **kwargs): 
        t = self.time_embedder(t)
        if self.prop_embedder is not None:
            if prop is None:
                raise Exception("You have specified a property embedder but no property value is specified")
            else:
                prop = self.prop_embedder(prop)
        x = self.encoder(x, t, prop, **kwargs) 
        x = self.head(x, t, prop)
        return x

