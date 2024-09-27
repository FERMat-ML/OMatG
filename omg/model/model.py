import torch.nn as nn

class Model(nn.Module):
    def __init__(self, encoder, head, time_embedder, prop_embedder=None, property_keys=None):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.time_embedder = time_embedder
        self.prop_embedder = prop_embedder
        self.property_keys = property_keys

    def forward(self, x, t, prop=None, **kwargs): # prop=None is redundent here
        t = self.time_embedder(t)
        
        property_key = self.property_keys # currently tested on one property key
        if property_key is not None:
            prop = x.property[property_key]
            prop = prop.double()
        else:
            prop = None # or prop = None is redundent here

        if self.prop_embedder is not None:
            if prop is None:
                raise Exception("You have specified a property embedder but no property value is specified")
            else:
                prop = self.prop_embedder(prop)
        x = self.encoder(x, t, prop, **kwargs) 
        x = self.head(x, t, prop)
        return x

