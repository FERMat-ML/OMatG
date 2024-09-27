import torch.nn as nn

# Might not even need base class for head but it's probably safer to assume we do so we can add additional functionalties if needed
class Head(nn.Module):

    def __init__(self):
        pass

    def forward(self, x, t, prop=None):
        # need to define standard way of performing message passing

        raise NotImplementedError


        #return ( (type_b, type_eta), (coord_b, coord_eta), (lattice_b, lattice_eta) ) 
