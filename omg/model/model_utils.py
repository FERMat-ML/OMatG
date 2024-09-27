import torch.nn as nn
import torch
import math
from torch_geometric.data import Data

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# TODO: move to overall utils and make options accesible to OMG via CLI
def xyz_saver(data: Data):
    """
    Takes data that has been generated and saves it as xyz file
    """
    from ase import Atoms
    from ase.io import write
    import time
    batch_size = len(data.n_atoms)
    atoms = []
    for i in range(batch_size):
        lower, upper = data.ptr[i*1], data.ptr[(i*1)+1]
        atoms.append(Atoms(numbers=data.species[lower:upper], scaled_positions =data.pos[lower:upper, :], cell=data.cell[i, :, :]))
    write(f'{time.strftime("%Y%m%d-%H%M%S")}.xyz', atoms)

    

