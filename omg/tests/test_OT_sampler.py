import pytest
import torch
from omg.sampler.distance_metrics import *
from torch_geometric.data import Data

def test_min_perm_data():
    
    # Set up data dictionary
    n_atoms = 20
    batch_size = 5
    ptr = (torch.arange(batch_size) * n_atoms / batch_size).long()
    x_0_pos = torch.rand(size=(n_atoms, 3))
    x_1_pos = x_0_pos.clone()
    x_0_cell = torch.rand(size=(batch_size, 3, 3))
    x_1_cell = torch.rand(size=(batch_size, 3, 3))
    x_0_spec = torch.randint(size=(n_atoms,), low=1, high=10).long()
    x_1_spec = x_0_spec
    x_0 = Data(pos=x_0_pos, cell=x_0_cell, species=x_0_spec, ptr=ptr, n_atoms=n_atoms)
    x_1 = Data(pos=x_1_pos, cell=x_1_cell, species=x_1_spec, ptr=ptr, n_atoms=n_atoms)

    # Minimum Permutational Distance
    x_0, x_1 = min_perm_dist(x_0, x_1, periodic_distance)

    # Assert
    assert torch.all(x_0.species == x_1.species)
    assert torch.all(x_0.pos == x_1.pos)