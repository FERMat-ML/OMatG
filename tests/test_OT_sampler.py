import pytest
import torch
from omg.sampler.distance_metrics import *
from torch_geometric.data import Data

def test_min_perm():
    '''
    Test minimum permutational distance.
    '''

    # Initialize
    x_0 = torch.rand(size=(10,3))
    x_1 = x_0.clone() + 1e-3 * torch.randn_like(x_0)
    shuffle_inds = torch.randperm(x_1.shape[0])
    x_1_shuff = x_1[shuffle_inds]

    # Compute ideal permutation
    row, col = min_perm_dist(x_0, x_1_shuff, euclidian_distance)
    assert torch.equal(x_0[row], x_0)
    assert torch.equal(x_1_shuff[col], x_1)

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
    for i in range(len(ptr)-1):

        # Shuffle
        assert torch.all(x_0.pos == x_1.pos)
        shuffled = torch.randperm(ptr[i+1] - ptr[i])
        x_1.pos[ptr[i]:ptr[i+1]] = x_1.pos[ptr[i]:ptr[i+1]][shuffled]
        x_1.species[ptr[i]:ptr[i+1]] = x_1.species[ptr[i]:ptr[i+1]][shuffled]

        # Minimum Perm
        row, col = min_perm_dist(x_0.pos[ptr[i]:ptr[i+1]], x_1.pos[ptr[i]:ptr[i+1]], periodic_distance)
        print(row)
        print(x_0.pos)
        print(col)
        print(x_1.pos)

        # Reassign
        x_0.pos[ptr[i]:ptr[i+1]] = x_0.pos[ptr[i]:ptr[i+1]][row]
        x_0.species[ptr[i]:ptr[i+1]] = x_0.species[ptr[i]:ptr[i+1]][row]
        x_1.pos[ptr[i]:ptr[i+1]] = x_1.pos[ptr[i]:ptr[i+1]][col]
        x_1.species[ptr[i]:ptr[i+1]] = x_1.species[ptr[i]:ptr[i+1]][col]

    # Assert
    assert torch.all(x_0.species == x_1.species)
    assert torch.all(x_0.pos == x_1.pos)