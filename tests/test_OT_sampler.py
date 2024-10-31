import pytest
import torch
from omg.sampler.distance_metrics import *

def test_min_perm():
    '''
    Test minimum permutational distance.
    '''

    # Initialize
    x_0_init = torch.rand(size=(10,3))
    x_1_init = x_0_init.clone() + 1e-3 * torch.randn_like(x_0_init)
    shuffle_inds = torch.randperm(x_1_init.shape[0])

    # Compute ideal permutation
    x_0, x_1 = min_perm_dist(x_0_init, x_1_init[shuffle_inds])
    assert torch.equal(x_0_init, x_0)
    assert torch.equal(x_1_init, x_1)