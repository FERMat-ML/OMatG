import pytest
import torch
from torch_geometric.data import Data
from omg.sampler.distance_metrics import correct_for_min_perm_dist
from omg.si.corrector import IdentityCorrector, PeriodicBoundaryConditionsCorrector


@pytest.fixture(params=[IdentityCorrector(), PeriodicBoundaryConditionsCorrector(0.0, 1.0)])
def corrector(request):
    return request.param


def test_min_perm_data_trivial(corrector):
    # Test minimum permutation data for equal data.
    ptr = torch.tensor([0, 4, 9, 12, 15, 20]).long()
    n_atoms = [4, 5, 3, 3, 5]
    assert sum(n_atoms) == ptr[-1]
    sum_n_atoms = sum(n_atoms)
    assert len(ptr) - 1 == len(n_atoms)
    batch_size = len(n_atoms)

    x_0_pos = torch.rand(size=(sum_n_atoms, 3))
    x_1_pos = x_0_pos.clone()
    original_x_0_pos = x_0_pos.clone()
    original_x_1_pos = x_1_pos.clone()

    x_0_cell = torch.rand(size=(batch_size, 3, 3))
    x_1_cell = x_0_cell.clone()
    original_x_0_cell = x_0_cell.clone()
    original_x_1_cell = x_1_cell.clone()

    x_0_spec = torch.randint(size=(sum_n_atoms,), low=1, high=10).long()
    x_1_spec = x_0_spec.clone()
    original_x_0_spec = x_0_spec.clone()
    original_x_1_spec = x_1_spec.clone()

    x_0 = Data(pos=x_0_pos, cell=x_0_cell, species=x_0_spec, ptr=ptr, n_atoms=n_atoms)
    x_1 = Data(pos=x_1_pos, cell=x_1_cell, species=x_1_spec, ptr=ptr, n_atoms=n_atoms)

    correct_for_min_perm_dist(x_0, x_1, corrector)

    # x_1 should always be unchanged.
    assert torch.all(x_1.pos == original_x_1_pos)
    assert torch.all(x_1.cell == original_x_1_cell)
    assert torch.all(x_1.species == original_x_1_spec)

    # In this case, x_0 should be unchanged as well.
    assert torch.all(x_0.pos == original_x_0_pos)
    assert torch.all(x_0.cell == original_x_0_cell)
    assert torch.all(x_0.species == original_x_0_spec)

    # Therefore, x_0 and x_1 should be equal.
    assert torch.all(x_0.pos == x_1.pos)
    assert torch.all(x_0.cell == x_1.cell)
    assert torch.all(x_0.species == x_1.species)


def test_min_perm_data_permuted(corrector):
    # Test minimum permutation data for permuted data.
    ptr = torch.tensor([0, 4, 9, 12, 15, 20]).long()
    n_atoms = [4, 5, 3, 3, 5]
    assert sum(n_atoms) == ptr[-1]
    sum_n_atoms = sum(n_atoms)
    assert len(ptr) - 1 == len(n_atoms)
    batch_size = len(n_atoms)

    x_1_pos = torch.rand(size=(sum_n_atoms, 3))
    x_0_pos = x_1_pos.clone()
    original_x_1_pos = x_1_pos.clone()

    x_1_species = torch.randint(size=(sum_n_atoms,), low=1, high=10).long()
    x_0_species = x_1_species.clone()
    original_x_1_species = x_1_species.clone()

    # Cells shouldn't matter.
    x_1_cell = torch.rand(size=(batch_size, 3, 3))
    x_0_cell = torch.rand(size=(batch_size, 3, 3))
    original_x_1_cell = x_1_cell.clone()
    original_x_0_cell = x_0_cell.clone()

    # Permute x_0.
    for i in range(len(ptr) - 1):
        shuffled = torch.randperm(ptr[i + 1] - ptr[i])
        # Reshuffle if trivial permutation.
        while torch.all(shuffled == torch.arange(ptr[i + 1] - ptr[i])):
            shuffled = torch.randperm(ptr[i + 1] - ptr[i])
        x_0_pos[ptr[i]:ptr[i + 1]] = x_0_pos[ptr[i]:ptr[i + 1]][shuffled]
        x_0_species[ptr[i]:ptr[i + 1]] = x_0_species[ptr[i]:ptr[i + 1]][shuffled]
    original_x_0_pos = x_0_pos.clone()
    original_x_0_species = x_0_species.clone()
    assert not torch.all(original_x_0_pos == original_x_1_pos)
    assert not torch.all(original_x_0_species == original_x_1_species)

    x_0 = Data(pos=x_0_pos, cell=x_0_cell, species=x_0_species, ptr=ptr, n_atoms=n_atoms)
    x_1 = Data(pos=x_1_pos, cell=x_1_cell, species=x_1_species, ptr=ptr, n_atoms=n_atoms)
    assert not torch.all(x_0.pos == x_1.pos)

    correct_for_min_perm_dist(x_0, x_1, corrector)

    # x_1 should always be unchanged.
    assert torch.all(x_1.pos == original_x_1_pos)
    assert torch.all(x_1.cell == original_x_1_cell)
    assert torch.all(x_1.species == original_x_1_species)

    # In this case, x_0 should be changed (except for the cell).
    assert not torch.all(x_0.pos == original_x_0_pos)
    assert not torch.all(x_0.species == original_x_0_species)
    assert torch.all(x_0.cell == original_x_0_cell)

    # x_0 and x_1 should now be equal (except for the cell).
    assert torch.all(x_0.pos == x_1.pos)
    assert torch.all(x_0.species == x_1.species)


def test_min_perm_data_permuted_and_distorted(corrector):
    # Test minimum permutation data for permuted and slightly distorted data.
    ptr = torch.tensor([0, 4, 9, 12, 15, 20]).long()
    n_atoms = [4, 5, 3, 3, 5]
    assert sum(n_atoms) == ptr[-1]
    sum_n_atoms = sum(n_atoms)
    assert len(ptr) - 1 == len(n_atoms)
    batch_size = len(n_atoms)

    x_1_pos = torch.rand(size=(sum_n_atoms, 3))
    x_0_pos = x_1_pos.clone()
    original_x_1_pos = x_1_pos.clone()

    # Species shouldn't matter.
    x_1_species = torch.randint(size=(sum_n_atoms,), low=1, high=10).long()
    x_0_species = torch.randint(size=(sum_n_atoms,), low=1, high=10).long()
    original_x_1_species = x_1_species.clone()
    original_x_0_species = x_0_species.clone()

    # Cells shouldn't matter.
    x_1_cell = torch.rand(size=(batch_size, 3, 3))
    x_0_cell = torch.rand(size=(batch_size, 3, 3))
    original_x_1_cell = x_1_cell.clone()
    original_x_0_cell = x_0_cell.clone()

    # Permute x_0.
    for i in range(len(ptr) - 1):
        shuffled = torch.randperm(ptr[i + 1] - ptr[i])
        # Reshuffle if trivial permutation.
        while torch.all(shuffled == torch.arange(ptr[i + 1] - ptr[i])):
            shuffled = torch.randperm(ptr[i + 1] - ptr[i])
        x_0_pos[ptr[i]:ptr[i + 1]] = x_0_pos[ptr[i]:ptr[i + 1]][shuffled]
        x_0_species[ptr[i]:ptr[i + 1]] = x_0_species[ptr[i]:ptr[i + 1]][shuffled]

    # Distort x_0.
    x_0_pos += (torch.rand(size=(sum_n_atoms, 3)) * 1.0e-3 - 5.0e-4)  # Don't correct with respect to periodic boundaries.

    original_x_0_pos = x_0_pos.clone()
    assert not torch.all(original_x_0_pos == original_x_1_pos)

    x_0 = Data(pos=x_0_pos, cell=x_0_cell, species=x_0_species, ptr=ptr, n_atoms=n_atoms)
    x_1 = Data(pos=x_1_pos, cell=x_1_cell, species=x_1_species, ptr=ptr, n_atoms=n_atoms)
    assert not torch.allclose(x_0.pos, x_1.pos, atol=1.0e-3)

    correct_for_min_perm_dist(x_0, x_1, corrector)

    # x_1 should always be unchanged.
    assert torch.all(x_1.pos == original_x_1_pos)
    assert torch.all(x_1.cell == original_x_1_cell)
    assert torch.all(x_1.species == original_x_1_species)

    # In this case, x_0 should be changed (except for the cell).
    assert not torch.all(x_0.pos == original_x_0_pos)
    # Species should be the original unpermuted sequence.
    assert torch.all(x_0.species == original_x_0_species)
    assert torch.all(x_0.cell == original_x_0_cell)

    # x_0 and x_1 should now be approximately equal (except for the cell).
    assert not torch.all(x_0.pos == x_1.pos)
    assert not torch.all(x_0.pos == x_1.pos)
    assert torch.allclose(x_0.pos, x_1.pos, atol=1.0e-3)


def test_min_perm_data_permuted_pbc():
    # Test minimum permutation data for specific permuted data with periodic boundaries.
    ptr = torch.tensor([0, 2]).long()
    n_atoms = [2]
    assert sum(n_atoms) == ptr[-1]
    assert len(ptr) - 1 == len(n_atoms)
    batch_size = len(n_atoms)

    x_1_pos = torch.tensor([[0.99, 0.015, 0.98], [0.5, 0.4, 0.55]])
    x_0_pos = torch.tensor([[0.6, 0.38, 0.7], [0.02, 0.97, 0.01]])

    x_1_species = torch.tensor([1, 2]).long()
    x_0_species = torch.tensor([4, 7]).long()

    # Cells shouldn't matter here.
    x_1_cell = torch.rand(size=(batch_size, 3, 3))
    x_0_cell = torch.rand(size=(batch_size, 3, 3))
    original_x_1_cell = x_1_cell.clone()
    original_x_0_cell = x_0_cell.clone()

    x_0 = Data(pos=x_0_pos, cell=x_0_cell, species=x_0_species, ptr=ptr, n_atoms=n_atoms)
    x_1 = Data(pos=x_1_pos, cell=x_1_cell, species=x_1_species, ptr=ptr, n_atoms=n_atoms)

    correct_for_min_perm_dist(x_0, x_1, PeriodicBoundaryConditionsCorrector(0.0, 1.0))

    # x_1 should always be unchanged.
    assert torch.all(x_1.pos == torch.tensor([[0.99, 0.015, 0.98], [0.5, 0.4, 0.55]]))
    assert torch.all(x_1_species == torch.tensor([1, 2]).long())
    assert torch.all(x_1.cell == original_x_1_cell)

    # In this case, x_0 should be changed (except for the cell).
    assert torch.all(x_0.pos == torch.tensor([[0.02, 0.97, 0.01], [0.6, 0.38, 0.7]]))
    assert torch.all(x_0.species == torch.tensor([7, 4]).long())
    assert torch.all(x_0.cell == original_x_0_cell)


if __name__ == '__main__':
    pytest.main([__file__])
