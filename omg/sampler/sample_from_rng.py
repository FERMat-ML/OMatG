from typing import Union, Callable
from functools import partial

import torch

import numpy as np

from omg.globals import MAX_ATOM_NUM
from .sampler import Sampler
from ..datamodule.dataloader import OMGData
from torch_geometric.data import Batch
from .distributions import NDependentGamma, InformedLatticeDistribution

class SampleFromRNG(Sampler):
    """
    This is a sampler that generates random samples from the
    numpy random distributions. Option is to sample from a list of distributions,
    or a single distribution, and it will sample accordingly.

    If 3 samplers are provided, then
    species ~ distributions[0]
    pos ~ distributions[1]
    cell ~ distributions[2]

    The samplers should have the following api
    n_samples = sampler(size=n)

    functools.partial can be used to create a sampler with fixed arguments

    Example:
        import numpy as np

        rng = np.random.default_rng()
        species_rng = partial(rng.integers, low=1, high=118)
        pos_rng = partial(rng.uniform, low=0.0, high=1.0)
        cell_rng = partial(rng.lognormal, loc=1.0, scale=1.0)

        sampler = SampleFromDistributions([species_rng, pos_rng, cell_rng])

    Raises:
        RuntimeError: If the distributions > 3
    """

    def __init__(self, species_distribution = None,
                pos_distribution = None,
                cell_distribution = None,
                n_particle_sampler: Union[int, Callable] = 1,
                convert_to_fractional: bool = True,
                batch_size: int = 1):

        super().__init__()


        if species_distribution is None:
            # Sample uniformly in interval [1, MAX_ATOM_NUM].
            species_distribution = partial(np.random.randint, low=1, high=MAX_ATOM_NUM + 1)
        if pos_distribution is None:
            pos_distribution = partial(np.random.uniform, low=0.0, high=1.0)
        if cell_distribution is None:
            cell_distribution = partial(np.random.normal, loc=1.0, scale=1.0)
        self.distribution = [species_distribution, pos_distribution, cell_distribution]


        if isinstance(n_particle_sampler, int):
            def _constant_sampler():
                return n_particle_sampler

            self.n_particle_sampler = _constant_sampler
        else:
            self.n_particle_sampler = n_particle_sampler

        self._frac = convert_to_fractional
        self.batch_size = batch_size

    def sample_p_0(self, x1: "OMGDataBatch" = None) -> "OMGDataBatch":
        if x1 is not None:
            n = x1.n_atoms
            n = n.to(torch.int64)
        else:
            n = torch.zeros(self.batch_size, dtype=torch.int64)
            for i in range(self.batch_size):
                n[i] = torch.tensor(self.n_particle_sampler()).to(torch.int64)
            print(n, self.batch_size)

        configs = []
        for i in range(len(n)):
            species = self.distribution[0](size=n[i].item())

            pos = self.distribution[1](size=(n[i].item(), 3))
            pos = pos - np.floor(pos) # wrap to [0,1) fractional coordinates

            # TODO: maybe we don't need to restrict to symmetric->At least we aren't doing so for p1
            # TODO: make more generic in the future
            if isinstance(self.distribution[2], (NDependentGamma, InformedLatticeDistribution)):
                lattice_ = self.distribution[2](n[i].item())
            else:
                lattice_ = self.distribution[2](size=(3,3))
            #lattice_ = self.distribution[2](size=(3,3))
            cell = lattice_
            #cell = np.zeros((3,3))
            #cell[np.triu_indices(3)] = lattice_
            #cell = cell + cell.T # TODO: A27 equation looks redundant.

            # its already [0,1) fractional coordinates so no need to convert
            if not self._frac:
                pos = np.dot(pos, cell)

            configs.append(OMGData.from_data(species, pos, cell, convert_to_fractional=False))
        return Batch.from_data_list(configs)

    def add_n_particle_sampler(self, n_particle_sampler: Callable):
        self.n_particle_sampler = n_particle_sampler
        return
