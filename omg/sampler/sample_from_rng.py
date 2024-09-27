from typing import List, Union, Callable
from functools import partial

import torch

# import numpy as np

from omg.globals import MAX_ATOM_NUM
from .sampler import Sampler
from ..datamodule.dataloader import OMGData
from torch_geometric.data import Batch


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
        import torch

        rng = torch.Generator()
        species_rng = lambda size: torch.randint(1, 92, size=size, generator=rng)
        pos_rng = lambda size: torch.rand(size=size, generator=rng)
        cell_rng = lambda size: torch.randn(size=size, generator=rng)

        sampler = SampleFromDistributions([species_rng, pos_rng, cell_rng])

    Raises:
        RuntimeError: If the distributions > 3
    """

    def __init__(self, distributions: Union[List[Callable], torch.Generator] = None,
                 n_particle_sampler: Union[int, Callable] = 1,
                 convert_to_fractional: bool = True,
                 batch_size: int = 1):

        super().__init__()

        if isinstance(distributions, list):
            if len(distributions) > 3:
                raise RuntimeError("Cannot sample from more than 3 distributions")
            self.distribution = distributions

        elif isinstance(distributions, Callable):
            self.distribution = [distributions, distributions, distributions]

        elif distributions is None:
            rng = torch.Generator()
            _species_sampler = lambda size: torch.randint(1, MAX_ATOM_NUM, size=size, generator=rng)
            _pos_sampler = lambda size: torch.rand(size=size, generator=rng)
            _cell_sampler = lambda size: torch.randn(size=size, generator=rng)
            self.distribution = [_species_sampler, _pos_sampler, _cell_sampler]

        else:
            raise RuntimeError(
                "Distributions must be a numpy random generator or a list of numpy random generators or None")

        if isinstance(n_particle_sampler, int):
            def _constant_sampler():
                return n_particle_sampler

            self.n_particle_sampler = _constant_sampler
        else:
            self.n_particle_sampler = n_particle_sampler

        self._frac = convert_to_fractional
        self.batch_size = batch_size

    def sample_p_0(self, x1: "OMGDataBatch" = None):
        if x1 is not None:
            n = x1.n_atoms
            n = n.to(torch.int64)
        else:
            n = torch.zeros(self.batch_size, dtype=torch.int64)
            for i in range(self.batch_size):
                n[i] = torch.tensor(self.n_particle_sampler()).to(torch.int64)

        configs = []
        for i in range(len(n)):
            species = self.distribution[0](size=n[i])

            pos = self.distribution[1](size=(n[i], 3))
            # pos = pos - np.floor(pos) # wrap to [0,1) fractional coordinates

            lattice_ = self.distribution[2](size=6)
            cell = torch.zeros((3,3))
            cell[torch.triu_indices(3)] = lattice_
            cell = cell + cell.T # TODO: A27 equation looks redundant.

            # its already [0,1) fractional coordinates so no need to convert
            if not self._frac:
                pos = torch.dot(pos, cell)

            configs.append(OMGData.from_data(species, pos, cell, convert_to_fractional=False))

        return Batch.from_data_list(configs)

    def add_n_particle_sampler(self, n_particle_sampler: Callable):
        self.n_particle_sampler = n_particle_sampler
        return
