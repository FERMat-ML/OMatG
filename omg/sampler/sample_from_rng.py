from typing import List, Union, Callable
from functools import partial

from .sampler import Sampler
import numpy as np

MAX_ATOMIC_NUMBER = 92


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

    def __init__(self, distributions: Union[
        None, np.random.Generator, List[np.random.Generator]] = None,
                 n_particle_sampler: Union[int, Callable] = 1,
                 fractional_coordinates: bool = True):

        super().__init__()

        if isinstance(distributions, list):
            if len(distributions) > 3:
                raise RuntimeError("Cannot sample from more than 3 distributions")
            self.distribution = distributions

        elif isinstance(distributions, np.random.Generator):
            self.distribution = [distributions, distributions, distributions]

        elif distributions is None:
            rng = np.random.default_rng()
            _species_sampler = partial(rng.integers, low=1, high=MAX_ATOMIC_NUMBER)
            _pos_sampler = partial(rng.uniform, low=0.0, high=1.0)
            _cell_sampler = partial(rng.normal, loc=1.0, scale=1.0)
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

        self._frac = fractional_coordinates

    def sample_p_0(self):
        n = self.n_particle_sampler()
        species = self.distribution[0](size=n)

        pos = self.distribution[1](size=(n, 3))
        pos = pos - np.floor(pos) # wrap to [0,1) fractional coordinates

        lattice_ = self.distribution[2](size=6)
        cell = np.zeros((3,3))
        cell[np.triu_indices(3)] = lattice_
        cell = cell + cell.T # TODO: A27 equation looks redundant.

        if not self._frac:
            pos = np.dot(pos, cell)

        return species, pos, cell

    def add_n_particle_sampler(self, n_particle_sampler: Callable):
        self.n_particle_sampler = n_particle_sampler
        return
