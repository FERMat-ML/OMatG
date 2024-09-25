from .sampler import Sampler
from ..datamodule.dataloader import OMGData
import numpy as np
from ase.data import atomic_numbers


class SampleFromDataset(Sampler):
    """
    This is a sampler that generates random samples from the
    """
    def __init__(self, dataset, convert_to_fractional=True):
        super().__init__()
        self.dataset = dataset
        self._rng = np.random.default_rng()
        self._frac = convert_to_fractional

    def sample_p_0(self):

        sample_idx = self._rng.integers(0, len(self.dataset))
        sample_idx = int(sample_idx) # np.int64 to int

        sample = self.dataset[sample_idx].to_dict()

        species = np.array([atomic_numbers[s] for s in sample["species"]])
        pos = sample["coords"]
        cell = sample["cell"]

        return OMGData.from_data(species, pos, cell, convert_to_fractional=self._frac)
