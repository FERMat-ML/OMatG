from .sampler import Sampler
from ..datamodule.dataloader import OMGData
# import numpy as np
from ase.data import atomic_numbers

import torch
from torch_geometric.data import Batch


class SampleFromDataset(Sampler):
    """
    This is a sampler that generates random samples from the
    """
    def __init__(self, dataset, convert_to_fractional=True, niggli=True, batch_size=1):
        super().__init__()
        self.dataset = dataset
        self._frac = convert_to_fractional
        self.niggli = niggli
        self.batch_size = batch_size

    def sample_p_0(self):

        config = []
        for i in range(self.batch_size):
            sample_idx = torch.randint(0, len(self.dataset), (1,)).item()

            sample = self.dataset[sample_idx].to_dict()

            species = torch.asarray([atomic_numbers[s] for s in sample["species"]])
            pos = sample["coords"]
            cell = sample["cell"]

            config.append(OMGData.from_data(species, pos, cell, convert_to_fractional=self._frac, niggli=self.niggli))

        return Batch.from_data_list(config)
