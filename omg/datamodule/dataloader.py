import numpy as np
from torch_geometric.data import Data, Dataset
import torch
from .datamodule import Configuration
from ase.data import atomic_numbers


class OMGData(Data):
    """
    A Pytorch Geometric compatible graph representation of a configuration. When loaded
    into a class:`torch_geometric.data.DataLoader` the graphs of type OMGData
    will be automatically collated and batched.
    """

    def __init__(self):
        super().__init__()
        self.n_atoms = None
        self.species = None
        self.cell = None
        self.batch = None
        self.pos = None

    def __inc__(self, key: str, value: torch.Tensor, *args, **kwargs):
        if "index" in key or "face" in key:
            return self.n_atoms
        elif "batch" in key:
            # number of unique contributions
            return torch.unique(value).size(0)
        else:
            return 0

    def __cat_dim__(self, key: str, value: torch.Tensor, *args, **kwargs):
        if "index" in key or "face" in key:
            return 1
        else:
            return 0

    @classmethod
    def from_omg_configuration(cls, config: Configuration):
        graph = cls()
        n_atoms = torch.tensor(len(config.species))
        graph.n_atoms = n_atoms
        graph.batch = torch.zeros(n_atoms, dtype=torch.int64)
        graph.species = torch.tensor([atomic_numbers[z] for z in config.species], dtype=torch.int64)

        if isinstance(config.cell, np.ndarray):
            graph.cell = torch.from_numpy(config.cell)
        else:
            graph.cell = config.cell

        if isinstance(config.coords, np.ndarray):
            graph.pos = torch.from_numpy(config.coords)
        else:
            graph.pos = config.coords

        return graph


class OMGTorchDataset(Dataset):
    """
    This class is a wrapper for the :class:`torch_geometric.data.Dataset` class to enable
    the use of :class:`omg.datamodule.Dataset` as a data source for the graph based models.
    """

    def __init__(self, dataset: Dataset, transform=None):
        super().__init__("./", transform, None, None)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return OMGData.from_omg_configuration(self.dataset[idx])