import os
from typing import Dict, Any

import numpy as np
from torch_geometric.data import Data, Dataset
import torch
from .datamodule import Configuration
from ase.data import atomic_numbers
from torch_geometric.data.lightning import LightningDataset


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
        self.property = None

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
    def from_omg_configuration(cls, config: Configuration, convert_to_fractional=False):
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

        if config.property_dict is not None:
            graph.property = config.property_dict

        if convert_to_fractional:
            with torch.no_grad():
                graph.pos = torch.matmul(graph.pos, torch.inverse(graph.cell))

        return graph

    @classmethod
    def from_data(cls, species, pos, cell, convert_to_fractional=False):
        graph = cls()
        n_atoms = torch.tensor(len(species))
        graph.n_atoms = n_atoms
        graph.batch = torch.zeros(n_atoms, dtype=torch.int64)
        if isinstance(species[0], str):
            graph.species = torch.tensor([atomic_numbers[z] for z in species], dtype=torch.int64)
        else:
            graph.species = torch.tensor(species, dtype=torch.int64)

        if isinstance(cell, np.ndarray):
            graph.cell = torch.from_numpy(cell)
        else:
            graph.cell = cell

        if isinstance(pos, np.ndarray):
            graph.pos = torch.from_numpy(pos)
        else:
            graph.pos = pos

        if convert_to_fractional:
            with torch.no_grad():
                graph.pos = torch.matmul(graph.pos, torch.inverse(graph.cell))

        return graph


class OMGTorchDataset(Dataset):
    """
    This class is a wrapper for the :class:`torch_geometric.data.Dataset` class to enable
    the use of :class:`omg.datamodule.Dataset` as a data source for the graph based models.
    """

    def __init__(self, dataset: Dataset, transform=None, convert_to_fractional=False):
        super().__init__("./", transform, None, None)
        self.dataset = dataset
        self.convert_to_fractional = convert_to_fractional

    def __len__(self):
        return len(self.dataset)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return OMGData.from_omg_configuration(self.dataset[idx], convert_to_fractional=self.convert_to_fractional)


def get_lightning_datamodule(train_dataset: Dataset, val_dataset: Dataset, batch_size: int):
    """
    Create a PyTorch Lightning datamodule from the datasets

    Params:
        train_dataset:
        val_dataset:
        batch_size:

    Returns:
        A PyTorch Lightning datamodule
    """
    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))
    lightning_datamodule = LightningDataset(train_dataset, val_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers)
    return lightning_datamodule

