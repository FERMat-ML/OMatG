import copy
import os
from typing import Any, Optional

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch
from ase.data import atomic_numbers
from torch_geometric.data.lightning import LightningDataset
import lightning as L

from .datamodule import Configuration, DataModule
from .utils import niggli_reduce_configuration, niggli_reduce_data


class OMGData(Data):
    """
    A Pytorch Geometric compatible graph representation of a configuration. When loaded
    into a class:`torch_geometric.data.DataLoader` the graphs of type OMGData
    will be automatically collated and batched.

    OMGData format:
    For a batch size of batch_size, the data format is as follows:
    - n_atoms: torch.Tensor of shape (batch_size, ) containing the number of atoms in each configuration
    - species: torch.Tensor of shape (sum(n_atoms), ) containing the atomic numbers of the atoms in the configurations
    - cell: torch.Tensor of shape (batch_size, 3, 3) containing the cell vectors of the configurations
    - batch: torch.Tensor of shape (sum(n_atoms), ) containing the index of the configuration to which each atom belongs
    - pos: torch.Tensor of shape (sum(n_atoms), 3) containing the atomic positions of the atoms in the configurations
    - property: dict containing the properties of the configurations
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
    def from_omg_configuration(cls, config: Configuration, convert_to_fractional=True, niggli=False):
        """
        Create a OMGData object from a :class:`omg.datamodule.Configuration` object.

        :param config:  :class:`omg.datamodule.Configuration` object to convert to OMGData
        :param convert_to_fractional: Whether to convert the atomic positions to fractional coordinates
                                    WARNING: This will always convert the atomic positions to fractional coordinates
                                    regardless of the current coordinate system. So, if the atomic positions are already
                                    in fractional coordinates, you need to be careful when setting this flag to True.
        :return:
            OMGData object.
        """
        graph = cls()
        if niggli:
            config = niggli_reduce_configuration(config)

        n_atoms = torch.tensor(len(config.species))
        graph.n_atoms = n_atoms
        graph.batch = torch.zeros(n_atoms, dtype=torch.int64)
        graph.species = torch.tensor([atomic_numbers[z] for z in config.species], dtype=torch.int64)

        assert isinstance(config.cell, torch.Tensor)
        graph.cell = config.cell

        assert isinstance(config.coords, torch.Tensor)
        graph.pos = config.coords

        if config.property_dict is not None:
            graph.property = config.property_dict

        if convert_to_fractional:
            with torch.no_grad():
                graph.pos = torch.remainder(torch.matmul(graph.pos, torch.inverse(graph.cell)), 1.0)

        graph.cell = graph.cell.unsqueeze(0)
        return graph

    @classmethod
    def from_data(cls, species, pos, cell, property_dict={}, convert_to_fractional=True, niggli=False):
        """
        Create a OMGData object from the atomic species, positions and cell vectors.

        :param species: Integer array containing the atomic numbers of the atoms
        :param pos: Array containing the atomic positions
        :param cell: Array containing the cell vectors
        :param convert_to_fractional:  Whether to convert the atomic positions to fractional coordinates
                                    WARNING: This will always convert the atomic positions to fractional coordinates
                                    regardless of the current coordinate system. So, if the atomic positions are already
                                    in fractional coordinates, you need to be careful when setting this flag to True.

        :return:
            OMGData object.
        """
        if niggli:
            cell, pos = niggli_reduce_data(species, pos, cell)

        graph = cls()
        n_atoms = torch.tensor(len(species))
        graph.n_atoms = n_atoms
        graph.batch = torch.zeros(n_atoms, dtype=torch.int64)
        if isinstance(species[0], str):
            graph.species = torch.asarray([atomic_numbers[z] for z in species], dtype=torch.int64)
        else:
            graph.species = torch.asarray(species, dtype=torch.int64)

        assert isinstance(cell, torch.Tensor)
        graph.cell = cell

        assert isinstance(pos, torch.Tensor)
        graph.pos = pos

        graph.property = {}
        if convert_to_fractional:
            with torch.no_grad():
                graph.pos = torch.remainder(torch.matmul(graph.pos, torch.inverse(graph.cell).to(graph.pos.dtype)), 1.0)

        graph.property = property_dict

        graph.cell = graph.cell.unsqueeze(0)
        return graph


class OMGTorchDataset(Dataset):
    """
    This class is a wrapper for the :class:`torch_geometric.data.Dataset` class to enable
    the use of :class:`omg.datamodule.Dataset` as a data source for the graph based models.
    """

    def __init__(self, dataset: DataModule, transform=None, convert_to_fractional=True, niggli=False):
        super().__init__("./", transform, None, None)
        self.dataset = dataset
        self.convert_to_fractional = convert_to_fractional
        self.niggli = niggli

    def __len__(self):
        return len(self.dataset)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return OMGData.from_omg_configuration(self.dataset[idx], convert_to_fractional=self.convert_to_fractional,
                                              niggli=self.niggli)


def get_lightning_datamodule(train_dataset: Dataset, val_dataset: Dataset, batch_size: int):
    """
    Create a PyTorch Lightning datamodule from the datasets. This is just provided for
    ease of use, and the user can create their own datamodule if needed.

    :param train_dataset: Training dataset
    :param val_dataset: Validation dataset
    :param batch_size: Batch size

    """
    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))
    lightning_datamodule = LightningDataset(train_dataset, val_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers)
    return lightning_datamodule

# TODO: Make len be the number of times we run through the generation pipeline
class NullDataset(Dataset):
    def __init__(self,):
        super().__init__()

    def get(self, idx: int) -> int:
        return idx

    def len(self,):
        return 1

class OMGDataModule(L.LightningDataModule):
    """
    Need to do this because LightningDataset doesn't directly subclass LightningDataModule
    """
    def __init__(self, train_dataset: OMGTorchDataset, val_dataset: OMGTorchDataset,
                 predict_dataset: Optional[OMGTorchDataset] = None, **kwargs) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.predict_dataset = predict_dataset
        if self.val_dataset is None:
            self.val_dataloader = None
        self.batch_size = kwargs.get('batch_size', 1)
        self.kwargs = kwargs

    def dataloader(self, dataset: Dataset, **kwargs: Any) -> DataLoader:
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        from torch.utils.data import IterableDataset

        shuffle = not isinstance(self.train_dataset, IterableDataset)
        shuffle &= self.kwargs.get('sampler', None) is None
        shuffle &= self.kwargs.get('batch_sampler', None) is None

        return self.dataloader(
            self.train_dataset,
            shuffle=shuffle,
            **self.kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)
        return self.dataloader(self.val_dataset, shuffle=False, **self.kwargs)

    def predict_dataloader(self) -> DataLoader:
        return self.dataloader(self.predict_dataset, shuffle=False, **self.kwargs)
