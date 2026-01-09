from enum import Enum, auto
from pathlib import Path
from typing import Sequence, Union
from ase import Atoms
from ase.io import read, write
import torch
from torch_geometric.data import Data
import numpy as np
from omg.datamodule import OMGData


class DataField(Enum):
    """
    Enum for the different data fields in the omg.datamodule.dataloader.OMGData class relevant for stochastic
    interpolants.
    """
    pos = auto()
    """Atomic positions."""
    cell = auto()
    """Cell vectors."""
    species = auto()
    """Atomic numbers."""


def reshape_t(t: torch.Tensor, n_atoms: torch.Tensor, data_field: DataField) -> torch.Tensor:
    """
    Reshape the given tensor of times for every configuration of the batch so that it can be used for the given data field.  
    For a batch size of batch_size, the data format for the different data fields is as follows:
    - species: torch.Tensor of shape (sum(n_atoms), ) containing the atomic numbers of the atoms in the configurations
    - cell: torch.Tensor of shape (batch_size, 3, 3) containing the cell vectors of the configurations
    - pos: torch.Tensor of shape (sum(n_atoms), 3) containing the atomic positions of the atoms in the configurations

    The returned tensor will have the same shape as the tensor of the given data field, and the correct time for every
    element of the data field tensor.
 
    :param t:
        Tensor of times for the configurations in the batch.
    :type t: torch.Tensor
    :param n_atoms:
        Tensor of the number of atoms in each configuration in the batch.
    :type n_atoms: torch.Tensor
    :param data_field:
        Data field for which the tensor of times should be reshaped.
    :type data_field: DataField
   
    :return:
        Tensor of times for the given data field.
        :rtype: torch.Tensor
    """
    assert len(t.shape) == len(n_atoms.shape) == 1
    t_per_atom = t.repeat_interleave(n_atoms)
    sum_n_atoms = int(n_atoms.sum())
    batch_size = len(t)
    if data_field == DataField.pos:
        return t_per_atom.repeat_interleave(3).reshape(sum_n_atoms, 3)
    elif data_field == DataField.cell:
        return t.repeat_interleave(3 * 3).reshape(batch_size, 3, 3)
    else:
        assert data_field == DataField.species
        return t_per_atom


def xyz_saver(data: Union[OMGData, list[OMGData]], filename: Path) -> None:
    """
    Save structures from OMGData instances to an xyz file.

    :param data:
        OMGData or list of OMGData instances to save.
    :type data: Union[OMGData, list[OMGData]]
    :param filename:
        Path to the xyz file where the structures will be saved.
    :type filename: Path
    """
    if not filename.suffix == ".xyz":
        raise ValueError("The filename must have the suffix '.xyz'.")
    if not isinstance(data, list):
        data = [data]
    atoms = []
    for d in data:
        batch_size = len(d.n_atoms)
        for i in range(batch_size):
            lower, upper = d.ptr[i * 1], d.ptr[(i * 1) + 1]
            if d.pos_is_fractional[i]:
                atoms.append(Atoms(numbers=d.species[lower:upper], scaled_positions=d.pos[lower:upper, :],
                                   cell=d.cell[i, :, :], pbc=True))
            else:
                atoms.append(Atoms(numbers=d.species[lower:upper], positions=d.pos[lower:upper, :],
                                   cell=d.cell[i, :, :], pbc=True))
    write(filename, atoms, append=True)


def xyz_reader(filename: Path) -> list[Atoms]:
    """
    Read structures from an xyz file and return a list of Atoms instances.

    :param filename:
        Path to the xyz file to read.
    :type filename: Path

    :return:
        List of Atoms instances read from the xyz file.
    :rtype: list[Atoms]
    """
    if not filename.suffix == ".xyz":
        raise ValueError("The filename must have the suffix '.xyz'.")
    # Read all atoms from the file by using index=":".
    all_configs = read(filename, index=":", format='extxyz')
    return all_configs


def convert_ase_atoms_to_data(all_configs: Sequence[Atoms]) -> Data:
    """
    Convert a list of ASE Atoms objects to a PyTorch Geometric Data object similar to OMGData.

    :param all_configs:
        List of ASE Atoms objects to convert.
    :type all_configs: Sequence[Atoms]

    :return:
        PyTorch Geometric Data object containing the batched configurations.
    :rtype: Data
    """
    batch_size = len(all_configs)
    n_atoms = torch.tensor([len(config) for config in all_configs], dtype=torch.int64)
    sum_n_atoms = n_atoms.sum()
    batch = torch.repeat_interleave(torch.arange(batch_size), n_atoms)
    assert len(batch) == sum_n_atoms
    ptr = torch.cat((torch.zeros(1, dtype=torch.int64), torch.cumsum(n_atoms, dim=0)))
    assert len(ptr) == batch_size + 1
    all_pos = torch.zeros((sum_n_atoms, 3))
    all_species = torch.zeros(sum_n_atoms, dtype=torch.int64)
    all_cell = torch.zeros((batch_size, 3, 3))
    all_pos_is_fractional = torch.ones(batch_size, dtype=torch.bool)

    for config_index, config in enumerate(all_configs):
        species = config.get_atomic_numbers()
        pos = config.get_scaled_positions(wrap=True)
        cell = config.get_cell()
        assert len(species) == len(pos)
        assert ptr[config_index + 1] - ptr[config_index] == len(species)
        all_pos[ptr[config_index]:ptr[config_index + 1]] = torch.tensor(pos)
        all_species[ptr[config_index]:ptr[config_index + 1]] = torch.tensor(species)
        # cell[:] converts the ase.cell.Cell object to a numpy array.
        all_cell[config_index] = torch.tensor(cell[:])

    return Data(pos=all_pos, cell=all_cell, species=all_species, ptr=ptr, n_atoms=n_atoms, batch=batch,
                pos_is_fractional=all_pos_is_fractional)


# Copied from https://github.com/jiaor17/DiffCSP/blob/7121d159826efa2ba9500bf299250d96da37f146/diffcsp/common/data_utils.py
class StandardScaler:

    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means),
                              np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds),
                             np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(
            self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none
