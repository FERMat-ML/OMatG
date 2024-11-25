from enum import Enum, auto
from pathlib import Path
from typing import List, Union
from ase import Atoms
from ase.io import read, write
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.loggers.wandb import WandbLogger
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree as kdtree


class DataField(Enum):
    pos = auto()
    cell = auto()
    species = auto()


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


def xyz_saver(data: Union[Data, List[Data]], filename: Path) -> None:
    """
    Takes data that has been generated and saves it as xyz file
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
            atoms.append(Atoms(numbers=d.species[lower:upper], scaled_positions=d.pos[lower:upper, :],
                               cell=d.cell[i, :, :], pbc=(1, 1, 1)))
    write(filename, atoms)


def xyz_reader(filename: Path) -> Data:
    """
    Reads an xyz file and returns a Data object.
    """
    if not filename.suffix == ".xyz":
        raise ValueError("The filename must have the suffix '.xyz'.")
    # Read all atoms from the file by using index=":".
    all_configs = read(filename, index=":")
    return convert_ase_atoms_to_data(all_configs)

# TODO: please move me to analysis.py when branches have been merged!
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

def structure_matcher(s1, s2, ltol=0.2, stol=0.3, angle_tol=5):
    """ Checks if structures s1 and s2 of ase type Atoms are the same."""
    sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    # conversion to pymatgen type
    a1 = AseAtomsAdaptor.get_structure(s1)
    a2 = AseAtomsAdaptor.get_structure(s2)
    return sm.fit(a1, a2)

# TODO: please move me to analysis.py when branches have been merged!
from omg.globals import MAX_ATOM_NUM
import numpy as np
def element_check(s1, s2):
    """Check if s1 and s2 (both ase Atoms types) are of same composition
    """
    s1_counts = np.bincount(s1.numbers, minlength=MAX_ATOM_NUM)
    s2_counts = np.bincount(s2.numbers, minlength=MAX_ATOM_NUM)
    
    s1_min = np.amin(s1_counts[np.where(s1_counts>0)])
    s2_min = np.amin(s2_counts[np.where(s2_counts>0)])

    return np.array_equal(s1_counts/s1_min, s2_counts/s2_min)

def convert_ase_atoms_to_data(all_configs: List[Atoms]) -> Data:
    """
    Convert a list of ASE Atoms objects to a PyTorch Geometric Data object.
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

    return Data(pos=all_pos, cell=all_cell, species=all_species, ptr=ptr, n_atoms=n_atoms, batch=batch)


class OMGLearningRateFinder(LearningRateFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_fit_start(self, trainer, pl_module):
        self.lr_find(trainer, pl_module)
        fig = self.optimal_lr.plot(suggest=True)
        if isinstance(trainer.logger, WandbLogger):
            # See https://github.com/Lightning-AI/pytorch-lightning/issues/2725
            directory = trainer.logger.experiment.dir
        else:
            directory = trainer.logger.log_dir
        plt.savefig(directory + "/lr-finder.png")
