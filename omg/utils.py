import torch
from enum import Enum, auto
from torch_geometric.data import Data
from typing import List, Union
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.loggers.wandb import WandbLogger
import matplotlib.pyplot as plt 

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

# TODO: make options accesible to OMG via CLI
def xyz_saver(data: Union [Data, List[Data]]):
    """
    Takes data that has been generated and saves it as xyz file
    """
    from ase import Atoms
    from ase.io import write
    import time
    if not isinstance(data, list):
        data = [data]
    atoms = []
    for d in data:
        batch_size = len(d.n_atoms)
        for i in range(batch_size):
            lower, upper = d.ptr[i*1], d.ptr[(i*1)+1]
            atoms.append(Atoms(numbers=d.species[lower:upper], scaled_positions=d.pos[lower:upper, :], cell=d.cell[i, :, :], pbc=(1,1,1)))
    write(f'{time.strftime("%Y%m%d-%H%M%S")}.xyz', atoms)

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
