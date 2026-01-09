from typing import Optional
import torch
from torch_geometric.data import Data
from omg.datamodule import Structure


class OMGData(Data):
    """
    Representation of a single crystalline structure of atoms that is compatible with Pytorch Geometric.

    When using this class with a dataloader from Pytorch Geometric, multiple OMGData objects can be batched together
    into a single batched graph object containing several structures.

    For a batch of size batch_size, the batched graph object will have the following attributes:

    - n_atoms: torch.Tensor of shape (batch_size, ) containing the number of atoms in each structure.
    - species: torch.Tensor of shape (sum(n_atoms), ) containing the atomic numbers of the atoms in the structures.
    - cell: torch.Tensor of shape (batch_size, 3, 3) containing the cell vectors of the structures.
    - pos: torch.Tensor of shape (sum(n_atoms), 3) containing the fractional or Cartesian atomic positions of the
           atoms in the structures.
    - pos_is_fractional: torch.Tensor of shape (batch_size,) with booleans indicating whether the atomic positions are
                         given in fractional (or alternatively, Cartesian) coordinates.
    - property: Dictionary containing the properties of the structures.
    - batch: torch.Tensor of shape (sum(n_atoms), ) containing the index of the structure to which each atom belongs
             (added automatically by PyTorch Geometric).
    - ptr: torch.Tensor of shape (batch_size + 1, ) containing the cumulative sum of n_atoms (added automatically by
           PyTorch Geometric).

    Note that Pytorch Geometric dataloaders while batching will initialize the OMGData object without any arguments,
    and then populate the attributes directly. Therefore, the constructor has to handle being called with no arguments.
    In this case, all attributes are initialized to None.

    :param structure:
        Structure object containing the information about the crystalline structure or None.
        Defaults to None
    :type structure: Optional[Structure]
    """

    def __init__(self, structure: Optional[Structure] = None) -> None:
        """Constructor for the OMGData class."""
        super().__init__()
        if structure is None:
            self.n_atoms = None
            self.species = None
            self.cell = None
            self.pos = None
            self.pos_is_fractional = None
            self.property = None
        else:
            self.n_atoms = torch.tensor(len(structure.atomic_numbers))  # Shape: (1, )
            self.species = structure.atomic_numbers  # Shape: (n_atoms, )
            self.cell = structure.cell.unsqueeze(0) # Shape: (1, 3, 3).
            self.pos = structure.pos  # Shape: (n_atoms, 3).
            self.pos_is_fractional = torch.tensor(structure.pos_is_fractional, dtype=torch.bool)  # Shape: (1, )
            self.property = structure.property_dict
