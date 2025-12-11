from typing import Any, Optional, Sequence
from ase import Atoms
from ase.build.tools import niggli_reduce
from ase.symbols import Symbols
from pymatgen.core import Structure as PymatgenStructure, Lattice as PymatgenLattice
import torch


class Structure(object):
    """
    Storage for a single crystalline structure of atoms.

    The structure is represented by its cell, atom numbers, and Cartesian or fractional atom coordinates.
    Additionally, one can store arbitrary properties and metadata associated with the structure in this class.

    This class also provides methods to change the representation of the structure. It is possible to convert the
    Cartesian atom coordinates to fractional coordinates and vice versa, and to Niggli reduce the structure.

    :param cell:
        A 3x3 matrix of the lattice vectors. The [i, j]-th element is the jth Cartesian coordinate of the ith unit
        vector.
    :type cell: torch.Tensor
    :param atomic_numbers:
        A vector of N integers giving the atomic numbers of the atoms, where N is the number of atoms.
    :type atomic_numbers: torch.Tensor
    :param pos:
        A Nx3 matrix of the fractional or Cartesian coordinates of the atoms in the structure, where N is the number of
        atoms.
    :type pos: torch.Tensor
    :param property_dict:
        An optional dictionary of properties associated with the structure.
        Defaults to None.
    :type property_dict: Optional[dict[str, Any]]
    :param metadata:
        An optional dictionary of metadata associated with the structure.
        Defaults to None.
    :type metadata: Optional[dict[str, Any]]
    :param pos_is_fractional:
        Whether the given atomic positions are in fractional coordinates.
        Defaults to False.
    :type pos_is_fractional: bool

    :raises AssertionError:
        If the cell is not a 3x3 matrix or if the pos is not a Nx3 matrix, where N is the length of species.
    """

    def __init__(self, cell: torch.Tensor, atomic_numbers: torch.Tensor, pos: torch.Tensor,
                 property_dict: Optional[dict[str, Any]] = None, metadata: Optional[dict[str, Any]] = None,
                 pos_is_fractional: bool = False) -> None:
        """Constructor for the Structure class."""
        assert cell.shape == (3, 3)
        assert atomic_numbers.dim() == 1
        assert pos.shape == (len(atomic_numbers), 3)
        self._cell = cell
        self._atomic_numbers = atomic_numbers
        self._pos = pos
        self._property_dict = property_dict if property_dict is not None else {}
        self._metadata = metadata if metadata is not None else {}
        self._fractional = pos_is_fractional

    @classmethod
    def from_dictionary(cls, data: dict[str, torch.Tensor], property_keys: Sequence[str],
                        metadata: dict[str, Any], pos_is_fractional: bool = False) -> "Structure":
        """
        Create a Structure object from the given data dictionary and metadata.

        The data dictionary is expected to contain at least the following keys:

        - "cell": A 3x3 torch.Tensor of the lattice vectors.
           The [i, j]-th element is the jth Cartesian coordinate of the ith unit vector.
        - "pos": A Nx3 torch.Tensor of fractional or Cartesian atomic positions.
        - "atomic_numbers": A torch tensor of N integers giving the atomic numbers of the atoms.

        Additionally, the data dictionary should contain additional property keys specified in the property_keys
        sequence.

        :param data:
            A dictionary representing the structure.
        :type data: dict[str, torch.Tensor]
        :param property_keys:
            A sequence of property keys to be included in the Structure's property_dict.
        :type property_keys: Sequence[str]
        :param metadata:
            A dictionary of metadata to be included in the Structure's metadata.
        :type metadata: dict[str, Any]
        :param pos_is_fractional:
            Whether the atomic positions in the data dictionary are in fractional coordinates.
            Defaults to False.
        :type pos_is_fractional: bool

        :return:
            A Structure object created from the data dictionary and metadata.
        :rtype: Structure

        """
        return cls(
            cell=data["cell"],
            atomic_numbers=data["atomic_numbers"].to(torch.int64),  # For potential cross-entropy loss, ensure int64.
            pos=data["pos"],
            property_dict={prop: data[prop] for prop in property_keys},
            metadata=metadata,
            pos_is_fractional=pos_is_fractional
        )

    def to_dictionary(self) -> dict[str, Any]:
        """
        Return the structure as a data dictionary.

        The returned dictionary contains the following keys:
        - "cell": A 3x3 torch.Tensor of the lattice vectors.
        - "pos": A Nx3 torch.Tensor of Cartesian atomic positions.
        - "atomic_numbers": A torch tensor of N integers giving the atomic numbers of the atoms.

        Additionally, the dictionary includes all properties from the property_dict and metadata.

        :return:
            A data dictionary representing the structure.
        :rtype: dict[str, Any]
        """
        # Always return Cartesian coordinates in the dictionary.
        if self.pos_is_fractional:
            return {
                "pos": torch.matmul(self.pos, self.cell),
                "cell": self.cell,
                "atomic_numbers": self.atomic_numbers
            } | self.property_dict | self.metadata
        else:
            return {
                "pos": self.pos,
                "cell": self.cell,
                "atomic_numbers": self.atomic_numbers
            } | self.property_dict | self.metadata

    @property
    def cell(self) -> torch.Tensor:
        """
        Return the cell vectors of the structure.

        :return:
            A 3x3 matrix of the lattice vectors. The [i, j]-th element is the jth Cartesian coordinate of the ith unit
            vector.
        :rtype: torch.Tensor
        """
        return self._cell

    @property
    def atomic_numbers(self) -> torch.Tensor:
        """
        Return the atomic numbers of the atoms in the structure.

        :return:
            A vector of N integers giving the atomic numbers of the atoms, where N is the number of atoms.
        :rtype: torch.Tensor
        """
        return self._atomic_numbers

    @property
    def symbols(self) -> list[str]:
        """
        Return the chemical symbols of the atoms in the structure.

        :return:
            A list of N strings giving the chemical symbols of the atoms, where N is the number of atoms.
        :rtype: list[str]
        """
        return list(Symbols(self._atomic_numbers.numpy()))

    @property
    def pos(self) -> torch.Tensor:
        """
        Return the Cartesian or fractional coordinates of the atoms in the structure.

        If convert_to_fractional() has been called, these coordinates will be fractional coordinates. Otherwise, they
        will be Cartesian coordinates.

        :return:
            A Nx3 matrix of the coordinates of the atoms, where N is the number of atoms.
        :rtype: torch.Tensor
        """
        return self._pos

    @property
    def pos_is_fractional(self) -> bool:
        """
        Return whether the atomic positions returned by pos are in fractional coordinates.

        This will be True after calling convert_to_fractional().

        :return:
            True if the atomic positions are in fractional coordinates, False otherwise.
        :rtype: bool
        """
        return self._fractional

    @property
    def property_dict(self) -> dict[str, Any]:
        """
        Return the property dictionary of the structure.

        :return:
            A dictionary of properties associated with the structure.
        :rtype: dict[str, Any]
        """
        return self._property_dict

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Return the metadata of the structure.

        :return:
            A dictionary of metadata associated with the structure.
        :rtype: dict[str, Any]
        """
        return self._metadata

    def to(self, floating_point_precision: torch.dtype) -> None:
        """
        Convert the floating point precision of the structure to the given torch precision.

        :param floating_point_precision:
            The torch floating point dtype to convert to.
            Should be one of torch.float32, torch.float64, torch.float16, torch.bfloat16,
            torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz, torch.float8_e8m0fnu, or torch.float4_e2m1fn_x2.
        :type floating_point_precision: torch.dtype

        :raises ValueError:
            If the given floating point precision is not supported.
        """
        # See https://docs.pytorch.org/docs/stable/tensor_attributes.html.
        if not floating_point_precision in (torch.float32, torch.float64, torch.float16, torch.bfloat16,
                                            torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz,
                                            torch.float8_e5m2fnuz, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2):
            raise ValueError(f"Unsupported floating point precision: {floating_point_precision}. "
                             f"Supported precisions are torch.float32, torch.float64, torch.float16, "
                             f"torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, "
                             f"torch.float8_e4m3fnuz, torch.float8_e5m2fnuz, torch.float8_e8m0fnu, "
                             f"and torch.float4_e2m1fn_x2.")
        self._cell = self._cell.to(floating_point_precision)
        self._pos = self._pos.to(floating_point_precision)
        for key, value in self._property_dict.items():
            if torch.is_tensor(value) and value.is_floating_point():
                self._property_dict[key] = value.to(floating_point_precision)

    def get_ase_atoms(self) -> Atoms:
        """
        Convert the structure to an ASE Atoms object.

        :return:
            The ASE Atoms object.
        :rtype: Atoms
        """
        if self._fractional:
            # noinspection PyTypeChecker
            return Atoms(numbers=self.atomic_numbers.tolist(), scaled_positions=self.pos.numpy(),
                         cell=self.cell.numpy(), pbc=True, info=self.property_dict | self.metadata)
        else:
            # noinspection PyTypeChecker
            return Atoms(numbers=self.atomic_numbers.tolist(), positions=self.pos.numpy(), cell=self.cell.numpy(),
                         pbc=True, info=self.property_dict | self.metadata)

    def get_pymatgen_structure(self) -> PymatgenStructure:
        """
        Convert the structure to a pymatgen Structure object.

        :return:
            The pymatgen Structure object.
        :rtype: PymatgenStructure
        """
        if self._fractional:
            return PymatgenStructure(
                lattice=PymatgenLattice(self.cell.numpy()),
                species=self.symbols,
                coords=self.pos.numpy(),
                coords_are_cartesian=False,
                properties=self.property_dict | self.metadata
            )
        else:
            return PymatgenStructure(
                lattice=PymatgenLattice(self.cell.numpy()),
                species=self.symbols,
                coords=self.pos.numpy(),
                coords_are_cartesian=True,
                properties=self.property_dict | self.metadata
            )

    def niggli_reduce(self) -> None:
        """
        Niggli reduce the structure.
        """
        atoms = self.get_ase_atoms()
        niggli_reduce(atoms)
        self._cell = torch.tensor(atoms.cell[:], dtype=self._cell.dtype)
        if self._fractional:
            self._pos = torch.tensor(atoms.get_scaled_positions(), dtype=self._pos.dtype)
        else:
            self._pos = torch.tensor(atoms.positions, dtype=self._pos.dtype)

    def convert_to_fractional(self) -> None:
        """
        Convert the atomic positions to fractional coordinates.
        """
        if not self._fractional:
            with torch.no_grad():
                # Solve r = f * cell for f.
                self._pos = torch.remainder(torch.linalg.solve(self._cell, self._pos, left=False), 1.0)
            self._fractional = True

    def convert_to_cartesian(self) -> None:
        """
        Convert the atomic positions to cartesian coordinates.
        """
        if self._fractional:
            with torch.no_grad():
                self._pos = torch.matmul(self._pos, self._cell)
            self._fractional = False


if __name__ == '__main__':
    from numpy import transpose
    from omg.datamodule import StructureDataset

    dataset_lmdb = StructureDataset(file_path="data/mp_20/test.lmdb", lazy_storage=False)

    structure = dataset_lmdb[0]
    structure.niggli_reduce()
    print(structure.get_pymatgen_structure().cart_coords)
    print(structure.get_ase_atoms().positions)
    print(structure.get_pymatgen_structure().lattice.matrix)
    print(structure.get_ase_atoms().cell)
    structure.convert_to_fractional()
    print(structure.get_pymatgen_structure().cart_coords)
    print(structure.get_ase_atoms().positions)
    print(structure.get_pymatgen_structure().lattice.matrix)
    print(structure.get_ase_atoms().cell)
    print()

    print(structure.pos)
    structure.convert_to_fractional()
    print(structure.pos)
    structure.convert_to_cartesian()
    print(structure.pos)
    structure.convert_to_cartesian()
    print(structure.pos)
