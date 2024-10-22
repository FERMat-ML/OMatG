import ase
import torch
from ase.build.niggli import niggli_reduce_cell
from typing import TYPE_CHECKING, Union, Tuple
import numpy as np
from ase.build.tools import update_cell_and_positions
from ase.data import atomic_numbers
from pathlib import Path
import lmdb
import tqdm
import pickle as pkl

if TYPE_CHECKING:
    from .datamodule import Configuration


def niggli_reduce_configuration(configuration: "Configuration") -> None:
    """
    Modified version of  :class:`ase.build.tools.niggli_reduce` that works with torch tensors.

    ------------------------------------------------------------------------------------
    Convert the supplied atoms object's unit cell into its
    maximally-reduced Niggli unit cell. Even if the unit cell is already
    maximally reduced, it will be converted into its unique Niggli unit cell.
    This will also wrap all atoms into the new unit cell.

    References:

    Niggli, P. "Krystallographische und strukturtheoretische Grundbegriffe.
    Handbuch der Experimentalphysik", 1928, Vol. 7, Part 1, 108-176.

    Krivy, I. and Gruber, B., "A Unified Algorithm for Determining the
    Reduced (Niggli) Cell", Acta Cryst. 1976, A32, 297-298.

    Grosse-Kunstleve, R.W.; Sauter, N. K.; and Adams, P. D. "Numerically
    stable algorithms for the computation of reduced unit cells", Acta Cryst.
    2004, A60, 1-6.
    """
    from ase.geometry.geometry import permute_axes

    # everything we have is periodic
    atoms = ase.Atoms(
        symbols=configuration.species,
        positions=configuration.coords,
        cell=configuration.cell,
        pbc=configuration.PBC,
    )
    # Make sure non-periodic cell vectors are orthogonal
    non_periodic_cv = atoms.cell[~atoms.pbc]
    periodic_cv = atoms.cell[atoms.pbc]
    if not np.isclose(np.dot(non_periodic_cv, periodic_cv.T), 0).all():
        raise ValueError('Non-orthogonal cell along non-periodic dimensions')

    input_atoms = atoms

    # Permute axes, such that the non-periodic are along the last dimensions,
    # since niggli_reduce_cell will change the order of axes.

    permutation = np.argsort(~atoms.pbc)
    ipermutation = np.empty_like(permutation)
    ipermutation[permutation] = np.arange(len(permutation))
    atoms = permute_axes(atoms, permutation)

    # Perform the Niggli reduction on the cell
    nonpbc = ~atoms.pbc
    uncompleted_cell = atoms.cell.uncomplete(atoms.pbc)
    new_cell, op = niggli_reduce_cell(uncompleted_cell)
    new_cell[nonpbc] = atoms.cell[nonpbc]
    update_cell_and_positions(atoms, new_cell, op)

    # Undo the prior permutation.
    atoms = permute_axes(atoms, ipermutation)
    input_atoms.cell[:] = atoms.cell
    input_atoms.positions[:] = atoms.positions

    # TODO: Remove accessing private attributes
    configuration._cell = torch.from_numpy(atoms.cell[:]).to(configuration.cell.dtype)
    configuration._coords = torch.from_numpy(atoms.positions).to(configuration.coords.dtype)


def niggli_reduce_data(species, coordinates, cell) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modified version of  :class:`ase.build.tools.niggli_reduce` that works with torch tensors.
    Same as niggli_reduce_configuration but takes species, coordinates and cell as input.

    TODO: Merge the two methods, they are almost identical.

    :param species: Atomic numbers of the atoms
    :param coordinates: Coordinates of the atoms
    :param cell: Cell vectors of the system
    :return: Reduced cell and coordinates
    """
    from ase.geometry.geometry import permute_axes

    if isinstance(species, list):
        species_np = np.array([atomic_numbers[s] for s in species]) if isinstance(species[0], str) else np.array(species)
    elif isinstance(species, torch.Tensor):
        species_np = species.numpy()
    coordinates_np = coordinates if isinstance(coordinates, np.ndarray) else coordinates.numpy()
    cell_np = cell if isinstance(cell, np.ndarray) else cell.numpy()

    # everything we have is periodic
    atoms = ase.Atoms(
        numbers=species_np,
        positions=coordinates_np,
        cell=cell_np,
        pbc=[True, True, True],
    )

    # Make sure non-periodic cell vectors are orthogonal
    non_periodic_cv = atoms.cell[~atoms.pbc]
    periodic_cv = atoms.cell[atoms.pbc]
    if not np.isclose(np.dot(non_periodic_cv, periodic_cv.T), 0).all():
        raise ValueError('Non-orthogonal cell along non-periodic dimensions')

    input_atoms = atoms

    # Permute axes, such that the non-periodic are along the last dimensions,
    # since niggli_reduce_cell will change the order of axes.

    permutation = np.argsort(~atoms.pbc)
    ipermutation = np.empty_like(permutation)
    ipermutation[permutation] = np.arange(len(permutation))
    atoms = permute_axes(atoms, permutation)

    # Perform the Niggli reduction on the cell
    nonpbc = ~atoms.pbc
    uncompleted_cell = atoms.cell.uncomplete(atoms.pbc)
    new_cell, op = niggli_reduce_cell(uncompleted_cell)
    new_cell[nonpbc] = atoms.cell[nonpbc]
    update_cell_and_positions(atoms, new_cell, op)

    # Undo the prior permutation.
    atoms = permute_axes(atoms, ipermutation)
    input_atoms.cell[:] = atoms.cell
    input_atoms.positions[:] = atoms.positions

    # TODO: Remove accessing private attributes
    return (torch.from_numpy(atoms.cell[:]).to(cell.dtype),
            torch.from_numpy(atoms.positions).to(coordinates.dtype))


def diffscp_to_lmdb(diffcsp_ds_file: Union[str, Path], lmdb_file: Union[str, Path], properties: list = None):
    """
    Convert a diffcsp csv dataset to an lmdb file.

    Note: ASE niggli reduction gives rotated cell vectors, as opposed to pymatgen.
    It rotates it such that first vector is alined along x axis.

    :param diffcsp_ds_file:
    :param lmdb_file:
    :return:
    """
    if isinstance(diffcsp_ds_file, str):
        diffcsp_ds_file = Path(diffcsp_ds_file)
    if isinstance(lmdb_file, str):
        lmdb_file = Path(lmdb_file)

    if not diffcsp_ds_file.exists():
        raise FileNotFoundError(f"{diffcsp_ds_file} does not exist.")

    if lmdb_file.exists():
        raise FileExistsError(f"{lmdb_file} already exists.")

    try:
        import pandas as pd
        from pymatgen.core.structure import Structure
        from pymatgen.core.lattice import Lattice

    except ImportError:
        raise ImportError("pandas and pymatgen is required to convert diffcsp dataset to lmdb")

    ds = pd.read_csv(diffcsp_ds_file)

    env = lmdb.open(lmdb_file, map_size=int(1e12), subdir=False)
    txn = env.begin(write=True)

    n_structures = len(ds)
    pbc = torch.tensor([1, 1, 1], dtype=torch.int32) # all are assumed to be periodic

    pbar = tqdm.tqdm(total=n_structures)
    for i in range(n_structures):
        # data_utils.py :1158 crystal_str = row['cif']
        crystal = Structure.from_str(ds["cif"][i], fmt="cif")

        # data_utils.py : 110 , niggli  = True, conf/data/*.yaml
        crystal_likely = crystal.get_reduced_structure()

        crystal_likely = Structure(
                lattice=Lattice.from_parameters(*crystal_likely.lattice.parameters),
                species=crystal_likely.species,
                coords=crystal_likely.frac_coords,
                coords_are_cartesian=False,
        )

        pos = crystal_likely.lattice.get_cartesian_coords(crystal_likely.frac_coords)
        cell = np.array(crystal_likely.lattice.as_dict()["matrix"])
        atomic_numbers = np.array(crystal_likely.atomic_numbers, dtype=np.int32)

        property_dict = {}
        for prop in properties:
            property_dict[prop] = torch.tensor(ds[prop][i])

        ids = ds["material_id"][i]
        txn.put(f"{i}".encode(), pkl.dumps({"pos": torch.from_numpy(pos),
                                            "cell": torch.from_numpy(cell),
                                            "atomic_numbers": torch.from_numpy(atomic_numbers),
                                            "ids": ids,
                                            "pbc": pbc} | property_dict))
        pbar.update(1)

    txn.commit()
    env.close()
