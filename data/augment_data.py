import sys
import lmdb
import torch
import numpy as np
import pickle as pkl
from ase import Atoms
from omg.utils import xyz_saver, add_ghost_particles

def tesselate_dataset(filename:str):
    '''
    Voronoi tesselation of dataset
    :param filename:
        Path of dataset to tesselate
    :type filename: String
    '''

    # Open source and target
    base, ext = filename.rsplit(".", 1)
    new_base = f"{base}_voronoi"
    new_filename = f"{new_base}.{ext}"
    source_env = lmdb.open(filename, subdir=False, readonly=True)
    output_env = lmdb.open(f'{new_filename}', subdir=False, map_size=1e12)

    # Run loop
    with source_env.begin() as source_txn, output_env.begin(write=True) as output_txn:
        source_cursor = source_txn.cursor()
        for idx, (_, value) in enumerate(source_cursor):

            # Read atom
            value = pkl.loads(value)
            positions = value['pos'].numpy()
            cell = value['cell'].numpy()
            atomic_numbers = value['atomic_numbers'].numpy()
            pbc = value['pbc'].bool().numpy()

            # Create ASE Atoms object
            atoms = Atoms(
                numbers=atomic_numbers,
                positions=positions,
                cell=cell,
                pbc=pbc
            )

            # Tesselate
            atoms = add_ghost_particles(atoms)

            # Save new dataset
            output_txn.put(f"{idx}".encode('ascii'), pkl.dumps(atoms))

    print('TESSELATION COMPLETE')
    return

if __name__ == '__main__':

    # Augment dataset
    filename = sys.argv[1]
    tesselate_dataset(filename)