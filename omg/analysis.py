from ase.io import read
from tqdm import tqdm
from ase.data import covalent_radii as CR
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import spglib

# Adapted from: https://gist.github.com/tgmaxson/8b9d8b40dc0ba4395240
def get_coordination_numbers(atoms, covalent_percent=1.25, dis=None):
    """
    Returns an array of coordination numbers determined by
    distance and covalent radii.  By default a bond is defined as 120% of the combined radii
    or less. This can be changed by setting 'covalent_percent' to a float representing a 
    factor to multiple by (default = 1.2).

    If 'exclude' is set to an array,  these atomic numbers with be unable to form bonds.
    This only excludes them from being counted from other atoms,  the coordination
    numbers for these atoms will still be calculated,  but will be unable to form
    bonds to other excluded atomic numbers.

    atoms: type ase.Atoms. The structure to calculate the average coordination number for.
    covalent_percent: type float. The percentage of the covalent radii to use as a cutoff distance.
    dis: type float. radial cutoff distance (to be used instead of covalent radii) 
    """

    # Get all the distances
    distances = np.divide(atoms.get_all_distances(mic=True), covalent_percent)
    
    # Atomic Numbers
    numbers = atoms.numbers
    # Coordination Numbers for each atom
    cn = []
    cr = np.take(CR, numbers)
    # Array of indices of bonded atoms.  len(bonded[x]) == cn[x]
    bonded = []
    indices = list(range(len(atoms)))
    for i in indices:
        bondedi = []
        for ii in indices:
            # Skip if measuring the same atom
            if i == ii:
                continue
            # if dis is set, use that instead of the covalent radii
            if dis:
                if dis >= distances[i,ii]:
                    bondedi.append(ii)
            else:
                if (cr[i] + cr[ii]) >= distances[i,ii]:
                    bondedi.append(ii)
        # Add this atoms bonds to the bonded list
        bonded.append(bondedi)
    for i in bonded:
        cn.append(len(i))
    return cn

def get_coordination_numbers_species(atoms, covalent_percent=1.25, dis=None, naming='species'):
    """
    Returns a dictionary with the species as keys and the coordination numbers as values, determined by
    distance and covalent radii averaged over each species.  By default a bond is defined as 120% of the combined radii
    or less. This can be changed by setting 'covalent_percent' to a float representing a 
    factor to multiple by (default = 1.2).

    If 'exclude' is set to an array,  these atomic numbers with be unable to form bonds.
    This only excludes them from being counted from other atoms,  the coordination
    numbers for these atoms will still be calculated,  but will be unable to form
    bonds to other excluded atomic numbers.

    atoms: type ase.Atoms. The structure to calculate the average coordination number for.
    covalent_percent: type float. The percentage of the covalent radii to use as a cutoff distance.
    dis: type float. radial cutoff distance (to be used instead of covalent radii) 
    naming: type str. 'species' or 'number'. 
    If 'species', the keys of the dictionary will be the chemical symbol. If 'number', the keys will be the atomic number.
    """

    # Get all the distances
    distances = np.divide(atoms.get_all_distances(mic=True), covalent_percent)
    
    # Atomic Numbers
    numbers = atoms.numbers
    # Coordination Numbers for each atom
    cn = []
    cr = np.take(CR, numbers)
    # Array of indices of bonded atoms.  len(bonded[x]) == cn[x]
    bonded = []
    indices = list(range(len(atoms)))
    for i in indices:
        bondedi = []
        for ii in indices:
            # Skip if measuring the same atom
            if i == ii:
                continue
            # if dis is set, use that instead of the covalent radii
            if dis:
                if dis >= distances[i,ii]:
                    bondedi.append(ii)
            else:
                if (cr[i] + cr[ii]) >= distances[i,ii]:
                    bondedi.append(ii)
        # Add this atoms bonds to the bonded list
        bonded.append(bondedi)
    for i in bonded:
        cn.append(len(i))
    
    cn_dict = {}

    if naming == 'species':
        symbols = atoms.get_chemical_symbols()
        species_list = [str(i) for i in np.unique(symbols)]
        for species in species_list:
            cn_dict[species] = [coord_num for atom, coord_num in zip(symbols, cn) if atom == species]
        
    elif naming == 'number':
        elements = np.unique(numbers)
        for element in elements:
            cn_dict[element] = [coord_num for atom, coord_num in zip(numbers, cn) if atom == element]

    else:
        raise ValueError("Invalid naming argument. Must be 'species' or 'number'.")
    
    return cn_dict


def get_space_group(atoms, niggli=False, symprec=1e-3, angle_tolerance=0.5):
    """Calculate the space group of a given structure, optionally using the primitive cell.
    Assumes atoms is a ase Atoms object. 
    Niggli will generate the Niggli reduced cell. 
    Symprec is the symmetry precision used by spglib.

    Returns the space group symbol (str) and space group number (int) and crystal system (str).
    """

    if niggli:
        atoms = atoms.niggli_reduce() # TODO: SEEMS BROKEN (returning NoneType on test case)
    
    spglib_cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    #spglib_cell_primitive = spglib.find_primitive(spglib_cell)

    sg = spglib.get_spacegroup(spglib_cell, symprec=symprec, angle_tolerance=angle_tolerance)

    if sg is None:
        #raise RuntimeError("Space group could not be determined.")
        print("Space group could not be determined.")
        print(spglib.get_error_message())
        return None, None, None
    
    sg_group = sg.split()[0]
    sg_num = int(sg.split()[1].replace('(', '').replace(')', ''))

    if sg_num < 1 or sg_num > 230:
        raise RuntimeError("Space group number is out of range.")
    elif sg_num < 3:
        crystal_system = "Triclinic"
    elif 3 <= sg_num <= 15:
        crystal_system = "Monoclinic"
    elif 16 <= sg_num <= 74:
        crystal_system = "Orthorhombic"
    elif 75 <= sg_num <= 142:
        crystal_system = "Tetragonal"
    elif 143 <= sg_num <= 167:
        crystal_system = "Trigonal"
    elif 168 <= sg_num <= 194:
        crystal_system = "Hexagonal"
    elif 195 <= sg_num <= 230:
        crystal_system = "Cubic"
    else:
        raise RuntimeError("Crystal system could not be determined.")
    
    return sg_group, sg_num, crystal_system

