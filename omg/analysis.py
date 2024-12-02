from ase.io import read
from tqdm import tqdm
from ase.data import covalent_radii as CR
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import spglib
from ase import Atoms

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

from collections import Counter
def identify_spacegroup_varprec(spg_cell, angle_tolerance, max_iterations = 200):
    """Identifying the best space group using varying tolerances"""

    # precision to spglib is in cartesian distance,
    prec = 5.
    precs = []
    grps = []
    grp = 'None'
    highest_symmetry_group = 'None'
    max_group = 0

    # Try a range of precisions and record the determined spacegroups
    counter = 0
    while grp is None or grp.split()[-1] not in ['(1)', '(2)']:
        counter += 1
        if counter > max_iterations:
            break
        grp = spglib.get_spacegroup(spg_cell, symprec=prec, angle_tolerance=angle_tolerance)
        grps.append(grp)
        precs.append(prec)
        prec /= 2

        if grp is not None:
            group_num = int(grp.split()[-1].replace('(', '').replace(')', ''))
            if group_num > max_group:
                max_group = group_num
                highest_symmetry_group = grp

    if all(g is None for g in grps):
        #raise ValueError("No symmetry groups found!")
        print("No symmetry groups found!")
        return None
    # One measure of best group is the highest symmetry group
    highest_symmetry_group_prec = precs[::-1][grps[::-1].index(highest_symmetry_group)]

    # An alternative measure is the most commonly occurring group
    counts = Counter(grps)
    if None in counts:
        del counts[None]
    most_common_group = counts.most_common(1)[0][0]
    most_common_group_prec = precs[::-1][grps[::-1].index(most_common_group)]

    return {'common': (most_common_group, most_common_group_prec),
            'highest': (highest_symmetry_group, highest_symmetry_group_prec)}


def get_space_group(atoms, niggli=False, var_prec=True, symprec=1e-3, angle_tolerance=4.):
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

    if var_prec:
        sg_symprec_dict = identify_spacegroup_varprec(spglib_cell, angle_tolerance)
        if sg_symprec_dict is not None:
            print(sg_symprec_dict['common'], sg_symprec_dict['highest'])
            sg = spglib.get_spacegroup(spglib_cell, symprec=sg_symprec_dict['common'][1], angle_tolerance=angle_tolerance)
            sym_data = spglib.get_symmetry_dataset(spglib_cell, sg_symprec_dict['common'][1], angle_tolerance=angle_tolerance)
            #sg = spglib.get_spacegroup(spglib_cell, symprec=sg_symprec_dict['highest'][1], angle_tolerance=angle_tolerance)
            #sym_data= spglib.get_symmetry_dataset(spglib_cell, sg_symprec_dict['highest'][1], angle_tolerance=angle_tolerance)
            sym_struc = Atoms(numbers=sym_data.std_types, scaled_positions=sym_data.std_positions, cell=sym_data.std_lattice, pbc=True)
        else:
            print("Space group could not be determined.")
            print(spglib.get_error_message())
            return None, None, None, None
    else:
        sg = spglib.get_spacegroup(spglib_cell, symprec=symprec, angle_tolerance=angle_tolerance)
        if sg is None:
            #raise RuntimeError("Space group could not be determined.")
            print("Space group could not be determined.")
            print(spglib.get_error_message())
            return None, None, None, None
        else:
            sym_data = spglib.get_symmetry_dataset(spglib_cell, symprec=symprec, angle_tolerance=angle_tolerance)
            sym_struc = Atoms(numbers=sym_data.std_types, scaled_positions=sym_data.std_positions, cell=sym_data.std_lattice, pbc=True)

    sg_group = sg.split()[0]
    sg_num = int(sg.split()[1].replace('(', '').replace(')', ''))

    if sg_num < 1 or sg_num > 230:
        # RuntimeError("Space group number is out of range.")
        print("Space group number is out of range.")
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
        #raise RuntimeError("Crystal system could not be determined.")
        print("Crystal system could not be determined.")
    
    return sg_group, sg_num, crystal_system, sym_struc


def structure_matcher(s1, s2, ltol=0.2, stol=0.3, angle_tol=5):
    """ Checks if structures s1 and s2 of ase type Atoms are the same."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor

    sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    # conversion to pymatgen type
    a1 = AseAtomsAdaptor.get_structure(s1)
    a2 = AseAtomsAdaptor.get_structure(s2)
    return sm.fit(a1, a2)

def element_check(s1, s2):
    """Check if s1 and s2 (both ase Atoms types) are of same composition
    """
    from omg.globals import MAX_ATOM_NUM
    import numpy as np

    s1_counts = np.bincount(s1.numbers, minlength=MAX_ATOM_NUM)
    s2_counts = np.bincount(s2.numbers, minlength=MAX_ATOM_NUM)
    
    s1_min = np.amin(s1_counts[np.where(s1_counts>0)])
    s2_min = np.amin(s2_counts[np.where(s2_counts>0)])

    return np.array_equal(s1_counts/s1_min, s2_counts/s2_min)

def match_rate(atoms_list, ref_list, ltol=0.2, stol=0.3, angle_tol=5):
    """
    Compare the structures in two xyz files. Return rate of matches between the two files.
    Returns rate of matches within atoms_list.
    """
    match_count = 0
    # Check if the structures are the same
    for atoms_1 in atoms_list:
        for atoms_2 in ref_list:
            if element_check(atoms_1, atoms_2):
                if structure_matcher(atoms_1, atoms_2, ltol=ltol, stol=stol, angle_tol=angle_tol):
                    match_count += 1
                    break

    return match_count / len(atoms_list)

def reduce(atoms_1_list):
    """
    Compare structures within one xyz file. Return rate of unique structures.
    """
    match_counts = {}
    for i, atoms_1 in enumerate(atoms_1_list):
        match_counts[i] = []
        for j, atoms_2 in enumerate(atoms_1_list):
            # to save time, only compare if not already compared or not self
            if (i != j) and not (j in match_counts[i]):
                if element_check(atoms_1, atoms_2):
                    if structure_matcher(atoms_1, atoms_2):
                        match_counts[i].append(j)
    
    match_count = len(atoms_1_list)
    for key in match_counts.keys():
        match_count -= len(match_counts[key])

    return match_count / len(atoms_1_list)