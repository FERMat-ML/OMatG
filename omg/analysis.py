from collections import Counter
from functools import partial
from multiprocessing import Pool
import os
from typing import Dict, List, Optional, Sequence, Tuple
from ase import Atoms
from ase.data import covalent_radii
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
import spglib
from omg.globals import MAX_ATOM_NUM
# Suppress spglib warnings.
os.environ["SPGLIB_WARNING"] = "OFF"


def get_bonds(atoms: Atoms, covalent_increase_factor: float = 1.25) -> List[List[int]]:
    """
    Compute the list of bonds for every atom in the given structure.

    The bonds are determined by comparing the distances between the atoms and the involved covalent radii. A bond is
    present if the two atoms are closer than the sum of their covalent radii times a given multiplicative factor.

    :param atoms:
        The structure to calculate the bonds for.
    :type atoms: Atoms
    :param covalent_increase_factor:
        The factor by which to multiply the sum of the covalent radii to determine the bond distance.
        Defaults to 1.25.
    :type covalent_increase_factor: float

    :return:
        List of bonds for the atoms in the structure.
    :rtype: List[List[int]]
    """
    distances = atoms.get_all_distances(mic=True)
    cr = [covalent_radii[number] for number in atoms.numbers]

    # List of bonded atoms for every atom.
    bonds = [[] for _ in range(len(atoms))]

    for first_index in range(len(atoms)):
        for second_index in range(first_index + 1, len(atoms)):
            if (cr[first_index] + cr[second_index]) * covalent_increase_factor >= distances[first_index, second_index]:
                bonds[first_index].append(second_index)
                bonds[second_index].append(first_index)

    return bonds


def get_coordination_numbers(atoms: Atoms, covalent_increase_factor: float = 1.25) -> List[int]:
    """
    Compute the coordination numbers for the atoms in the given structure.

    The coordination numbers are determined by comparing the distances between the atoms and the involved covalent
    radii. A bond that increases the coordination number is present if the two atoms are closer than the sum of their
    covalent radii times a given multiplicative factor.

    :param atoms:
        The structure to calculate the coordination numbers for.
    :type atoms: Atoms
    :param covalent_increase_factor:
        The factor by which to multiply the sum of the covalent radii to determine the bond distance.
        Defaults to 1.25.
    :type covalent_increase_factor: float

    :return:
        List of coordination numbers for the atoms in the structure.
    :rtype: List[int]
    """
    bonds = get_bonds(atoms, covalent_increase_factor)
    return [len(b) for b in bonds]


def get_coordination_numbers_species(atoms: Atoms, covalent_increase_factor: float = 1.25) -> Dict[str, List[int]]:
    """
    Compute a dictionary from the species to their coordination numbers in the given structure.

    The coordination numbers are determined by comparing the distances between the atoms and the involved covalent
    radii. A bond that increases the coordination number is present if the two atoms are closer than the sum of their
    covalent radii times a given multiplicative factor.

    Since any species can be present more than once in the structure, their coordination numbers are stored in a list.

    :param atoms:
        The structure to calculate the coordination numbers for.
    :type atoms: Atoms
    :param covalent_increase_factor:
        The factor by which to multiply the sum of the covalent radii to determine the bond distance.
        Defaults to 1.25.
    :type covalent_increase_factor: float

    :return:
        Dictionary from the species to their coordination numbers in the structure.
    :rtype: Dict[str, List[int]]
    """
    coordination_numbers = get_coordination_numbers(atoms, covalent_increase_factor)
    symbols = atoms.get_chemical_symbols()
    unique_species = [str(i) for i in np.unique(symbols)]
    return {species: [cn for cn, symb in zip(coordination_numbers, symbols) if symb == species]
            for species in unique_species}


def get_space_group(atoms: Atoms, symprec: float = 1.0e-5, angle_tolerance: float = -1.0,
                    var_prec: bool = False) -> (Optional[str], Optional[int], Optional[str], Optional[Atoms]):
    """
    Calculate the space group of a given structure using spglib with the given precision arguments.

    In addition to the space group name and number, the crystal system and a perfectly symmetrized structure are
    returned.

    If var_prec is False, this function effectively uses the get_spacegroup function of spglib (see
    https://spglib.readthedocs.io/en/stable/api/python-api/spglib/spglib.html#spglib.get_spacegroup and
    https://spglib.readthedocs.io/en/stable/api/python-api/spglib/spglib.html#spglib.get_symmetry for a documentation
    of the arguments). Note, however, that we directly use the get_symmetry_dataset function of spglib instead of the
    get_spacegroup function and then mirror the order of operations in the get_spacegroup function. This allows us to
    create a symmetrized structure.

    Spglib's get_symmetry_dataset function possibly returns None when the space group could not be determined, which
    mostly happens if symprec was chosen too large (or if there are overlaps between atoms). At the same time, if we
    choose symprec too small, one only gets triclinic space groups except for perfect crystals. If var_prec is True, in
    order to find a symprec value that gives something non-trivial, we start at a very large symprec value which either
    returns None or a non-triclinic space group. We then iteratively decrease symprec until the spacegroup becomes
    triclinic. We then return symmetry dataset corresponding to the most commonly occuring space group during this
    iteration.

    :param atoms:
        Structure to calculate the space group for.
    :type atoms: Atoms
    :param symprec:
        Symmetry search tolerance in the unit of length.
        Defaults to 1.0e-05 (spglib's default).
    :type symprec: float
    :param angle_tolerance:
        Symmetry search tolerance in the unit of angle deg. Normally, spglib does not recommend to use this argument. If
        the value is negative, spglib uses an internally optimized routine to judge symmetry.
        Defaults to -1.0 (spglib's default).
    :type angle_tolerance: float
    :param var_prec:
        If True, the function uses a variable precision to determine the space group.
        Defaults to False.
    :type var_prec: bool

    :return:
        The space-group name, the space-group number, the crystal system, symmetrized structure.
        If the space-group could not be determined, all return values are None.
    :rtype: (Optional[str], Optional[int], Optional[str], Optional[Atoms])
    """
    # Cell arguments for spglib, see
    # https://spglib.readthedocs.io/en/stable/api/python-api/spglib/spglib.html#spglib.get_symmetry.
    spglib_cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())

    if var_prec:
        sym_data = _get_symmetry_dataset_var_prec(atoms, angle_tolerance=angle_tolerance)
    else:
        sym_data = spglib.get_symmetry_dataset(spglib_cell, symprec=symprec, angle_tolerance=angle_tolerance)

    # This is the order of operations in spglib's get_spacegroup function.
    if sym_data is None:
        print(f"[WARNING] get_space_group: Space group could not be determined ({spglib.get_error_message()}).")
        return None, None, None, None

    spg_type = spglib.get_spacegroup_type(sym_data.hall_number)
    if spg_type is None:
        print("[WARNING] get_space_group: Space group could not be determined.")
        return None, None, None, None

    sg_group = str(spg_type.international_short)
    sg_num = int(sym_data.number)

    if sg_num < 1 or sg_num > 230:
        print("[WARNING] get_space_group: Space group could not be determined.")
        return None, None, None, None
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
    else:
        assert 195 <= sg_num <= 230
        crystal_system = "Cubic"

    sym_struc = Atoms(numbers=sym_data.std_types, scaled_positions=sym_data.std_positions,
                      cell=sym_data.std_lattice, pbc=True)

    return sg_group, sg_num, crystal_system, sym_struc


def _get_symmetry_dataset_var_prec(atoms: Atoms, angle_tolerance: float = -1.0,
                                   max_iterations: int = 200) -> Optional[spglib.SpglibDataset]:
    """
    Calculate the symmetry dataset of a given structure using spglib with a variable precision.

    Spglib's get_symmetry_dataset function possibly returns None when the space group could not be determined, which
    mostly happens if symprec was chosen too large (or if there are overlaps between atoms). At the same time, if we
    choose symprec too small, one only gets triclinic space groups except for perfect crystals. In order to find a
    symprec value that gives something non-trivial, we start at a very large symprec value which either returns None or
    a non-triclinic space group. We then iteratively decrease symprec until the spacegroup becomes triclinic. We then
    return symmetry dataset corresponding to the most commonly occuring space group during this iteration.

    :param atoms:
        Structure to calculate the symmetry dataset for.
    :type atoms: Atoms
    :param angle_tolerance:
        Symmetry search tolerance in the unit of angle deg. Normally, spglib does not recommend to use this argument. If
        the value is negative, spglib uses an internally optimized routine to judge symmetry.
        Defaults to -1.0 (spglib's default).
    :type angle_tolerance: float
    :param max_iterations:
        Maximum number of iterations to try to find a space group.
        Defaults to 200.
    :type max_iterations: int

    :return:
        The most commonly occurring symmetry dataset.
    :rtype: Optional[spglib.SpglibDataset]
    """
    # Cell arguments for spglib, see
    # https://spglib.readthedocs.io/en/stable/api/python-api/spglib/spglib.html#spglib.get_symmetry.
    spglib_cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    # Precision to spglib is in cartesian distance.
    prec = 5.0
    symmetry_datasets = []
    groups = []
    current_group = None
    iteration = 0

    # Space groups ending with (1) and (2) are triclinic. We decrease the precision until we get something triclinic.
    while current_group is None or current_group.split()[-1] not in ['(1)', '(2)']:
        iteration += 1
        if iteration > max_iterations:
            break

        dataset = spglib.get_symmetry_dataset(spglib_cell, symprec=prec, angle_tolerance=angle_tolerance)
        prec /= 2.0
        if dataset is None:
            continue
        spg_type = spglib.get_spacegroup_type(dataset.hall_number)
        if spg_type is None:
            continue
        current_group = "%s (%d)" % (spg_type.international_short, dataset.number)
        symmetry_datasets.append(dataset)
        groups.append(current_group)

    # All space groups were None.
    if len(groups) == 0:
        return None

    # Counting the number of occurrences should be done on the groups which are simple strings.
    counts = Counter(groups)
    most_common_group = counts.most_common(1)[0][0]
    return symmetry_datasets[groups.index(most_common_group)]


def _structure_matcher(atoms_one: Atoms, atoms_two: Atoms, ltol: float = 0.2, stol: float = 0.3,
                       angle_tol: float = 5.0) -> Optional[float]:
    """
    Checks if the two structures are the same by using pymatgen's StructureMatcher and, if so, return the
    root-mean-square displacement between the two structures. If the structures are different, return None.

    The root-mean-square displacement is normalized by (volume / number_sites) ** (1/3).

    The documentation of pymatgen's StructureMatcher can be found here: https://pymatgen.org/pymatgen.analysis.html.

    :param atoms_one:
        First structure.
    :type atoms_one: Atoms
    :param atoms_two:
        Second structure.
    :type atoms_two: Atoms
    :param ltol:
        Fractional length tolerance for pymatgen's StructureMatcher.
        Defaults to 0.2 (pymatgen's default).
    :type ltol: float
    :param stol:
        Site tolerance for pymatgen's StructureMatcher.
        Defaults to 0.3 (pymatgen's default).
    :type stol: float
    :param angle_tol:
        Angle tolerance in degrees for pymatgen's StructureMatcher.
        Defaults to 5.0 (pymatgen's default).
    :type angle_tol: float

    :return:
        Root-mean-square displacement between the two structures if they are the same, None otherwise.
    :rtype: Optional[float]
    """
    sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    # Conversion to pymatgen type.
    a1 = AseAtomsAdaptor.get_structure(atoms_one)
    a2 = AseAtomsAdaptor.get_structure(atoms_two)
    res = sm.get_rms_dist(a1, a2)
    assert res is None or res[0] <= stol
    return res[0] if res is not None else None


def _element_check(atoms_one: Atoms, atoms_two: Atoms) -> bool:
    """
    Check whether the two structures are of the same composition.

    Note that one of the structures could be a simple multiple of the other structure (e.g., storing C H_4 twice
    resulting in the species C_2 H_8). This method will still return True in this case. This is achieved by finding
    the element with the minimum number of occurrences in each structure and dividing all occurrences by that number.

    :param atoms_one:
        First structure.
    :type atoms_one: Atoms
    :param atoms_two:
        Second structure.
    :type atoms_two: Atoms

    :return:
        True if the structures are of the same composition, False otherwise.
    :rtype: bool
    """
    atoms_one_counts = np.bincount(atoms_one.numbers, minlength=MAX_ATOM_NUM)
    atoms_two_counts = np.bincount(atoms_two.numbers, minlength=MAX_ATOM_NUM)

    # Find the element with the minimum number of occurrences in each structure.
    atoms_one_min = np.min(atoms_one_counts[np.nonzero(atoms_one_counts > 0)])
    atoms_two_min = np.min(atoms_two_counts[np.nonzero(atoms_two_counts > 0)])

    return np.allclose(atoms_one_counts / atoms_one_min, atoms_two_counts / atoms_two_min)


def _get_match_and_rmsd(atoms_one: Atoms, list_atoms_two: List[Atoms], ltol: float, stol: float,
                        angle_tol: float) -> Tuple[bool, Optional[float]]:
    """
    Helper function to check whether the given first structure appears in the second list of structures by using
    pymatgen's StructureMatcher and, if so, to find the minimum root-mean-square displacement of the first structure to
    the closest structure in the second list of structures.

    If the first structure does not appear in the second list of structures, the minimum root-mean-square displacement
    is None.

    The root-mean-square displacement is normalized by (volume / number_sites) ** (1/3).

    This function is required for multiprocessing (see match_rate and unique_rate functions below).

    :param atoms_one:
        First structure.
    :type atoms_one: Atoms
    :param list_atoms_two:
        List of second structures.
    :type list_atoms_two: List[Atoms]
    :param ltol:
        Fractional length tolerance for pymatgen's StructureMatcher.
    :type ltol: float
    :param stol:
        Site tolerance for pymatgen's StructureMatcher.
    :type stol: float
    :param angle_tol:
        Angle tolerance in degrees for pymatgen's StructureMatcher.
    :type angle_tol: float

    :return:
        (Whether the first structure appears in the second list of structures, minimum root-mean-square displacement).
    :rtype: Tuple[bool, Optional[float]]
    """
    rms_dists = []
    for atoms_two in list_atoms_two:
        if _element_check(atoms_one, atoms_two):
            res = _structure_matcher(atoms_one, atoms_two, ltol=ltol, stol=stol, angle_tol=angle_tol)
            if res is not None:
                rms_dists.append(res)
    if len(rms_dists) > 0:
        if len(rms_dists) > 1:
            print(f"[WARNING] _check: Found {len(rms_dists)} matches in the reference list.")
        return True, min(rms_dists)
    return False, None


def match_rate_and_rmsd(atoms_list: Sequence[Atoms], ref_list: Sequence[Atoms], ltol: float = 0.2, stol: float = 0.3,
                        angle_tol: float = 5.0, full: bool = False) -> Tuple[float, float]:
    """
    Compute the rate of structures in the first sequence of atoms appearing in the second sequence atoms, and the
    mean root-mean-square displacement between the appearing structures.

    The averaged root-mean-square displacements are normalized by (volume / number_sites) ** (1/3).

    This method uses pymatgen's StructureMatcher to compare the structures (see
    https://pymatgen.org/pymatgen.analysis.html).

    :param atoms_list:
        First sequence of ase.Atoms instances.
    :type atoms_list: Sequence[Atoms]
    :param ref_list:
        Second sequence of ase.Atoms instances.
    :type ref_list: Sequence[Atoms]
    :param ltol:
        Fractional length tolerance for pymatgen's StructureMatcher.
        Defaults to 0.2 (pymatgen's default).
    :type ltol: float
    :param stol:
        Site tolerance for pymatgen's StructureMatcher.
        Defaults to 0.3 (pymatgen's default).
    :type stol: float
    :param angle_tol:
        Angle tolerance in degrees for pymatgen's StructureMatcher.
        Defaults to 5.0 (pymatgen's default).
    :type angle_tol: float
    :param full:
        If True, try to match every generated structure to every structure in the prediction dataset.
        If False,try to match every generated structure to the structure at the same index in the prediction dataset.
        Defaults to False.
    :type full: bool

    :return:
        (The match rate, the mean root-mean-square displacement between the appearing structures).
    :rtype: Tuple[float, float]
    """
    if full:
        with Pool() as p:
            # We cannot use lambda functions with Pool.map so we use (partial) global functions instead.
            cfunc = partial(_get_match_and_rmsd, list_atoms_two=ref_list, ltol=ltol, stol=stol, angle_tol=angle_tol)
            res = list(p.map(cfunc, atoms_list))
    else:
        if len(atoms_list) != len(ref_list):
            raise ValueError("The number of structures in the two lists must be equal.")
        res = [_get_match_and_rmsd(atoms_list[i], [ref_list[i]], ltol=ltol, stol=stol, angle_tol=angle_tol)
               for i in range(len(atoms_list))]
    assert all(r[1] is None for r in res if not r[0])
    match_count = sum(r[0] for r in res)
    mean_rmsd = np.mean([r[1] for r in res if r[0]])

    return match_count / len(atoms_list), float(mean_rmsd)


def _compare_pair(atoms: Tuple[Atoms, Atoms], ltol: float, stol: float, angle_tol: float) -> bool:
    """
    Helper function to check whether the two given structures match by using pymatgen's StructureMatcher.

    This function is required for multiprocessing (see unique_rate function below).

    :param atoms:
        Tuple of two structures.
    :type atoms: Tuple[Atoms, Atoms]
    :param ltol:
        Fractional length tolerance for pymatgen's StructureMatcher.
    :type ltol: float
    :param stol:
        Site tolerance for pymatgen's StructureMatcher.
    :type stol: float
    :param angle_tol:
        Angle tolerance in degrees for pymatgen's StructureMatcher.
    :type angle_tol: float

    :return:
        True if the structures are the same, False otherwise.
    :rtype: bool
    """
    return (_element_check(atoms[0], atoms[1])
            and (_structure_matcher(atoms[0], atoms[1], ltol=ltol, stol=stol, angle_tol=angle_tol) is not None))


def unique_rate(atoms_list: Sequence[Atoms], ltol: float = 0.2, stol: float = 0.3, angle_tol: float = 5.0) -> float:
    """
    Compute the rate of unique structures in the given sequence of atoms.

    This method uses pymatgen's StructureMatcher to compare the structures with the sequence (see
    https://pymatgen.org/pymatgen.analysis.html).

    :param atoms_list:
        Sequence of ase.Atoms instances.
    :type atoms_list: Sequence[Atoms]
    :param ltol:
        Fractional length tolerance for pymatgen's StructureMatcher.
        Defaults to 0.2 (pymatgen's default).
    :type ltol: float
    :param stol:
        Site tolerance for pymatgen's StructureMatcher.
        Defaults to 0.3 (pymatgen's default).
    :type stol: float
    :param angle_tol:
        Angle tolerance in degrees for pymatgen's StructureMatcher.
        Defaults to 5.0 (pymatgen's default).
    :type angle_tol: float
    """
    with Pool() as p:
        # We cannot use lambda functions with Pool.map so we use (partial) global functions instead.
        cfunc = partial(_compare_pair, ltol=ltol, stol=stol, angle_tol=angle_tol)
        # (len(atoms_list) * (len(ref_list) - 1) / 2) // os.cpu_count() would be optimal value for chunksize because it
        # distributes the same amount of work to all processes. However, the memory usage can become quite large.
        # Therefore, we cap the chunksize at 100000.
        matches = list(p.imap(
            cfunc,
            ((atoms_list[first_index], atoms_list[second_index])
             for first_index in range(len(atoms_list))
             for second_index in range(first_index + 1, len(atoms_list))),
            chunksize=max(min((len(atoms_list) * (len(atoms_list) - 1) // 2) // os.cpu_count(), 100000), 1)))

    assert len(matches) == len(atoms_list) * (len(atoms_list) - 1) // 2

    # matches basically stores the upper triangle of the match matrix (without the diagonal).
    # We now count the number of unique structures by returning the number of rows where all values are False.
    # This should automatically consider cases where the same structure appears multiple times because only the last
    # appearance is considered unique.
    unique_count = 0
    start_index = 0
    for first_index in range(len(atoms_list)):
        number_second_indices = len(atoms_list) - first_index - 1
        if all(not matches[start_index + i] for i in range(number_second_indices)):
            unique_count += 1
        start_index += number_second_indices

    return unique_count / len(atoms_list)
