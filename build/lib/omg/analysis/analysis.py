from collections import Counter
from functools import partial
import os
from typing import Dict, List, Optional, Sequence, Tuple
import warnings
from ase import Atoms
from ase.data import covalent_radii
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from scipy.spatial.distance import cdist
import spglib
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from omg.analysis.valid_atoms import ValidAtoms
from omg.globals import MAX_ATOM_NUM
from omg.utils import StandardScaler


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


def get_volume_frac(structure: Structure, num_density: bool = False) -> float:
    """
    Calculate the volume fraction or number density of the given structure.

    The volume density is defined as the number of atoms or volume of atoms in the structure divided by the volume of the unit cell.

    :param structure:
        Structure to calculate the volume fraction for. From pymatgen's Structure class.
    :type structure: Structure
    :param num_density:
        If True, the number density is calculated instead of the volume density.
        Defaults to False.
    :type num_density: bool

    :return:
        Volume fraction or number density of the structure.
    :rtype: float
    """
    if num_density:
        return structure.num_sites / structure.volume
    else:
        element_list = [elem.number for elem in structure.species]
        element_vol = [(4/3)*np.pi*covalent_radii[elem]**3 for elem in element_list]
    return sum(element_vol) / structure.volume


def get_cov(gen_atoms: List[ValidAtoms], ref_atoms: List[ValidAtoms], struc_cutoff: float, comp_cutoff: float,
            num_gen_crystals: Optional[int] = None) -> Tuple[float, float]:
    """ Adapted from DiffCSP code mostly, with some additions from FlowMM code. """

    CompScaler = StandardScaler(
        means=np.array([
            21.194441759304013, 58.20212663122281, 37.0076848719188, 36.52738520455582, 13.350626389725019,
            29.468922184630255, 28.71735137747704, 78.8868535524408, 50.16950217496375, 59.56764743604155,
            19.020429484306277, 61.335572740454325, 47.14515893344343, 141.75135923307818, 94.60620029962553,
            85.95794070476977, 34.07300576173523, 68.06189371516912, 637.9862061297893, 1817.2394155466848,
            1179.2532094169414, 1127.2743149568837, 431.51034284549826, 909.1060025135899, 3.7744320927984534,
            13.673707104881585, 9.899275012083132, 9.620186927095652, 3.8426065581251856, 9.96950217496375,
            3.305461575640406, 5.483035282745288, 2.1775737071048815, 4.215114560306594, 0.8206087101824266,
            3.732092798453359, 109.16732721121315, 179.5570323827936, 70.38970517158047, 136.0978305229613,
            27.027545809538527, 119.16713388110198, 1.2721433060967857, 2.4614001837260617, 1.1892568776289631,
            1.9844483610247092, 0.4691462290494881, 2.100143582306204, 1.4829869502174964, 1.9899951667472209,
            0.5070082165297245, 1.7956250375970633, 0.2056251946617602, 1.745867568873852, 0.05650072498791687,
            2.3618656355727405, 2.3053649105848235, 1.2829636137262992, 0.9995555685850794, 1.5150314161430642,
            0.7731271145480909, 7.4648139197680035, 6.691686805219913, 4.010677272036105, 2.612307566507693,
            3.303528274528758, 0.2739487675205413, 5.889753504108265, 5.615804736587724, 2.3244356612494683,
            2.1426251769710905, 1.4464475592073465, 4.739246012566457, 14.578395360077332, 9.839149347510874,
            9.413701584608935, 3.537059747455868, 8.550410826486225, 0.008119864668922184, 0.43286611889801835,
            0.4247462542290962, 0.16687837041055423, 0.17139889490813626, 0.10898985016916385, 0.06283228612856452,
            2.6573707104881583, 2.594538424359594, 1.219602938224228, 1.0596390454742999, 1.1120831319478008,
            0.14842919284678588, 3.8473658772353794, 3.6989366843885936, 1.4541605082183982, 1.3862277372859781,
            0.8018849685838569, 0.03542774287095215, 2.4474625422909617, 2.4120347994200095, 0.7745217539010397,
            0.9145812330586208, 0.3198646689221846, 1.552730787820203, 6.910681488641856, 5.357950700821653,
            3.615163570754227, 1.9072256165179793, 2.6702271628806185, 14.608536589568727, 34.83222477045747,
            20.223688180890715, 22.47901710732293, 7.17674504190757, 18.641837024143584, 0.009066988883518605,
            0.9185191396809959, 0.9094521507974755, 0.4368550481994018, 0.38905942883427047, 0.48375558240695804,
            0.0012985909686158003, 0.21708593995837092, 0.21578734898975546, 0.08167977375391729, 0.08155386250705281,
            0.06036340747305611, 116.32010633156113, 217.5905751570807, 101.27046882551957, 162.87154200548844,
            41.920624308665566, 136.4664572257129]),
        stds=np.array([
            16.35781741152948, 20.189540126474725, 20.516298414514758, 16.816765336550194, 7.966591328222124,
            22.270791076753067, 21.802116630115243, 12.804546460581966, 24.756629388687983, 13.930306216047477,
            10.214535652334533, 27.801612936980938, 39.74031558353379, 54.269739685575814, 53.70466607591569,
            42.852342044453444, 20.78341194242935, 56.28783510219931, 563.8004405882157, 732.0722574247563,
            736.2122907972664, 606.351603075103, 272.62646060896407, 810.6156779688841, 3.0362262146833428,
            3.2075174256751606, 4.0633818989245665, 2.9738244769894764, 1.7805586029644034, 5.643243225066782,
            1.1994336274579853, 0.8939013979423364, 1.2297581799896975, 1.0066021334519983, 0.49129747526397105,
            1.4159553146070951, 31.754756468836774, 28.054241463256226, 38.16336054795611, 25.83485338379922,
            15.388376641904662, 39.67137484594156, 0.31988340032011076, 0.6833658037760536, 0.7464197945553585,
            0.4881349085029781, 0.3176591553643101, 0.8601748146737138, 0.5864801661863596, 0.10048913710210677,
            0.5836289120986499, 0.2811748167435902, 0.2468696279341553, 0.5007375747433073, 0.37237566669029587,
            1.7235989187720187, 1.7058836077743305, 1.1558859351244697, 0.7677842566598179, 1.9203550253462733,
            2.1289400248865182, 3.5326064169848332, 3.708508303762512, 2.8709941136664567, 1.6110681295257014,
            4.310192504023775, 1.6644182118209292, 6.228287671164213, 6.1200848808512305, 3.1986202996110302,
            2.4492978142248867, 4.030497343977163, 3.662028270049814, 6.8192125550358345, 6.614243783887738,
            4.334987449618594, 2.568319610320196, 5.9494890200106925, 0.08974370432893491, 0.4954725441517777,
            0.494304434278516, 0.2309340434963803, 0.2072873961103969, 0.31162647950590266, 0.39805702757060923,
            1.8111691089355726, 1.7973395144505941, 0.9486995373104102, 0.7538753151875139, 1.5233177017753785,
            0.7952606701778913, 3.711190225170556, 3.638721437232604, 1.7171165424006831, 1.4307904413917036,
            2.1047820817622904, 0.49193748323158065, 4.064840532426175, 4.035286619587313, 1.4858577214526643,
            1.5799117659864677, 1.6130080156145745, 1.555249156140194, 4.776932951077492, 4.569790780459629,
            2.224617778217326, 1.7217507416156546, 2.5969733650703763, 7.215001918238936, 19.252513469778584,
            18.775394044177858, 9.447222764774764, 6.7467931836261235, 11.106825644766616, 0.27206794253092115,
            1.6449321034573106, 1.6236282792648686, 0.8506917026741503, 0.7020945355184042, 1.2281895279350408,
            0.04134438177238229, 0.5508855867341717, 0.5486095551438679, 0.24239297524046477, 0.2127779137935831,
            0.3036750942874694, 80.06063945615361, 21.345794811194104, 80.16475677581042, 52.58533928558554,
            35.40836791039412, 85.980205895116
        ]),
        replace_nan_token=0.)

    ref_comp_fps = [struc.composition_fingerprint for struc in ref_atoms]
    ref_struc_fps = [struc.structure_fingerprint for struc in ref_atoms]

    gen_comp_fps = [struc.composition_fingerprint for struc in gen_atoms]
    gen_struc_fps = [struc.structure_fingerprint for struc in gen_atoms]

    assert len(ref_struc_fps) == len(ref_comp_fps)
    assert len(gen_struc_fps) == len(gen_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(gen_struc_fps)

    filtered_gen_struc_fps, filtered_gen_comp_fps = [], []
    for struc_s, struc_c in zip(gen_struc_fps, gen_comp_fps):
        if struc_s is not None and struc_c is not None:
            filtered_gen_struc_fps.append(struc_s)
            filtered_gen_comp_fps.append(struc_c)

    ref_comp_fps = CompScaler.transform(ref_comp_fps)
    filtered_gen_comp_fps = CompScaler.transform(filtered_gen_comp_fps)

    ref_struc_fps = np.array(ref_struc_fps)
    filtered_gen_struc_fps = np.array(filtered_gen_struc_fps)
    ref_comp_fps = np.array(ref_comp_fps)
    filtered_gen_comp_fps = np.array(filtered_gen_comp_fps)

    struc_pdist = cdist(filtered_gen_struc_fps, ref_struc_fps)
    comp_pdist = cdist(filtered_gen_comp_fps, ref_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals

    return cov_recall, cov_precision


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


def get_space_group(atoms: Atoms, symprec: float = 1.0e-5, angle_tolerance: float = -1.0,
                    var_prec: bool = False) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[Atoms]]:
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
    :rtype: Tuple[Optional[str], Optional[int], Optional[str], Optional[Atoms]]
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


def _structure_matcher(structure_one: Structure, structure_two: Structure, ltol: float = 0.2, stol: float = 0.3,
                       angle_tol: float = 5.0) -> Optional[float]:
    """
    Checks if the two structures are the same by using pymatgen's StructureMatcher and, if so, return the
    root-mean-square displacement between the two structures. If the structures are different, return None.

    The root-mean-square displacement is normalized by (volume / number_sites) ** (1/3).

    The documentation of pymatgen's StructureMatcher can be found here: https://pymatgen.org/pymatgen.analysis.html.

    :param structure_one:
        First structure.
    :type structure_one: Structure
    :param structure_two:
        Second structure.
    :type structure_two: Structure
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
    res = sm.get_rms_dist(structure_one, structure_two)
    assert res is None or res[0] <= stol
    return res[0] if res is not None else None


def _element_check(atoms_one: Atoms, atoms_two: Atoms, check_reduced: bool) -> bool:
    """
    Check whether the two structures are of the same composition.

    Note that one of the structures could be a simple multiple of the other structure (e.g., storing C H_4 twice
    resulting in the species C_2 H_8). If check_reduced is True, this method will still return True in this case. This
    is achieved by finding the element with the minimum number of occurrences in each structure and dividing all
    occurrences by that number.

    :param atoms_one:
        First structure.
    :type atoms_one: Atoms
    :param atoms_two:
        Second structure.
    :type atoms_two: Atoms
    :param check_reduced:
        If True, the method will return True if the two structures are simple multiples of each other.
    :type check_reduced: bool

    :return:
        True if the structures are of the same composition, False otherwise.
    :rtype: bool
    """
    if check_reduced:
        atoms_one_counts = np.bincount(atoms_one.numbers, minlength=MAX_ATOM_NUM)
        atoms_two_counts = np.bincount(atoms_two.numbers, minlength=MAX_ATOM_NUM)

        # Find the element with the minimum number of occurrences in each structure.
        atoms_one_min = np.min(atoms_one_counts[np.nonzero(atoms_one_counts > 0)])
        atoms_two_min = np.min(atoms_two_counts[np.nonzero(atoms_two_counts > 0)])

        return np.allclose(atoms_one_counts / atoms_one_min, atoms_two_counts / atoms_two_min)
    else:
        return np.array_equal(np.sort(atoms_one.numbers), np.sort(atoms_two.numbers))


def _get_match_and_rmsd(atoms_one: ValidAtoms, atoms_two: ValidAtoms, ltol: float, stol: float,
                        angle_tol: float, check_reduced: bool) -> Optional[float]:
    """
    Helper function to check whether the given first structure matches the given second structure by using pymatgen's
    StructureMatcher and, if so, to find the root-mean-square displacement between them.

    Before checking two structures, this function first checks whether the two structures are of the same composition.
    If the two structures do not match, the root-mean-square displacement is None. If check_reduced is True, structures
    are checked even if they are simple multiples of each other.

    If the two structures do not match, the root-mean-square displacement is None.

    The root-mean-square displacement is normalized by (volume / number_sites) ** (1/3).

    This function is required for multiprocessing (see match_rate function below).

    :param atoms_one:
        First structure.
    :type atoms_one: ValidAtoms
    :param atoms_two:
        Second structure.
    :type atoms_two: ValidAtoms
    :param ltol:
        Fractional length tolerance for pymatgen's StructureMatcher.
    :type ltol: float
    :param stol:
        Site tolerance for pymatgen's StructureMatcher.
    :type stol: float
    :param angle_tol:
        Angle tolerance in degrees for pymatgen's StructureMatcher.
    :type angle_tol: float
    :param check_reduced:
        If True, the method will use pymatgen's StructureMatcher to check whether the two structures match even if their
        composition is a simple multiple of each other. If False, the method will first check whether the two structures
        are of the same composition.
    :type check_reduced: bool

    :return:
        Root-mean-square displacement.
    :rtype: Optional[float]
    """
    if _element_check(atoms_one.atoms, atoms_two.atoms, check_reduced):
        return _structure_matcher(atoms_one.structure, atoms_two.structure, ltol=ltol, stol=stol, angle_tol=angle_tol)
    return None


def _get_match_and_rmsd_sequence(atoms_one: ValidAtoms, sequence_atoms_two: Sequence[ValidAtoms], ltol: float,
                                 stol: float, angle_tol: float,
                                 check_reduced: bool) -> Optional[List[Tuple[float, int]]]:
    """
    Helper function to check whether the given first structure matches any of the structures in the given sequence of
    structures by using pymatgen's StructureMatcher and, if so, to find the root-mean-square displacements between them.

    This function returns a list of tuples each containing the root-mean-square displacement and the relevant index of
    the matching structure.

    Before checking two structures, this function first checks whether the two structures are of the same composition.
    If the two structures do not match, the root-mean-square displacement is None. If check_reduced is True, structures
    are checked even if they are simple multiples of each other.

    If the first structure does not match any of the structures in the sequence, this function returns None.

    The root-mean-square displacements are normalized by (volume / number_sites) ** (1/3).

    This function is required for the match-everyone-to-reference calculation (see metre_rmsds function below).

    :param atoms_one:
        First structure.
    :type atoms_one: ValidAtoms
    :param sequence_atoms_two:
        Sequence of structures.
    :type sequence_atoms_two: Sequence[ValidAtoms]
    :param ltol:
        Fractional length tolerance for pymatgen's StructureMatcher.
    :type ltol: float
    :param stol:
        Site tolerance for pymatgen's StructureMatcher.
    :type stol: float
    :param angle_tol:
        Angle tolerance in degrees for pymatgen's StructureMatcher.
    :type angle_tol: float
    :param check_reduced:
        If True, the method will use pymatgen's StructureMatcher to check whether the two structures match even if their
        composition is a simple multiple of each other. If False, the method will first check whether the two structures
        are of the same composition.
    :type check_reduced: bool

    :return:
        List of tuples each containing the root-mean-square displacement and the relevant index of the matching
        structure.
    :rtype: Optional[List[Tuple[float, int]]]
    """
    rmsds = []
    for index, atoms_two in enumerate(sequence_atoms_two):
        res = _get_match_and_rmsd(atoms_one, atoms_two, ltol, stol, angle_tol, check_reduced)
        if res is not None:
            rmsds.append((res, index))
    if len(rmsds) > 0:
        return rmsds
    else:
        return None


def match_rmsds(atoms_list: Sequence[ValidAtoms], ref_list: Sequence[ValidAtoms], ltol: float = 0.2,
                stol: float = 0.3, angle_tol: float = 5.0, number_cpus: Optional[int] = None,
                check_reduced: bool = True, enable_progress_bar: bool = True) -> Tuple[float, float, float, float,
                                                                                       List[Optional[float]],
                                                                                       List[Optional[float]], float,
                                                                                       float]:
    """
    Match the structures in the first sequence of validated atoms with the structures at the same index in the second
    sequence of validated atoms and return the root-mean-square displacements between the matching structures.
    The validity of structures is determined by the ValidAtoms class.

    The root-mean-square displacements are normalized by (volume / number_sites) ** (1/3).
    If the two structures do not match, the corresponding root-mean-square displacement is None, and it will be ignored
    in the mean root-mean-square displacement calculation. For the corrected root-mean-square displacement, if a
    structure does not match with any other structure, the corresponding corrected root-mean-square displacement is
    stol.

    This method uses PyMatgen's StructureMatcher to compare the structures (see
    https://pymatgen.org/pymatgen.analysis.html).

    Before comparing two structures, this function first checks whether the two structures are of the same composition.
    If the two compositions do not match, the structures do not match. If check_reduced is True, structures are checked
    even if they are simple multiples of each other.

    :param atoms_list:
        First sequence of validated atoms.
    :type atoms_list: Sequence[ValidAtoms]
    :param ref_list:
        Second sequence of validated atoms.
    :type ref_list: Sequence[ValidAtoms]
    :param ltol:
        Fractional length tolerance for PyMatgen's StructureMatcher.
        Defaults to 0.2 (PyMatgen's default).
    :type ltol: float
    :param stol:
        Site tolerance for PyMatgen's StructureMatcher.
        Defaults to 0.3 (PyMatgen's default).
    :type stol: float
    :param angle_tol:
        Angle tolerance in degrees for PyMatgen's StructureMatcher.
        Defaults to 5.0 (PyMatgen's default).
    :type angle_tol: float
    :param number_cpus:
        Number of CPUs to use for multiprocessing. If None, use os.cpu_count().
        Defaults to None.
    :type number_cpus: Optional[int]
    :param check_reduced:
        If True, structures are checked even if they are simple multiples of each other.
        Defaults to True.
    :type check_reduced: bool
    :param enable_progress_bar:
        If True, show a progress bar.
        Defaults to True.
    :type enable_progress_bar: bool

    :return:
        The match rate considering all structures,
        The standard mean root-mean-square displacement considering all structures,
        The match rate considering valid structures,
        The standard mean root-mean-square displacement considering valid structures,
        The list of rmsds measured with respect to the generated set,
        The list of valid rmsds measured with respect to the generated set,
        The corrected root-mean-square displacement associated with all structures,
        The corrected root-mean-square displacement associated with valid structures.
    :rtype: Tuple[float, float, float, float, List[Optional[float]], List[Optional[float]], float, float]

    :raises ValueError:
        If the number of structures in the first list is larger than the number of structures in the reference list.
        If the number of CPUs is not None and smaller than 1.
    """
    if len(atoms_list) != len(ref_list):
        warnings.warn("The number of structures in the generated atoms list differs from the number of atoms in "
                      "the reference list.")
    if len(atoms_list) > len(ref_list):
        raise ValueError("The number of structures in the generated atoms list is greater than the number of atoms in "
                         "the reference list.")
    if number_cpus is not None and number_cpus < 1:
        raise ValueError("The number of CPUs must be at least 1.")

    cpu_count = number_cpus if number_cpus is not None else os.cpu_count()

    # We cannot use lambda functions so we use (partial) global functions instead.
    match_func = partial(_get_match_and_rmsd, ltol=ltol, stol=stol, angle_tol=angle_tol,
                         check_reduced=check_reduced)
    if cpu_count > 1:
        res = process_map(match_func, atoms_list, ref_list, desc="Computing matches and RMSDs",
                          chunksize=max(min(len(atoms_list) // cpu_count, 100), 1), max_workers=cpu_count,
                          disable=not enable_progress_bar)
    else:
        # Apply tqdm to the shortest atoms list to show the correct progress bar.
        res = list(map(match_func, tqdm(atoms_list, desc="Computing matches and RMSDs",
                                        disable=not enable_progress_bar), ref_list))
    assert len(res) == len(atoms_list)

    valid_res = [r if atoms.valid and ref_atoms.valid else None
                 for r, atoms, ref_atoms in zip(res, atoms_list, ref_list)]

    # counting matches and RMSD
    match_count = sum(rmsd is not None for rmsd in res)
    filtered_rmsds = [rmsd for rmsd in res if rmsd is not None]
    mean_rmsd = np.mean(filtered_rmsds)

    # counting valid matches and RMSD
    valid_match_count = sum(rmsd is not None for rmsd in valid_res)
    filtered_valid_rmsds = [rmsd for rmsd in valid_res if rmsd is not None]
    valid_mean_rmsd = np.mean(filtered_valid_rmsds)

    # counting corrected RMSD in which non-matched structures are penalized
    corr_rmsds = [rmsd if rmsd is not None else stol for rmsd in res]
    corr_valid_rmsds = [rmsd if rmsd is not None else stol for rmsd in valid_res]
    corr_rmsd = np.mean(corr_rmsds)
    valid_corr_rmsd = np.mean(corr_valid_rmsds)

    return (match_count / len(atoms_list), float(mean_rmsd),
            valid_match_count / len(atoms_list), float(valid_mean_rmsd),
            res, valid_res,
            float(corr_rmsd),
            float(valid_corr_rmsd))


def metre_rmsds(atoms_list: Sequence[ValidAtoms], ref_list: Sequence[ValidAtoms], ltol: float = 0.2,
                stol: float = 0.3, angle_tol: float = 5.0, number_cpus: Optional[int] = None,
                check_reduced: bool = True, enable_progress_bar: bool = True) -> Tuple[float, float, float, float,
                                                                                       List[Optional[float]],
                                                                                       List[Optional[float]], float,
                                                                                       float]:
    """
    Match the structures in the first sequence of validated atoms with the structures at any index in the second
    sequence of validated atoms and return the minimum root-mean-square displacements between the matching structures.
    The validity of structures is determined by the ValidAtoms class.

    The root-mean-square displacements are normalized by (volume / number_sites) ** (1/3).
    If a structure does not match with any other structure, the corresponding root-mean-square displacement is None, and
    it will be ignored in the mean root-mean-square displacement calculation. For the corrected root-mean-square
    displacement, if a structure does not match with any other structure, the corresponding corrected root-mean-square
    displacement is stol.

    This method uses PyMatgen's StructureMatcher to compare the structures (see
    https://pymatgen.org/pymatgen.analysis.html).

    Before comparing two structures, this function first checks whether the two structures are of the same composition.
    If the two compositions do not match, the structures do not match. If check_reduced is True, structures are checked
    even if they are simple multiples of each other.

    :param atoms_list:
        First sequence of ase.Atoms instances.
    :type atoms_list: Sequence[ValidAtoms]
    :param ref_list:
        Second sequence of ase.Atoms instances.
    :type ref_list: Sequence[ValidAtoms]
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
    :param number_cpus:
        Number of CPUs to use for multiprocessing. If None, use os.cpu_count().
        Defaults to None.
    :type number_cpus: Optional[int]
    :param check_reduced:
        If True, structures are checked even if they are simple multiples of each other.
        Defaults to True.
    :type check_reduced: bool
    :param enable_progress_bar:
        If True, show a progress bar.
        Defaults to True.
    :type enable_progress_bar: bool

    :return:
        The match-everyone-to-reference rate considering all structures,
        The standard mean root-mean-square displacement considering all structures,
        The match-everyone-to-reference rate considering valid structures,
        The standard mean root-mean-square displacement considering valid structures,
        The list of rmsds measured with respect to the reference set.
        The list of valid rmsds measured with respect to the reference set.
        The corrected root-mean-square displacement associated with all structures,
        The corrected root-mean-square displacement associated with valid structures.
    :rtype: Tuple[float, float, float, float, List[Optional[float]], List[Optional[float]], float, float]

    :raises ValueError:
        If the number of CPUs is not None and smaller than 1.
    """
    if len(atoms_list) != len(ref_list):
        warnings.warn("The number of structures in the generated atoms list differs from the number of atoms in "
                      "the reference list.")
    if len(atoms_list) > len(ref_list):
        raise ValueError("The number of structures in the generated atoms list is greater than the number of atoms in "
                         "the reference list.")
    if number_cpus is not None and number_cpus < 1:
        raise ValueError("The number of CPUs must be at least 1.")

    cpu_count = number_cpus if number_cpus is not None else os.cpu_count()

    # We cannot use lambda functions so we use (partial) global functions instead.
    match_func = partial(_get_match_and_rmsd_sequence, sequence_atoms_two=ref_list, ltol=ltol, stol=stol,
                            angle_tol=angle_tol, check_reduced=check_reduced)
    if cpu_count > 1:
        res = process_map(match_func, atoms_list, desc="Computing METRe and RMSDs",
                            chunksize=max(min(len(atoms_list) // cpu_count, 100), 1), max_workers=cpu_count,
                            disable=not enable_progress_bar)
    else:
        res = list(map(match_func, tqdm(atoms_list, desc="Computing METRe and RMSDs", disable=not enable_progress_bar)))
    assert len(res) == len(atoms_list)

    rmsds_ref = [None for _ in range(len(ref_list))]
    valid_rmsds_ref = [None for _ in range(len(ref_list))]
    for r, atoms in zip(res, atoms_list):
        if r is not None:
            for rmsd, index in r:  # index is in ref_list
                assert rmsd is not None and index is not None
                if rmsds_ref[index] is None:
                    rmsds_ref[index] = rmsd  # add if there are no other entries
                elif rmsd < rmsds_ref[index]:
                    rmsds_ref[index] = rmsd  # update if this is the best match
                # repeat but if the structures are valid
                if atoms.valid and ref_list[index].valid:
                    if valid_rmsds_ref[index] is None:
                        valid_rmsds_ref[index] = rmsd
                    elif rmsd < valid_rmsds_ref[index]:
                        valid_rmsds_ref[index] = rmsd

    # counting matches and RMSD
    match_count = sum(rmsd is not None for rmsd in rmsds_ref)
    filtered_rmsds = [rmsd for rmsd in rmsds_ref if rmsd is not None]
    mean_rmsd = np.mean(filtered_rmsds)

    # counting valid matches and RMSD
    valid_match_count = sum(rmsd is not None for rmsd in valid_rmsds_ref)
    filtered_valid_rmsds = [rmsd for rmsd in valid_rmsds_ref if rmsd is not None]
    valid_mean_rmsd = np.mean(filtered_valid_rmsds)

    # counting corrected RMSD in which non-matched structures are penalized
    corr_rmsds = [rmsd if rmsd is not None else stol for rmsd in rmsds_ref]
    corr_valid_rmsds = [rmsd if rmsd is not None else stol for rmsd in valid_rmsds_ref]
    corr_rmsd = np.mean(corr_rmsds)
    valid_corr_rmsd = np.mean(corr_valid_rmsds)

    return (match_count / len(ref_list), float(mean_rmsd),
            valid_match_count / len(ref_list), float(valid_mean_rmsd),
            rmsds_ref, valid_rmsds_ref,
            float(corr_rmsd),
            float(valid_corr_rmsd))
