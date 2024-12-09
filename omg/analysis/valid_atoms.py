from functools import partial
import os
from typing import List, Optional, Sequence, Tuple
import warnings
from ase import Atoms
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
import numpy as np
from pymatgen.core import Composition, Structure
from pymatgen.io.ase import AseAtomsAdaptor
import smact.screening
from tqdm.contrib.concurrent import process_map


class ValidAtoms(object):
    """
    Validate an Atoms object based on volume, structure, composition, and fingerprint checks.

    This class is inspired by CDVAE's/DiffCSP's crystal class (see
    https://github.com/jiaor17/DiffCSP/blob/main/scripts/compute_metrics.py).

    The volume is considered valid if it is bigger than a given cutoff (default: 0.1 Å^3).

    The structure is considered valid if all pairwise distances are larger than a given cutoff (default: 0.5 Å).

    The composition is considered valid if it is valid according to the SMACT rules.

    This class also computes the CrystalNN structural fingerprints and the normalized Magpie compositional fingerprints.
    If the computation of the fingerprints fails, the atoms are considered invalid.

    :param atoms:
        The Atoms instance to validate.
    :type atoms: Atoms
    :param volume_check_cutoff:
        The cutoff for the volume check (in Å^3).
        Defaults to 0.1.
    :type volume_check_cutoff: float
    :param structure_check_cutoff:
        The cutoff for the structure check (in angstroms).
        Defaults to 0.5.
    :type structure_check_cutoff: float
    :param use_pauling_test:
        Whether to use the Pauling test for the composition check.
        Defaults to True.
    :type use_pauling_test: bool
    :param include_alloys:
        Whether to include alloys in the composition check.
        Defaults to True.
    :type include_alloys: bool
    """

    _CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
    with warnings.catch_warnings(action="ignore"):
        _CompFP = ElementProperty.from_preset("magpie")

    def __init__(self, atoms: Atoms, volume_check_cutoff: float = 0.1, structure_check_cutoff: float = 0.5,
                 use_pauling_test: bool = True, include_alloys: bool = True) -> None:
        """Constructor of the ValidAtoms class."""
        self._atoms = atoms
        self._structure = AseAtomsAdaptor.get_structure(atoms)
        self._composition = self._structure.composition
        self._structure_check_cutoff = structure_check_cutoff
        self._use_pauling_test = use_pauling_test
        self._include_alloys = include_alloys
        self._volume_valid = self._structure.volume > volume_check_cutoff
        self._structure_valid = self._structure_check(self._structure, self._structure_check_cutoff)
        self._composition_valid = self._smact_check(self._composition, self._use_pauling_test, self._include_alloys)
        with warnings.catch_warnings(action="ignore"):
            try:
                self._composition_fingerprint, self._structure_fingerprint = self._get_fingerprints(self._composition,
                                                                                                    self._structure)
                self._fingerprint_valid = True
            except ValueError:
                self._composition_fingerprint, self._structure_fingerprint = None, None
                self._fingerprint_valid = False

    @staticmethod
    def get_valid_atoms(atoms: Sequence[Atoms], structure_check_cutoff: float = 0.5, use_pauling_test: bool = True,
                        include_alloys: bool = True, desc: Optional[str] = None) -> List["ValidAtoms"]:
        """
        Generate a list of ValidAtoms instances from a list of Atoms instances in parallel.

        :param atoms:
            The list of Atoms instances to validate.
        :type atoms: Iterable[Atoms]
        :param structure_check_cutoff:
            The cutoff for the structure check (in angstroms).
            Defaults to 0.5.
        :type structure_check_cutoff: float
        :param use_pauling_test:
            Whether to use the Pauling test for the composition check.
            Defaults to True.
        :type use_pauling_test: bool
        :param include_alloys:
            Whether to include alloys in the composition check.
            Defaults to True.
        :type include_alloys: bool
        :param desc:
            The description for the progress bar.
            If None, no description is shown.
            Defaults to None.
        :type desc: str

        :return:
            The list of ValidAtoms instances.
        :rtype: List[ValidAtoms]
        """
        constructor = partial(ValidAtoms, structure_check_cutoff=structure_check_cutoff,
                              use_pauling_test=use_pauling_test, include_alloys=include_alloys)
        valid_atoms = process_map(constructor, atoms, desc=desc,
                                  chunksize=max(min(len(atoms) // os.cpu_count(), 100), 1))
        return valid_atoms

    @staticmethod
    def _structure_check(structure: Structure, cutoff: float = 0.5) -> bool:
        """
        Check the validity of the structure.

        The structure is considered valid if all pairwise distances are larger than a given cutoff (default: 0.5Å).

        :param structure:
            The Structure instance to validate.
        :type structure: Structure
        :param cutoff:
            The cutoff for the structure check (in angstroms).
            Defaults to 0.5.
        :type cutoff: float

        :return:
            Whether the structure is valid.
        :rtype: bool
        """
        dist_mat = structure.distance_matrix
        # Pad diagonal with a large number
        dist_mat = dist_mat + np.diag(
            np.ones(dist_mat.shape[0]) * (cutoff + 10.))
        if dist_mat.min() < cutoff or structure.volume < 0.1:
            return False
        else:
            return True

    @staticmethod
    def _smact_check(composition: Composition, use_pauling_test: bool = True, include_alloys: bool = True) -> bool:
        """
        Check the validity of the composition according to the SMACT rules.

        :param composition:
            The Composition instance to validate.
        :type composition: Composition
        :param use_pauling_test:
            Whether to use the Pauling test for the composition check.
            Defaults to True.
        :type use_pauling_test: bool
        :param include_alloys:
            Whether to include alloys in the composition check.
            Defaults to True.
        :type include_alloys: bool

        :return:
            Whether the composition is valid.
        :rtype: bool
        """
        return smact.screening.smact_validity(composition, use_pauling_test=use_pauling_test,
                                              include_alloys=include_alloys)

    @staticmethod
    def _get_fingerprints(composition: Composition, structure: Structure) -> Tuple[List[float], List[float]]:
        """
        Compute the normalized Magpie compositional and CrystalNN structural fingerprints.

        :param composition:
            The Composition instance.
        :type composition: Composition
        :param structure:
            The Structure instance.
        :type structure: Structure

        :return:
            (The compositional fingerprint, the structural fingerprint).
        """
        comp_fp = ValidAtoms._CompFP.featurize(composition)
        site_fps = [ValidAtoms._CrystalNNFP.featurize(structure, i) for i in range(len(structure))]
        struct_fp = list(np.array(site_fps).mean(axis=0))
        return comp_fp, struct_fp

    @property
    def atoms(self) -> Atoms:
        """
        Returns the validated Atoms instance.

        :return:
            The validated Atoms instance.
        :rtype: Atoms
        """
        return self._atoms

    @property
    def structure(self) -> Structure:
        """
        Returns the Structure instance corresponding to the validated Atoms instance.

        :return:
            The Structure instance.
        :rtype: Structure
        """
        return self._structure

    @property
    def volume_valid(self) -> bool:
        """
        Returns the volume validity of the Atoms instance.

        :return:
            The volume validity.
        :rtype: bool
        """
        return self._volume_valid

    @property
    def structure_valid(self) -> bool:
        """
        Returns the structural validity of the Atoms instance.

        :return:
            The structural validity.
        :rtype: bool
        """
        return self._structure_valid

    @property
    def composition_valid(self) -> bool:
        """
        Returns the compositional validity of the Atoms instance.

        :return:
            The compositional validity.
        :rtype: bool
        """
        return self._composition_valid

    @property
    def fingerprint_valid(self) -> bool:
        """
        Returns the fingerprint validity of the Atoms instance.

        :return:
            The fingerprint validity.
        :rtype: bool
        """
        return self._fingerprint_valid

    @property
    def valid(self) -> bool:
        """
        Returns the validity of the Atoms instance.

        :return:
            The validity.
        :rtype: bool
        """
        return self._volume_valid and self._structure_valid and self._composition_valid and self._fingerprint_valid

    @property
    def composition_fingerprint(self) -> Optional[List[float]]:
        """
        Returns the normalized Magpie compositional fingerprint of the Atoms instance.

        If the computation of the fingerprint failed, None is returned.

        :return:
            The compositional fingerprint.
        :rtype: Optional[List[float]]
        """
        return self._composition_fingerprint

    @property
    def structure_fingerprint(self) -> Optional[List[float]]:
        """
        Returns the CrystalNN structural fingerprint of the Atoms instance.

        If the computation of the fingerprint failed, None is returned.

        :return:
            The structural fingerprint.
        :rtype: Optional[List[float]]
        """
        return self._structure_fingerprint
