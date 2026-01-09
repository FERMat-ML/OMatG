from collections import OrderedDict
import json
from math import log
from pathlib import Path
from typing import Optional, Sequence
import warnings
from ase import Atoms
from ase.io import write
from lightning.pytorch import Trainer
import lmdb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pickle
from pymatgen.core import Composition, Lattice, Structure
from scipy.stats import lognorm, wasserstein_distance
from sklearn.neighbors import KernelDensity
import tqdm
import torch
from torch_geometric.data import Data
from omg.omg_lightning import OMGLightning
from omg.datamodule import OMGDataset, OMGDataModule
from omg.globals import MAX_ATOM_NUM
from omg.sampler.minimum_permutation_distance import correct_for_minimum_permutation_distance
from omg.si.corrector import PeriodicBoundaryConditionsCorrector
from omg.utils import convert_ase_atoms_to_data, xyz_reader
from omg.analysis import (get_coordination_numbers, get_coordination_numbers_species, get_cov, get_space_group,
                          get_volume_frac, match_rmsds, metre_rmsds, ValidAtoms)


class OMGTrainer(Trainer):
    """
    Trainer for the omg models.

    Extends the PyTorch Lightning Trainer class to provide additional subcommands.

    Any initialization args and kwargs are passed down to the PyTorch Lightning Trainer constructor.

    :param args:
        Positional arguments to pass to the PyTorch Lightning Trainer constructor.
    :param kwargs:
        Keyword arguments to pass to the PyTorch Lightning Trainer constructor.
    """
    def __init__(self, *args, **kwargs) -> None:
        """Constructor of the OMGTrainer class."""
        super().__init__(*args, **kwargs)

    def visualize(self, model: OMGLightning, datamodule: OMGDataModule, xyz_file: str,
                  plot_name: str = "viz.pdf", skip_init: bool = False) -> None:
        """
        Plot and compare distributions over the prediction and generated dataset.

        This method plots and compares the following distributions:
        - Atomic numbers.
        - Volumes.
        - Number of atoms per structure.
        - Number of unique elements per structure.
        - Average coordination numbers.
        - Average coordination numbers per species.
        - Space group numbers.
        - Crystal systems.
        - Root-mean-square distances between the fractional coordinates in the initial structures and the final
          generated structures (only if skip_init=False).

        The space group number and crystal system of the generated structures are determined with and without variable
        precision (varprec). Furthermore, the symmetrized structures are written (together with the original
        unsymmetrized structures for direct comparison) to an XYZ file with the same stem as the final generated
        structures, but with "_symmetric" appended to the stem of the filename.

        Compare the distributions of the volume, the element composition, and the number of unique elements per
        structure in the test and generated dataset. Also, plot the root-mean-square distance between the fractional
        coordinates in the initial structures (sampled from rho_0) and the final generated structures (generated from
        rho_1).

        The generated structures are read from an XYZ file. The initial structures are assumed to be stored in an XYZ
        file with the same name as the final generated structures, but with "_init" appended to the stem of the
        filename. If the skip_init flag is set to True, the initial structures are not read and only distributions for
        the final generated structures are visualized.

        Also, plot the root-mean-square distance between the
        fractional coordinates in the initial structures and the final generated structures.

        :param model:
            OMG model (argument required and automatically passed by lightning CLI).
        :type model: OMGLightning
        :param datamodule:
            OMG datamodule (argument required and automatically passed by lightning CLI).
        :type datamodule: OMGDataModule
        :param xyz_file:
            XYZ file containing the generated structures.
            This argument has to be set on the command line.
        :type xyz_file: str
        :param plot_name:
            Filename for the plots.
            Defaults to "viz.pdf".
            This argument can be optionally set on the command line.
        :type plot_name: str
        :param skip_init:
            Whether to skip the initial structures (sampled from rho_0) in the visualization.
            If set to True, only the final generated structures (generated from rho_1) will be visualized.
            Defaults to False.
            This argument can be optionally set on the command line.
        :type skip_init: bool
        """
        final_file = Path(xyz_file)
        initial_file = final_file.with_stem(final_file.stem + "_init")
        symmetry_filename = final_file.with_stem(final_file.stem + "_symmetric")

        # Get atoms
        if not skip_init:
            init_atoms = xyz_reader(initial_file)
        else:
            init_atoms = None
        gen_atoms = xyz_reader(final_file)
        ref_atoms = self._load_dataset_atoms(datamodule.pred_dataset)

        # Plot data
        self._plot_to_pdf(ref_atoms, init_atoms, gen_atoms, plot_name, model.use_min_perm_dist, symmetry_filename)

    @staticmethod
    def _load_dataset_atoms(dataset: OMGDataset) -> list[Atoms]:
        """
        Load lmdb file atoms into a list of Atoms instances.

        :param dataset:
            Dataset to load atoms from.
        :type dataset: OMGDataset

        :return:
            List of Atoms instances.
        :rtype: list[Atoms]
        """
        all_ref_atoms = []
        for struc in tqdm.tqdm(dataset, desc="Loading test dataset"):
            assert len(struc.species) == struc.pos.shape[0]
            assert struc.pos.shape[1] == 3
            assert struc.cell[0].shape == (3, 3)
            if bool(struc.pos_is_fractional):
                atoms = Atoms(numbers=struc.species, scaled_positions=struc.pos, cell=struc.cell[0],
                              pbc=(True, True, True))
            else:
                atoms = Atoms(numbers=struc.species, positions=struc.pos, cell=struc.cell[0],
                              pbc=(True, True, True))
            all_ref_atoms.append(atoms)
        return all_ref_atoms

    @staticmethod
    def _plot_to_pdf(reference: Sequence[Atoms], initial: Optional[Sequence[Atoms]], generated: Sequence[Atoms],
                     plot_name: str, use_min_perm_dist: bool, symmetry_filename: Path) -> None:
        """
        Helper plotting method for the `visualize` method of the OMGTrainer class.

        See the `visualize` method for a description of the plots.

        :param reference:
            Reference test structures.
        :type reference: Sequence[Atoms]
        :param initial:
            Initial structures or None.
        :type initial: Optional[Sequence[Atoms]]
        :param generated:
            Generated structures.
        :type generated: Sequence[Atoms]
        :param plot_name:
            Filename for the plots.
        :type plot_name: str
        :param use_min_perm_dist:
            Whether the minimum permutation distance coupling was used during training.
        :type use_min_perm_dist: bool
        :param symmetry_filename:
            Filename for the storage of the symmetric structures.
        :type symmetry_filename: Path
        """
        fractional_coordinates_corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

        # Keep ASE Atoms versions of certain inputs
        reference_atoms = reference
        generated_atoms = generated

        # Convert to Data
        reference = convert_ase_atoms_to_data(reference)
        if initial is not None:
            initial = convert_ase_atoms_to_data(initial)
        generated = convert_ase_atoms_to_data(generated)

        # List of volumes of all test structures.
        ref_vol = []
        # Dictionary mapping atom number to occurrences of that atom number in all test structures.
        ref_nums = {}
        # Dictionary mapping number of unique elements in every test structure to occurrences of that number of
        # unique elements in all test structures.
        ref_n_types = {}
        # Dictionary mapping number of elements in every test structure to occurrences of that number of elements.
        ref_n_atoms = {}
        # List of average coordination numbers across all test structures.
        ref_avg_cn = []
        # Dictionary mapping from species to their list of coordination numbers in test structures.
        ref_cn_species = {}
        # Dictionary mapping from space-group numbers to their occurences in test structures.
        ref_sg = {}
        # Dictionary mapping from crystal systems to their ccurences in test structures.
        ref_crystal_sys = {}

        for i in range(1, MAX_ATOM_NUM + 1):
            ref_nums[i] = 0
        for i in range(len(reference.ptr) - 1):
            num = reference.species[reference.ptr[i]:reference.ptr[i + 1]]
            ref_vol.append(float(torch.abs(torch.det(reference.cell[i]))))
            n_type = len(set(int(n) for n in num))
            if n_type not in ref_n_types:
                ref_n_types[n_type] = 0
            ref_n_types[n_type] += 1
            for n in num:
                ref_nums[int(n)] += 1
            n_atom = len(num)
            if n_atom not in ref_n_atoms:
                ref_n_atoms[n_atom] = 0
            ref_n_atoms[n_atom] += 1
        assert sum(v for v in ref_n_types.values()) == len(reference.n_atoms)

        rand_root_mean_square_distances = []
        rand_pos_one = torch.rand_like(reference.pos)
        rand_pos_two = torch.rand_like(reference.pos)
        # Cell and species are not important here.
        rand_data_one = Data(pos=rand_pos_one, cell=reference.cell, species=reference.species, ptr=reference.ptr,
                             n_atoms=reference.n_atoms, batch=reference.batch)
        rand_data_two = Data(pos=rand_pos_two, cell=reference.cell, species=reference.species, ptr=reference.ptr,
                             n_atoms=reference.n_atoms, batch=reference.batch)
        if use_min_perm_dist:
            correct_for_minimum_permutation_distance(rand_data_one, rand_data_two, fractional_coordinates_corrector,
                                                     switch_species=False)
            rand_pos_one = rand_data_one.pos
            rand_pos_two = rand_data_two.pos
        rand_pos_prime = fractional_coordinates_corrector.unwrap(rand_pos_one, rand_pos_two)
        distances_squared = torch.sum((rand_pos_prime - rand_pos_one) ** 2, dim=-1)
        for i in range(len(reference.ptr) - 1):
            ds = distances_squared[reference.ptr[i]:reference.ptr[i + 1]]
            rand_root_mean_square_distances.append(float(torch.sqrt(ds.mean())))

        ref_root_mean_square_distances = []
        rand_pos = torch.rand_like(reference.pos)
        # Cell and species are not important here.
        rand_data = Data(pos=rand_pos, cell=reference.cell, species=reference.species, ptr=reference.ptr,
                         n_atoms=reference.n_atoms, batch=reference.batch)
        if use_min_perm_dist:
            correct_for_minimum_permutation_distance(rand_data, reference, fractional_coordinates_corrector,
                                                     switch_species=False)
            rand_pos = rand_data.pos
        rand_pos_prime = fractional_coordinates_corrector.unwrap(reference.pos, rand_pos)
        distances_squared = torch.sum((rand_pos_prime - reference.pos) ** 2, dim=-1)
        for i in range(len(reference.ptr) - 1):
            ds = distances_squared[reference.ptr[i]:reference.ptr[i + 1]]
            ref_root_mean_square_distances.append(float(torch.sqrt(ds.mean())))

        ref_sg_fail = 0
        for struc in reference_atoms:
            ref_avg_cn.append(np.mean(get_coordination_numbers(struc)))

            cn_dict = get_coordination_numbers_species(struc)
            for key, val in cn_dict.items():
                if key not in ref_cn_species:
                    ref_cn_species[key] = []
                ref_cn_species[key].extend(val)

            sg_group, sg_num, cs, _ = get_space_group(struc, var_prec=False)
            if sg_group is None:
                assert sg_num is None and cs is None
                ref_sg_fail += 1
            else:
                assert 1 <= sg_num <= 230
                if sg_num not in ref_sg:
                    ref_sg[sg_num] = 0
                ref_sg[sg_num] += 1
                if cs not in ref_crystal_sys:
                    ref_crystal_sys[cs] = 0
                ref_crystal_sys[cs] += 1
        print("Number of times space group identification failed for prediction dataset: "
              "{}/{}".format(ref_sg_fail, len(reference_atoms)))

        # List of volumes of all generated structures.
        vol = []
        # Dictionary mapping atom number to occurrences of that atom number in all generated structures.
        nums = {}
        # Dictionary mapping number of unique elements in every generated structure to occurrences of that number of
        # unique elements in all generated structures.
        n_types = {}
        # Dictionary mapping number of elements in every generated structure to occurrences of that number of elements.
        n_atoms = {}
        # List of average coordination numbers across all generated structures.
        avg_cn = []
        # Dictionary mapping from species to their list of coordination numbers in generated structures.
        cn_species = {}
        # Dictionary mapping from space-group numbers to their occurences in generated structures (var_prec=True).
        sg = {}
        # Dictionary mapping from crystal systems to their ccurences in generated structures (var_prec=True).
        crystal_sys = {}
        # Dictionary mapping from space-group numbers to their occurences in generated structures (var_prec=False).
        sg_F = {}
        # Dictionary mapping from crystal systems to their ccurences in generated structures (var_prec=False).
        crystal_sys_F = {}

        for i in range(1, MAX_ATOM_NUM + 1):
            nums[i] = 0
        for i in range(len(generated.ptr) - 1):
            num = generated.species[generated.ptr[i]:generated.ptr[i + 1]]
            vol.append(float(torch.abs(torch.det(generated.cell[i]))))
            n_type = len(set(int(n) for n in num))
            if n_type not in n_types:
                n_types[n_type] = 0
            n_types[n_type] += 1
            for n in num:
                nums[int(n)] += 1
            n_atom = len(num)
            if n_atom not in n_atoms:
                n_atoms[n_atom] = 0
            n_atoms[n_atom] += 1
        assert sum(v for v in n_types.values()) == len(generated.n_atoms)

        if initial is not None:
            traveled_root_mean_square_distances = []
            assert initial.pos.shape == generated.pos.shape
            # noinspection PyTypeChecker
            assert torch.all(initial.ptr == generated.ptr)
            generated_pos_prime = fractional_coordinates_corrector.unwrap(initial.pos, generated.pos)
            distances_squared = torch.sum((generated_pos_prime - initial.pos) ** 2, dim=-1)
            for i in range(len(generated.ptr) - 1):
                ds = distances_squared[generated.ptr[i]:generated.ptr[i + 1]]
                traveled_root_mean_square_distances.append(float(torch.sqrt(ds.mean())))

            root_mean_square_distances = []
            rand_pos = torch.rand_like(generated.pos)
            # Cell and species are not important here.
            rand_data = Data(pos=rand_pos, cell=generated.cell, species=generated.species, ptr=generated.ptr,
                             n_atoms=generated.n_atoms, batch=generated.batch)
            if use_min_perm_dist:
                correct_for_minimum_permutation_distance(rand_data, generated, fractional_coordinates_corrector,
                                                         switch_species=False)
                rand_pos = rand_data.pos
            rand_pos_prime = fractional_coordinates_corrector.unwrap(generated.pos, rand_pos)
            distances_squared = torch.sum((rand_pos_prime - generated.pos) ** 2, dim=-1)
            for i in range(len(generated.ptr) - 1):
                ds = distances_squared[generated.ptr[i]:generated.ptr[i + 1]]
                root_mean_square_distances.append(float(torch.sqrt(ds.mean())))

        sg_fail = 0
        sg_fail_F = 0
        for struc in generated_atoms:
            avg_cn.append(np.mean(get_coordination_numbers(struc)))

            cn_dict = get_coordination_numbers_species(struc)
            for key, val in cn_dict.items():
                if key not in cn_species:
                    cn_species[key] = []
                cn_species[key].extend(val)

            sg_group, sg_num, cs, sym_struc = get_space_group(struc, var_prec=True, angle_tolerance=-1.0)
            if sg_group is None:
                assert sg_num is None and cs is None
                sg_fail += 1
            else:
                assert 1 <= sg_num <= 230
                if sg_num not in sg:
                    sg[sg_num] = 0
                sg[sg_num] += 1
                if cs not in crystal_sys:
                    crystal_sys[cs] = 0
                crystal_sys[cs] += 1
                # Only write symmetric structures.
                if sg_num >= 3:
                    # Write original and symmetrized structures one after another for easier comparison.
                    write(str(symmetry_filename), struc, format='extxyz', append=True)
                    write(str(symmetry_filename), sym_struc, format='extxyz', append=True)

            # Testing with var_prec = False, with tolerances reasonable for DFT-relaxed structures.
            sg_group_F, sg_num_F, cs_F, sym_struc_F = get_space_group(struc, var_prec=False, symprec=1.0e-2,
                                                                      angle_tolerance=-1.0)
            if sg_group_F is None:
                assert sg_num_F is None and cs_F is None
                sg_fail_F += 1
            else:
                assert 1 <= sg_num_F <= 230
                if sg_num_F not in sg_F:
                    sg_F[sg_num_F] = 0
                sg_F[sg_num_F] += 1
                if cs_F not in crystal_sys_F:
                    crystal_sys_F[cs_F] = 0
                crystal_sys_F[cs_F] += 1
                # Only write symmetric structures.
                if sg_num_F >= 3:
                    # Write original and symmetrized structures one after another for easier comparison.
                    symmetry_filename_F = str(symmetry_filename.with_stem(symmetry_filename.stem + "_F"))
                    write(symmetry_filename_F, struc, format='extxyz', append=True)
                    write(symmetry_filename_F, sym_struc_F, format='extxyz', append=True)

        print("Number of times space group identification failed for generated dataset (var_prec = True): "
              "{}/{} total".format(sg_fail, len(generated_atoms)))
        print("Number of times space group identification failed for generated dataset (var_prec = False): "
              "{}/{} total".format(sg_fail_F, len(generated_atoms)))

        # Plot
        with PdfPages(plot_name) as pdf:
            # Plot Element distribution
            total_number_atoms = sum(v for v in nums.values())
            plt.bar([k for k in nums.keys()], [v / total_number_atoms for v in nums.values()], alpha=0.8,
                    label="Generated", color="blueviolet")
            total_number_atoms_ref = sum(v for v in ref_nums.values())
            plt.bar([k for k in ref_nums.keys()], [v / total_number_atoms_ref for v in ref_nums.values()], alpha=0.5,
                    label="Training", color="darkslategrey")
            plt.title("Fractional element composition")
            plt.xlabel("Atomic Number")
            plt.ylabel("Density")
            plt.legend()
            pdf.savefig()
            plt.close()

            # Plot Volume KDE
            # KernelDensity expects array of shape (n_samples, n_features).
            # We only have a single feature.
            bandwidth = np.std(ref_vol) * len(ref_vol) ** (-1 / 5)  # Scott's rule.
            ref_vol = np.array(ref_vol)[:, np.newaxis]
            vol = np.array(vol)[:, np.newaxis]
            min_volume = min(ref_vol.min(), vol.min())
            max_volume = max(ref_vol.max(), vol.max())
            x_d = np.linspace(min_volume - 1.0, max_volume + 1.0, 1000)[:, np.newaxis]
            kde_gt = KernelDensity(kernel="tophat", bandwidth=bandwidth).fit(ref_vol)
            log_density_gt = kde_gt.score_samples(x_d)
            kde_gen = KernelDensity(kernel="tophat", bandwidth=bandwidth).fit(vol)
            log_density_gen = kde_gen.score_samples(x_d)
            plt.plot(x_d, np.exp(log_density_gen), color="blueviolet", label="Generated")
            plt.plot(x_d, np.exp(log_density_gt), color="darkslategrey", label="Test")
            # plt.text(
            #    0.05, 0.95,
            #    f'KS Test for identical distributions: p-value={kstest(vol, ref_vol).pvalue}',
            #    verticalalignment='top',
            #    bbox=props,
            #    transform=plt.gca().transAxes
            # )
            plt.xlabel(r"Volume ($\AA^3$)")
            plt.ylabel("Density")
            plt.title("Volume")
            plt.legend()
            pdf.savefig()
            plt.close()

            # Plot N-atoms
            plt.bar([k for k in n_atoms.keys()], [v / len(generated.n_atoms) for v in n_atoms.values()], alpha=0.8,
                    label="Generated", color="blueviolet")
            plt.bar([k for k in ref_n_atoms.keys()], [v / len(reference.n_atoms) for v in ref_n_atoms.values()],
                    alpha=0.5, label="Test", color="darkslategrey")
            plt.xticks(ticks=np.arange(min(min(k for k in n_atoms.keys()),
                                           min(k for k in ref_n_atoms.keys())),
                                       max(max(k for k in n_atoms.keys()),
                                           max(k for k in ref_n_atoms.keys())),
                                       1))
            plt.title("Number of atoms")
            plt.xlabel("Number of atoms per structure")
            plt.ylabel("Density")
            plt.legend()
            pdf.savefig()
            plt.close()

            # Plot N-ary
            plt.bar([k for k in n_types.keys()], [v / len(generated.n_atoms) for v in n_types.values()], alpha=0.8,
                    label="Generated", color="blueviolet")
            plt.bar([k for k in ref_n_types.keys()], [v / len(reference.n_atoms) for v in ref_n_types.values()],
                    alpha=0.5, label="Test", color="darkslategrey")
            plt.xticks(ticks=np.arange(min(min(k for k in n_types.keys()),
                                           min(k for k in ref_n_types.keys())),
                                       max(max(k for k in n_types.keys()),
                                           max(k for k in ref_n_types.keys())),
                                       1))
            plt.title("N-ary")
            plt.xlabel("Unique elements per structure")
            plt.ylabel("Density")
            plt.legend()
            pdf.savefig()
            plt.close()

            if initial is not None:
                # Compute distributions for fractional coordinate movement.
                # Scott's rule for bandwidth.
                bandwidth = np.std(ref_root_mean_square_distances) * len(ref_root_mean_square_distances) ** (-1 / 5)
                ref_rmsds = np.array(ref_root_mean_square_distances)[:, np.newaxis]
                rmsds = np.array(root_mean_square_distances)[:, np.newaxis]
                trmsds = np.array(traveled_root_mean_square_distances)[:, np.newaxis]
                rand_rmsds = np.array(rand_root_mean_square_distances)[:, np.newaxis]
                x_d = np.linspace(0.0, (3 * 0.5 * 0.5) ** 0.5, 1000)[:, np.newaxis]
                kde_gt = KernelDensity(kernel='tophat', bandwidth=bandwidth).fit(ref_rmsds)
                log_density_gt = kde_gt.score_samples(x_d)
                kde_gen = KernelDensity(kernel='tophat', bandwidth=bandwidth).fit(rmsds)
                log_density_gen = kde_gen.score_samples(x_d)
                kde_traveled = KernelDensity(kernel='tophat', bandwidth=bandwidth).fit(trmsds)
                log_density_traveled = kde_traveled.score_samples(x_d)
                kde_rand = KernelDensity(kernel='tophat', bandwidth=bandwidth).fit(rand_rmsds)
                log_density_rand = kde_rand.score_samples(x_d)
                plt.plot(x_d, np.exp(log_density_gen), color="blueviolet", label="Generated")
                plt.plot(x_d, np.exp(log_density_gt), color="darkslategrey", label="Test")
                plt.plot(x_d, np.exp(log_density_traveled), color="cadetblue", label="Traveled")
                plt.plot(x_d, np.exp(log_density_rand), color="steelblue", label="Random")
                plt.xlabel("Root Mean Square Distance of Fractional Coordinates")
                plt.ylabel("Density")
                plt.legend()
                # plt.text(
                #    0.05, 0.95,
                #    f'KS Test for identical distributions: p-value={kstest(trmsds, trmsds).pvalue}',
                #    verticalalignment='top',
                #    bbox=props,
                #    transform=plt.gca().transAxes
                # )
                pdf.savefig()
                plt.close()

            # Compute distributions of structures by average coordination number
            # Plot avg cn KDE
            # KernelDensity expects array of shape (n_samples, n_features).
            # We only have a single feature.
            bandwidth = np.std(ref_avg_cn) * len(ref_avg_cn) ** (-1 / 5)  # Scott's rule.
            ref_avg_cn = np.array(ref_avg_cn)[:, np.newaxis]
            avg_cn = np.array(avg_cn)[:, np.newaxis]
            min_cn = min(ref_avg_cn.min(), avg_cn.min())
            max_cn = max(ref_avg_cn.max(), avg_cn.max())
            x_d = np.linspace(min_cn - 1.0, max_cn + 1.0, 1000)[:, np.newaxis]
            kde_gt = KernelDensity(kernel="tophat", bandwidth=bandwidth).fit(ref_avg_cn)
            log_density_gt = kde_gt.score_samples(x_d)
            kde_gen = KernelDensity(kernel="tophat", bandwidth=bandwidth).fit(avg_cn)
            log_density_gen = kde_gen.score_samples(x_d)
            plt.plot(x_d, np.exp(log_density_gen), color="blueviolet", label="Generated")
            plt.plot(x_d, np.exp(log_density_gt), color="darkslategrey", label="Test")
            plt.title("Average coordination number by structure")
            plt.xlabel("Average CN")
            plt.ylabel("Density")
            plt.legend()
            pdf.savefig()
            plt.close()

            # Compute distributions of average coordination number by species
            ref_avg_cn_species = {}
            avg_cn_species = {}
            for key, val in ref_cn_species.items():
                ref_avg_cn_species[key] = np.mean(val)
            for key, val in cn_species.items():
                avg_cn_species[key] = np.mean(val)

            species_order = Atoms(numbers=np.arange(1, MAX_ATOM_NUM + 1)).get_chemical_symbols()
            avg_cn_species = OrderedDict((key, avg_cn_species[key]) for key in species_order if key in avg_cn_species)
            ref_avg_cn_species = OrderedDict(
                (key, ref_avg_cn_species[key]) for key in species_order if key in ref_avg_cn_species)
            plt.bar([k for k in avg_cn_species.keys()], [v for v in avg_cn_species.values()], alpha=0.8,
                    label="Generated", color="blueviolet")
            plt.bar([k for k in ref_avg_cn_species.keys()], [v for v in ref_avg_cn_species.values()], alpha=0.5,
                    label="Test", color="darkslategrey")
            plt.xticks(rotation=75, ha='right', fontsize=4)
            plt.title("Average coordination number by species")
            plt.xlabel("Species")
            plt.ylabel("Average CN")
            plt.tight_layout()
            plt.legend()
            pdf.savefig()
            plt.close()

            # Compute distributions of space groups
            total_sg = sum(v for v in sg.values())
            plt.bar([k for k in sg.keys()], [v / total_sg for v in sg.values()], alpha=0.8,
                    label="Generated", color="blueviolet")
            total_sg_ref = sum(v for v in ref_sg.values())
            plt.bar([k for k in ref_sg.keys()], [v / total_sg_ref for v in ref_sg.values()], alpha=0.5,
                    label="Test", color="darkslategrey")
            plt.title("Space group distribution, varprec=True")
            plt.xlabel("Space group number")
            plt.ylabel("Density")
            plt.legend()
            pdf.savefig()
            plt.close()

            # Compute distributions of space groups
            total_sg_F = sum(v for v in sg_F.values())
            plt.bar([k for k in sg_F.keys()], [v / total_sg_F for v in sg_F.values()], alpha=0.8,
                    label="Generated", color="blueviolet")
            total_sg_ref = sum(v for v in ref_sg.values())
            plt.bar([k for k in ref_sg.keys()], [v / total_sg_ref for v in ref_sg.values()], alpha=0.5,
                    label="Test", color="darkslategrey")
            plt.title("Space group distribution, varprec=False")
            plt.xlabel("Space group number")
            plt.ylabel("Density")
            plt.legend()
            pdf.savefig()
            plt.close()

            # Compute distributions of crystal systems
            cs_order = ['Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Hexagonal', 'Cubic']
            crystal_sys_ord = OrderedDict((key, crystal_sys[key]) for key in cs_order if key in crystal_sys)
            ref_crystal_sys_ord = OrderedDict((key, ref_crystal_sys[key]) for key in cs_order if key in ref_crystal_sys)
            total_cs = sum(v for v in crystal_sys.values())
            plt.bar([k for k in crystal_sys_ord.keys()], [v / total_cs for v in crystal_sys_ord.values()], alpha=0.8,
                    label="Generated", color="blueviolet")
            total_cs_ref = sum(v for v in ref_crystal_sys.values())
            plt.bar([k for k in ref_crystal_sys_ord.keys()], [v / total_cs_ref for v in ref_crystal_sys_ord.values()],
                    alpha=0.5, label="Test", color="darkslategrey")
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.title("Crystal system distribution, varprec=True")
            plt.xlabel("Crystal system")
            plt.ylabel("Density")
            plt.tight_layout()
            plt.legend()
            pdf.savefig()
            plt.close()

            crystal_sys_ord_F = OrderedDict((key, crystal_sys_F[key]) for key in cs_order if key in crystal_sys_F)
            total_cs_F = sum(v for v in crystal_sys_F.values())
            plt.bar([k for k in crystal_sys_ord_F.keys()], [v / total_cs_F for v in crystal_sys_ord_F.values()],
                    alpha=0.8, label="Generated", color="blueviolet")
            plt.bar([k for k in ref_crystal_sys_ord.keys()], [v / total_cs_ref for v in ref_crystal_sys_ord.values()],
                    alpha=0.5, label="Test", color="darkslategrey")
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.title("Crystal system distribution, varprec=False")
            plt.xlabel("Crystal system")
            plt.ylabel("Density")
            plt.tight_layout()
            plt.legend()
            pdf.savefig()
            plt.close()

    def csp_metrics(self, model: OMGLightning, datamodule: OMGDataModule, xyz_file: str, skip_validation: bool = False,
                    skip_match: bool = False, ltol: float = 0.3, stol: float = 0.5, angle_tol: float = 10.0,
                    metre: bool = False, number_cpus: Optional[int] = None, upper_narity_limit: Optional[int] = None,
                    xyz_file_prediction_data: Optional[str] = None, check_reduced: bool = True,
                    result_name: str = "csp_metrics.json", plot_name: str = "rmsds.pdf") -> None:
        """
        Compute the crystal-structure prediction (CSP) metrics for the generated structures.

        By default, this method first validates the generated structures and the structures in the prediction dataset
        based on volume, structure, composition, and fingerprint checks (see ValidAtoms class), and calculates a match
        rate metric between the valid generated structures and the valid structures in the prediction dataset. The
        validation can be skipped by setting the `skip_validation` argument to True.

        The match rate metric is the fraction of matching crystal structures between the generated dataset and the
        prediction dataset. It is either one-to-one match rate or the match-everyone-to-reference (METRe) match rate.
        The one-to-one match rate matches structures at the same index in the generated dataset and the prediction
        dataset. The METRe match rate matches each reference structure to the best matching generated structure.

        This function also computes the mean root-mean-square distance (RMSE) between the matched structures. For the
        METRe metric, the best matching generated structure with the smallest root-mean-square distance is used.

        Finally, this function computes the corrected root-mean-square distance (cRMSE) between the matched structures.
        In this metric, a penalty is applied by using stol as the rmsd for non-matching structures (which are ignored
        in the mean for the standard root-mean-square distance).

        Structures are considered to match based on PyMatgen's StructureMatcher (see
        https://pymatgen.org/pymatgen.analysis.html). The default tolerances for the matcher are taken from CDVAE,
        DiffCSP, and FlowMM.

        The one-to-one match rate and the mean root-mean-square distance is one of the benchmarks for the
        crystal-structure prediction task used by CDVAE, DiffCSP, and FlowMM.

        The METRe and corrected root-mean-square distance metrics are introduced in https://arxiv.org/abs/2509.12178.
        If the prediction dataset contains polymorphs, the METRe option should be enabled.
        If there are no polymorphs, METRe will give the same match rate as the standard one-to-one match rate.
        If the prediction dataset contains duplicate structures, the METRe option should be disabled.

        This method also plots the histogram of the (uncorrected) root-mean-square distances between the matched
        structures.

        :param model:
            OMG model (argument required and automatically passed by lightning CLI).
        :type model: OMGLightning
        :param datamodule:
            OMG datamodule (argument required and automatically passed by lightning CLI).
        :type datamodule: OMGDataModule
        :param xyz_file:
            XYZ file containing the generated structures.
            This argument has to be set on the command line.
        :type xyz_file: str
        :param skip_validation:
            Whether to consider all structures in the generated and prediction dataset as valid.
            Defaults to False.
            This argument can be optionally set on the command line.
        :type skip_validation: bool
        :param skip_match:
            Whether to skip the calculation of the match rate.
            Defaults to False.
            This argument can be optionally set on the command line.
        :type skip_match: bool
        :param ltol:
            Fractional length tolerance for PyMatgen's StructureMatcher.
            Defaults to 0.3.
            This argument can be optionally set on the command line.
        :type ltol: float
        :param stol:
            Site tolerance for PyMatgen's StructureMatcher.
            Defaults to 0.5.
            This argument can be optionally set on the command line.
        :type stol: float
        :param angle_tol:
            Angle tolerance in degrees for PyMatgen's StructureMatcher.
            Defaults to 10.0.
            This argument can be optionally set on the command line.
        :type angle_tol: float
        :param metre:
            Whether the match-everyone-to-reference (METRe) metric is computed instead of standard one-to-one match
            rate.
            Defaults to False.
            This argument can be optionally set on the command line.
        :type metre: bool
        :param number_cpus:
            Number of CPUs to use for multiprocessing. If None, use os.cpu_count().
            Defaults to None.
            This argument can be optionally set on the command line.
        :type number_cpus: Optional[int]
        :param upper_narity_limit:
            The upper limit for the n-arity of the composition during validation check.
            If the number of unique elements in a structure exceeds this limit, the structure is considered invalid.
            Validation of structures with large n-arities is very slow so using this limit can speed up the validation
            process significantly.
            If None, no limit is set.
            Defaults to None.
        :type upper_narity_limit: Optional[int]
        :param xyz_file_prediction_data:
            XYZ file containing the prediction data structures.
            If None, the prediction data structures are loaded from the datamodule.
            Defaults to None.
            This argument can be optionally set on the command line.
        :type xyz_file_prediction_data: Optional[str]
        :param check_reduced:
            If True, two structures will be checked to match even if only their reduced compositions match. If False,
            the structures will be checked to match only if their full compositions match.
            Defaults to True.
            This argument can be optionally set on the command line.
        :type check_reduced: bool
        :param result_name:
            Name of the json file to save the match results.
            Defaults to "match.json".
            This argument can be optionally set on the command line.
        :type result_name: str
        :param plot_name:
            Name of the file to save the RMSD distribution plot.
            Defaults to "rmsds.pdf".
            This argument can be optionally set on the command line.
        :type plot_name: str

        :raises FileNotFoundError:
            If the prediction data file does not exist.
        :raises ValueError:
            If both `skip_validation` and `skip_match` are True.
            If the `result_name` does not end with .json.
        """
        if skip_validation and skip_match:
            raise ValueError("Everything is skipped, nothing to do.")

        final_file = Path(xyz_file)
        if not final_file.exists():
            raise FileNotFoundError(f"File {final_file} does not exist.")

        if not result_name.endswith(".json"):
            raise ValueError("The result_name must end with .json")

        # Get atoms
        gen_atoms = xyz_reader(final_file)
        if xyz_file_prediction_data is not None:
            test_file = Path(xyz_file_prediction_data)
            if not test_file.exists():
                raise FileNotFoundError(f"File {test_file} does not exist.")
            ref_atoms = xyz_reader(test_file)
        else:
            ref_atoms = self._load_dataset_atoms(datamodule.pred_dataset)

        gen_valid_atoms = ValidAtoms.get_valid_atoms(gen_atoms, desc="Validating generated structures",
                                                     skip_validation=skip_validation, number_cpus=number_cpus,
                                                     upper_narity_limit=upper_narity_limit)
        ref_valid_atoms = ValidAtoms.get_valid_atoms(ref_atoms, desc="Validating reference structures",
                                                     skip_validation=skip_validation, number_cpus=number_cpus,
                                                     upper_narity_limit=upper_narity_limit)

        if not skip_validation:
            print(f"Rate of valid structures in reference dataset: "
                  f"{100 * sum(va.valid for va in ref_valid_atoms) / len(ref_valid_atoms)}%.")
            print(f"Rate of valid structures in generated dataset: "
                  f"{100 * sum(va.valid for va in gen_valid_atoms) / len(gen_valid_atoms)}%.")

        if not skip_match:
            if not metre:
                fmr, frmsd, vmr, vrmsd, rmsds, val_rmsds, corr_rmsd, vcorr_rmsd = match_rmsds(
                    gen_valid_atoms, ref_valid_atoms, ltol=ltol, stol=stol, angle_tol=angle_tol,
                    number_cpus=number_cpus, check_reduced=check_reduced)
                filtered_rmsds = [rmsd for rmsd in rmsds if rmsd is not None]
                filtered_valid_rmsds = [rmsd for rmsd in val_rmsds if rmsd is not None]

                assert len(rmsds) == len(val_rmsds) == len(gen_valid_atoms)

                print(f"The match rate between all generated structures and the prediction dataset is "
                        f"{100.0 * fmr}%.")
                print(f"The mean root-mean-square distance, normalized by (V / N) ** (1/3), between all generated "
                      f"structures and the prediction dataset is {frmsd}.")
                print(f"The corrected root-mean-square distance, normalized by (V / N) ** (1/3), between all generated "
                      f"structures and the prediction dataset is {corr_rmsd}.")
                print()
                print(f"The match rate between valid generated structures and the valid prediction dataset is "
                        f"{100.0 * vmr}%.")
                print(f"The mean root-mean-square distance, normalized by (V / N) ** (1/3), between valid generated "
                      f"structures and the valid prediction dataset is {vrmsd}.")
                print(f"The corrected root-mean-square distance, normalized by (V / N) ** (1/3), between valid "
                      f"generated structures and the valid prediction dataset is {vcorr_rmsd}.")

                with open(result_name, "w") as f:
                    json.dump({
                        "match_rate": fmr,
                        "mean_RMSE": frmsd,
                        "mean_cRMSE": corr_rmsd,
                        "valid_match_rate": vmr,
                        "valid_mean_RMSE": vrmsd,
                        "valid_mean_cRMSE": vcorr_rmsd
                    }, f, indent=4)

            else:
                fmr, frmsd, vmr, vrmsd, rmsds, val_rmsds, corr_rmsd, vcorr_rmsd = metre_rmsds(
                    gen_valid_atoms, ref_valid_atoms, ltol=ltol, stol=stol, angle_tol=angle_tol,
                    number_cpus=number_cpus, check_reduced=check_reduced)
                filtered_rmsds = [rmsd for rmsd in rmsds if rmsd is not None]
                filtered_valid_rmsds = [rmsd for rmsd in val_rmsds if rmsd is not None]

                assert len(rmsds) == len(val_rmsds) == len(gen_valid_atoms)

                print(f"The match-everyone-to-reference (METRe) rate for all generated structures with respect to the "
                      f"prediction dataset is {100.0 * fmr}%.")
                print(f"The mean root-mean-square distance, normalized by (V / N) ** (1/3), for all generated "
                      f"structures with respect to the prediction dataset is {frmsd}.")
                print(f"The corrected root-mean-square distance, normalized by (V / N) ** (1/3), for all generated "
                      f"structures with respect to the prediction dataset is {corr_rmsd}.")
                print()
                print(f"The match-everyone-to-reference (METRe) rate for valid generated structures with respect to the"
                      f"valid prediction dataset is {100.0 * vmr}%.")
                print(f"The mean root-mean-square distance, normalized by (V / N) ** (1/3), for valid generated "
                      f"structures with respect to the valid prediction dataset is {vrmsd}.")
                print(f"The corrected root-mean-square distance, normalized by (V / N) ** (1/3), for valid generated "
                      f"structures with respect to the valid prediction dataset is {vcorr_rmsd}.")

                with open(result_name, "w") as f:
                    json.dump({
                        "METRe": fmr,
                        "mean_RMSE": frmsd,
                        "mean_cRMSE": corr_rmsd,
                        "valid_METRe": vmr,
                        "valid_mean_RMSE": vrmsd,
                        "valid_mean_cRMSE": vcorr_rmsd
                    }, f, indent=4)

            plt.figure()
            bandwidth = np.std(filtered_rmsds) * len(filtered_rmsds) ** (-1 / 5)  # Scott's rule.
            filtered_rmsds = np.array(filtered_rmsds)[:, np.newaxis]
            filtered_valid_rmsds = np.array(filtered_valid_rmsds)[:, np.newaxis]
            max_rmsd = max(filtered_rmsds.max(), filtered_valid_rmsds.max())
            x_d = np.linspace(0.0, max_rmsd + 0.1 * max_rmsd, 1000)[:, np.newaxis]
            kde = KernelDensity(kernel="tophat", bandwidth=bandwidth).fit(filtered_rmsds)
            log_density = kde.score_samples(x_d)
            kde_val = KernelDensity(kernel="tophat", bandwidth=bandwidth).fit(filtered_valid_rmsds)
            log_density_val = kde_val.score_samples(x_d)
            plt.plot(x_d, np.exp(log_density), color="blueviolet", label="All")
            plt.plot(x_d, np.exp(log_density_val), color="darkslategrey", label="Valid")
            plt.xlabel(r"RMSE distribution ($\AA^3$)")
            plt.ylabel("Density")
            plt.title("RMSE")
            plt.legend()
            plt.savefig(plot_name)
            plt.close()

    def dng_metrics(self, model: OMGLightning, datamodule: OMGDataModule, xyz_file: str,
                    dataset_name: Optional[str] = None, number_cpus: Optional[int] = None,
                    xyz_file_prediction_data: Optional[str] = None, result_name: str = "dng_metrics.json") -> None:
        """
        Compute the de-novo generation metrics for the generated structures.

        The metrics include validity (structural and compositional) and Wasserstein distances between distributions of
        density, volume fraction, number of atoms, number of unique elements, and average coordination number.

        In addition, if `dataset_name` is set to `mp_20`, `carbon_24`, or `perov_5`, the metrics include coverage recall
        and  precision.

        The computed metrics are part of the benchmarks for the de-novo generation task used by CDVAE, DiffCSP, and
        FlowMM.

        Note that stability related metrics can be computed, for example, with the MatterGen codebase (see
        https://github.com/microsoft/mattergen).

        :param model:
            OMG model (argument required and automatically passed by lightning CLI).
        :type model: OMGLightning
        :param datamodule:
            OMG datamodule (argument required and automatically passed by lightning CLI).
        :type datamodule: OMGDataModule
        :param xyz_file:
            XYZ file containing the generated structures.
            This argument has to be set on the command line.
        :type xyz_file: str
        :param dataset_name:
            Name of the dataset used for training.
            This is used to set the cutoffs for the coverage metrics.
            Coverage metrics are only computed for the datasets "mp_20", "carbon_24", and "perov_5".
            If None, no coverage metrics are computed.
            Defaults to None.
            This argument can be optionally set on the command line.
        :param number_cpus:
            Number of CPUs to use for multiprocessing during validation. If None, use os.cpu_count().
            Defaults to None.
            This argument can be optionally set on the command line.
        :type number_cpus: Optional[int]
        :param xyz_file_prediction_data:
            XYZ file containing the prediction data structures.
            If None, the prediction data structures are loaded from the datamodule.
            Defaults to None.
            This argument can be optionally set on the command line.
        :type xyz_file_prediction_data: Optional[str]
        :param result_name:
            Name of the json file to save the match results.
            Defaults to "dng_metrics.json".
            This argument can be optionally set on the command line.
        :type result_name: str

        :raises FileNotFoundError:
            If the file does not exist.
        """
        final_file = Path(xyz_file)
        if not final_file.exists():
            raise FileNotFoundError(f"File {final_file} does not exist.")

        if dataset_name is not None and dataset_name not in ("mp_20", "carbon_24", "perov_5"):
            warnings.warn("Coverage metrics can only be computed for the datasets 'mp_20', 'carbon_24', and "
                          "'perov_5'.")

        if not result_name.endswith(".json"):
            raise ValueError("The result_name must end with .json")

        # Get atoms
        gen_atoms = xyz_reader(final_file)
        if xyz_file_prediction_data is not None:
            test_file = Path(xyz_file_prediction_data)
            if not test_file.exists():
                raise FileNotFoundError(f"File {test_file} does not exist.")
            ref_atoms = xyz_reader(test_file)
        else:
            ref_atoms = self._load_dataset_atoms(datamodule.pred_dataset)

        gen_valid_atoms = ValidAtoms.get_valid_atoms(gen_atoms, desc="Validating generated structures",
                                                     number_cpus=number_cpus)
        ref_valid_atoms = ValidAtoms.get_valid_atoms(ref_atoms, desc="Validating reference structures",
                                                     number_cpus=number_cpus)

        # Validity rates.
        print("Validity metrics:")
        valid_rate = sum(va.valid for va in gen_valid_atoms) / len(gen_valid_atoms)
        print(f"{valid_rate=}")
        valid_comp_rate = sum(va.composition_valid for va in gen_valid_atoms) / len(gen_valid_atoms)
        print(f"{valid_comp_rate=}")
        valid_struc_rate = sum(va.structure_valid for va in gen_valid_atoms) / len(gen_valid_atoms)
        print(f"{valid_struc_rate=}")
        print()

        # Wasserstein distance metrics.
        print("Wasserstein distance metrics:")
        gen_densities = [struc.structure.density for struc in gen_valid_atoms]
        ref_densities = [struc.structure.density for struc in ref_valid_atoms]
        wdist_density = float(wasserstein_distance(gen_densities, ref_densities))
        print(f"{wdist_density=}")

        gen_volume_fractions = [get_volume_frac(struc.structure) for struc in gen_valid_atoms]
        ref_volume_fractions = [get_volume_frac(struc.structure) for struc in ref_valid_atoms]
        wdist_vol_frac = float(wasserstein_distance(gen_volume_fractions, ref_volume_fractions))
        print(f"{wdist_vol_frac=}")

        gen_number_atoms = [len(struc.structure.species) for struc in gen_valid_atoms]
        ref_number_atoms = [len(struc.structure.species) for struc in ref_valid_atoms]
        wdist_number_atoms = float(wasserstein_distance(gen_number_atoms, ref_number_atoms))
        print(f"{wdist_number_atoms=}")

        gen_narity = [len(set(struc.structure.species)) for struc in gen_valid_atoms]
        ref_narity = [len(set(struc.structure.species)) for struc in ref_valid_atoms]
        wdist_narity = float(wasserstein_distance(gen_narity, ref_narity))
        print(f"{wdist_narity=}")

        gen_coordination_numbers = [np.mean(get_coordination_numbers(struc.atoms)) for struc in gen_valid_atoms]
        ref_coordination_numbers = [np.mean(get_coordination_numbers(struc.atoms)) for struc in ref_valid_atoms]
        wdist_coordination_numbers = float(wasserstein_distance(gen_coordination_numbers, ref_coordination_numbers))
        print(f"{wdist_coordination_numbers=}")
        print()

        if dataset_name is not None and dataset_name in ("mp_20", "carbon_24", "perov_5"):
            # Taken from https://github.com/jiaor17/DiffCSP/blob/7121d159826efa2ba9500bf299250d96da37f146/scripts/compute_metrics.py
            COV_Cutoffs = {
                "mp_20": {"struc": 0.4, "comp": 10.0},
                "carbon_24": {"struc": 0.2, "comp": 4.0},
                "perov_5": {"struc": 0.2, "comp": 4}
            }
            assert dataset_name in COV_Cutoffs
            struc_cutoff = COV_Cutoffs[dataset_name]["struc"]
            comp_cutoff = COV_Cutoffs[dataset_name]["comp"]
            cov_recall, cov_precision = get_cov(gen_valid_atoms, ref_valid_atoms, struc_cutoff, comp_cutoff, None)
            cov_recall = float(cov_recall)
            cov_precision = float(cov_precision)
            print(f"Coverage metrics for {dataset_name}:")
            print(f"{cov_recall=}")
            print(f"{cov_precision=}")
        else:
            cov_recall, cov_precision = None, None

        with open(result_name, "w") as f:
            json.dump({
                "valid_rate": valid_rate,
                "valid_comp_rate": valid_comp_rate,
                "valid_struc_rate": valid_struc_rate,
                "wdist_density": wdist_density,
                "wdist_vol_frac": wdist_vol_frac,
                "wdist_number_atoms": wdist_number_atoms,
                "wdist_narity": wdist_narity,
                "wdist_coordination_numbers": wdist_coordination_numbers,
                "cov_recall": cov_recall,
                "cov_precision": cov_precision
            }, f, indent=4)

    def fit_lattice(self, model: OMGLightning, datamodule: OMGDataModule) -> None:
        """
        Fit a log-normal distribution to the lattice lengths of the training dataset.

        This yields the parameters for the informed lattice base distribution introduced by FlowMM.

        :param model:
            OMG model (argument required and automatically passed by lightning CLI).
        :type model: OMGLightning
        :param datamodule:
            OMG datamodule (argument required and automatically passed by lightning CLI).
        :type datamodule: OMGDataModule
        """
        dataset = datamodule.train_dataset
        atoms_list = self._load_dataset_atoms(dataset)
        a = []
        b = []
        c = []
        for structure in atoms_list:
            cellpar = structure.cell.cellpar()
            assert len(cellpar) == 6  # a, b, c, alpha, beta, gamma
            a.append(cellpar[0])
            b.append(cellpar[1])
            c.append(cellpar[2])
        shape_a, loc_a, scale_a = lognorm.fit(a, floc=0.0)
        shape_b, loc_b, scale_b = lognorm.fit(b, floc=0.0)
        shape_c, loc_c, scale_c = lognorm.fit(c, floc=0.0)
        assert loc_a == loc_b == loc_c == 0.0
        print("Standard deviations of the log of the distributions: ", shape_a, shape_b, shape_c)
        print("Means of the log of the distributions: ", log(scale_a), log(scale_b), log(scale_c))

    def create_compositions(self, model: OMGLightning, datamodule: OMGDataModule, compositions: Sequence[str] | str,
                            lmdb_file: str = "compositions.lmdb", repeats: int = 1) -> None:
        """
        Create an LMDB file containing dummy structures with the given compositions.

        :param model:
            OMG model (argument required and automatically passed by lightning CLI).
        :type model: OMGLightning
        :param datamodule:
            OMG datamodule (argument required and automatically passed by lightning CLI).
        :type datamodule: OMGDataModule
        :param compositions:
            List of compositions (as strings) to create dummy structures for.
            This argument has to be set on the command line.
        :type compositions: Sequence[str] | str
        :param lmdb_file:
            Name of the LMDB file to create.
            This argument has to be set on the command line.
        :type lmdb_file: str
        :param repeats:
            Number of times to repeat the given compositions.
            This argument can be optionally set on the command line.
            Defaults to 1.
        :type repeats: int

        :raises ValueError:
            If repeats is less than 1.
        :raises FileExistsError:
            If the LMDB file already exists.
        """
        if repeats < 1:
            raise ValueError("The number of repeats must be at least 1.")

        lmdb_path = Path(lmdb_file)
        if lmdb_path.exists():
            raise FileExistsError(f"The LMDB file {lmdb_file} already exists.")

        if isinstance(compositions, str):
            compositions = [compositions]

        structures = []
        for comp in compositions:
            pymatgen_composition = Composition(comp, strict=True)
            species = sum(([element] * int(amount) for element, amount in pymatgen_composition.items()), start=[])
            assert len(species) == pymatgen_composition.num_atoms
            dummy_lattice = Lattice.cubic(3)  # Tom's favorite number.
            dummy_fractional_coordinates = [(i / len(species), 0.0, 0.0) for i in range(len(species))]
            dummy_structure = Structure(dummy_lattice, species, dummy_fractional_coordinates,
                                        coords_are_cartesian=False)
            structures.append(dummy_structure)
        all_structures = sum((structures for _ in range(repeats)), start=[])

        with (lmdb.Environment(str(lmdb_path), subdir=False, map_size=int(1e12), lock=False) as env,
              env.begin(write=True) as txn):
            for idx, struc in tqdm.tqdm(enumerate(all_structures), desc=f"Saving {len(all_structures)} structures to {lmdb_path}"):
                data = {
                    "pos": torch.from_numpy(struc.cart_coords),
                    "cell": torch.from_numpy(np.array(struc.lattice.matrix)),
                    "atomic_numbers": torch.from_numpy(np.array(struc.atomic_numbers, dtype=np.int32)),
                    "ids": str(struc)
                }
                txn.put(str(idx).encode(), pickle.dumps(data))
