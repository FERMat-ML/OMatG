from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Set
from ase import Atoms
from ase.io import write
from lightning.pytorch import Trainer
from lightning.pytorch.cli import LightningCLI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.neighbors import KernelDensity
import torch
from torch_geometric.data import Data
import yaml
from omg.omg_lightning import OMGLightning
from omg.datamodule.dataloader import OMGDataModule, OMGTorchDataset
from omg.globals import MAX_ATOM_NUM
from omg.sampler.minimum_permutation_distance import correct_for_minimum_permutation_distance
from omg.si.corrector import PeriodicBoundaryConditionsCorrector
from omg.utils import convert_ase_atoms_to_data, xyz_reader, DataField
from omg.analysis import (get_coordination_numbers, get_coordination_numbers_species, get_space_group,
                          match_rate_and_rmsd, unique_rate)


class OMGTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def visualize(self, model: OMGLightning, datamodule: OMGDataModule, xyz_file: str,
                  plot_name: str = "viz.pdf") -> None:
        """
        Compare the distributions of the volume, the element composition, and the number of unique elements per
        structure in the test and generated dataset. Also, plot the root mean-square distance between the fractional
        coordinates in the initial structures (sampled from rho_0) and the final generated structures (generated from
        rho_1).

        :param model:
            OMG model (argument required and automatically passed by lightning CLI).
        :type model: OMGLightning
        :param datamodule:
            OMG datamodule (argument required and automatically passed by lightning CLI).
        :param xyz_file:
            XYZ file containing the generated structures.
            This argument has to be set on the command line.
        :type xyz_file: str
        :param plot_name:
            Filename for the plots (defaults to viz.pdf).
            This argument can be optionally set on the command line.
        :type plot_name: str
        """
        final_file = Path(xyz_file)
        initial_file = final_file.with_stem(final_file.stem + "_init")
        symmetry_filename = final_file.with_stem(final_file.stem + "_symmetric")

        # Get atoms
        init_atoms = xyz_reader(initial_file)
        gen_atoms = xyz_reader(final_file)
        ref_atoms = self._load_dataset_atoms(datamodule.predict_dataset,
                                             datamodule.predict_dataset.convert_to_fractional)

        # Plot data
        self._plot_to_pdf(ref_atoms, init_atoms, gen_atoms, plot_name, model.use_min_perm_dist, symmetry_filename)

    @staticmethod
    def _load_dataset_atoms(dataset: OMGTorchDataset, fractional: bool = True) -> List[Atoms]:
        """
        Load lmdb file atoms into a list of Atoms instances.
        """
        all_ref_atoms = []
        for struc in dataset:
            assert len(struc.species) == struc.pos.shape[0]
            assert struc.pos.shape[1] == 3
            assert struc.cell[0].shape == (3, 3)
            if fractional:
                atoms = Atoms(numbers=struc.species, scaled_positions=struc.pos, cell=struc.cell[0],
                              pbc=(True, True, True))
            else:
                atoms = Atoms(numbers=struc.species, positions=struc.pos, cell=struc.cell[0],
                              pbc=(True, True, True))
            all_ref_atoms.append(atoms)
        return all_ref_atoms

    @staticmethod
    def _plot_to_pdf(reference: List[Atoms], initial: List[Atoms], generated: List[Atoms], plot_name: str,
                     use_min_perm_dist: bool, symmetry_filename: Path) -> None:
        """
        Plot figures for data analysis/matching between test and generated data.

        :param reference:
            Reference test structures.
        :type reference: List[Atoms]
        :param initial:
            Initial structures.
        :type initial: List[Atoms]
        :param generated:
            Generated structures.
        :type generated: List[Atoms]
        :param plot_name:
            Filename for the plots.
        :type plot_name: str
        :param use_min_perm_dist:
            Whether to use the minimum permutation distance.
        :type use_min_perm_dist: bool
        :param symmetry_filename:
            Filename for the symmetric structures.
        :type symmetry_filename: Path
        """
        fractional_coordinates_corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

        # Keep ASE Atoms versions of certain inputs
        reference_atoms = reference
        generated_atoms = generated

        # Convert to Data
        reference = convert_ase_atoms_to_data(reference)
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

        traveled_root_mean_square_distances = []
        assert initial.pos.shape == generated.pos.shape
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

    def match(self, model: OMGLightning, datamodule: OMGDataModule, xyz_file: str, full: bool = False) -> None:
        """
        Compute the match rate between the generated structures and the structures in the prediction dataset.

        This is one of the benchmarks for the crystal-structure prediction task used by DiffCSP and FlowMM.

        :param model:
            OMG model (argument required and automatically passed by lightning CLI).
        :type model: OMGLightning
        :param datamodule:
            OMG datamodule (argument required and automatically passed by lightning CLI).
        :param xyz_file:
            XYZ file containing the generated structures.
            This argument has to be set on the command line.
        :type xyz_file: str
        :param full:
            If True, try to match every generated structure to every structure in the prediction dataset.
            If False,try to match every generated structure to the structure at the same index in the prediction dataset.
            This argument can be optionally set on the command line.
            Defaults to False.
        :type full: bool

        :raises FileNotFoundError:
            If the file does not exist.
        """
        final_file = Path(xyz_file)
        if not final_file.exists():
            raise FileNotFoundError(f"File {final_file} does not exist.")

        # Get atoms
        gen_atoms = xyz_reader(final_file)
        ref_atoms = self._load_dataset_atoms(datamodule.predict_dataset,
                                             datamodule.predict_dataset.convert_to_fractional)

        # Tolerances from DiffCSP and FlowMM.
        mr, rmsd = match_rate_and_rmsd(gen_atoms, ref_atoms, ltol=0.3, stol=0.5, angle_tol=10.0, full=full)
        print(f"The match rate between the generated structures and dataset is {100 * mr}%.")
        print(f"The mean root-mean-square distance, normalized by (V / N) ** (1/3), between the generated structures "
              f"and dataset is {rmsd}.")

        r = unique_rate(gen_atoms, ltol=0.3, stol=0.5, angle_tol=10.0)
        print(f"The occurence of unique structures within the generated dataset is {100 * r}%.")

    def curriculum(self, model: OMGLightning, datamodule: OMGDataModule, lessons: List[str]) -> None:
        """
        Write a config file for a particular "lesson" in curriculum learning style whereby only the stochastic
        interpolants for the given data fields are kept intact. All other stochastic interpolants will be replaced
        with the identity interpolant.

        The possible lessons are "pos", "cell", and "species" or any combination of them.

        :param model:
            OMG model (argument required and automatically passed by lightning CLI).
        :type model: OMGLightning
        :param datamodule:
            OMG datamodule (argument required and automatically passed by lightning CLI).
        :type datamodule: OMGDataModule
        :param lessons:
            List of data fields for which the stochastic interpolants should be kept intact.
        :type lessons: List[str]

        :raises ValueError:
            If an invalid lesson is given.
        """
        # Since LightningCLI already expects the config command-line argument, there is apparently no way to make
        # it an argument of this function. Therefore, we just parse it from the given command-line arguments again.
        parser = ArgumentParser()
        parser.add_argument("--config", type=str)
        args, _ = parser.parse_known_args()
        config_file = args.config

        try:
            df_lessons = [DataField[lesson] for lesson in lessons]
        except KeyError:
            raise ValueError(f"Invalid lesson in {lessons}, allowed lessons are {[df.name for df in DataField]}")

        with open(config_file, 'r') as file:
            template = yaml.safe_load(file)

        # Get interpolant and swap with identity
        interpolants = template['model']['si']['init_args']['stochastic_interpolants']
        distributions = template['model']['sampler']['init_args']
        costs = template['model']['relative_si_costs']
        data_fields = template['model']['si']['init_args']['data_fields']

        for i, df in enumerate(data_fields):
            # This should not fail because the config file was already parsed.
            if DataField[df] not in df_lessons:
                distributions[f'{df}_distribution'] = {'class_path': 'omg.sampler.distributions.MirrorData'}
                interpolants[i] = {
                    'class_path': 'omg.si.single_stochastic_interpolant_identity.SingleStochasticInterpolantIdentity'}
                costs[i] = 0.0

        # Renormalize costs.
        new_total_costs = sum(costs)
        for i in range(len(costs)):
            costs[i] = costs[i] / new_total_costs

        # Save lesson.
        old_path = Path(config_file)
        new_filename = str(old_path.with_stem(old_path.stem + "_" + "_".join(lessons) + "_lesson"))
        with open(new_filename, 'w') as file:
            yaml.safe_dump(template, file)


class OMGCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, trainer_class=OMGTrainer)

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        d = LightningCLI.subcommands()
        d["visualize"] = {"model", "datamodule"}
        d["match"] = {"model", "datamodule"}
        d["curriculum"] = {"model", "datamodule"}
        return d
