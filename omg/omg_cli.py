from pathlib import Path
from typing import Dict, List, Set
from ase import Atoms
from lightning.pytorch import Trainer
from lightning.pytorch.cli import LightningCLI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.neighbors import KernelDensity
import torch
from torch_geometric.data import Data
from omg.omg import OMG
from omg.datamodule.dataloader import OMGDataModule, OMGTorchDataset
from omg.globals import MAX_ATOM_NUM
from omg.sampler.minimum_permutation_distance import correct_for_min_perm_dist
from omg.si.corrector import PeriodicBoundaryConditionsCorrector
from omg.utils import convert_ase_atoms_to_data, xyz_reader


class OMGTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def visualize(self, model: OMG, datamodule: OMGDataModule, xyz_file: str, plot_name: str = "viz.pdf") -> None:
        """
        Compare the distributions of the volume, the element composition, and the number of unique elements per
        structure in the training and generated dataset. Also, plot the root mean-square distance between the fractional
        coordinates in the initial structures (sampled from rho_0) and the final generated structures (generated from
        rho_1).

        :param model:
            OMG model (argument required and automatically passed by lightning CLI).
        :type model: OMG
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

        # Get atoms
        init_atoms = xyz_reader(initial_file)
        gen_atoms = xyz_reader(final_file)
        ref_atoms = self._load_dataset_atoms(datamodule.train_dataset, datamodule.train_dataset.convert_to_fractional)

        # Plot data
        self._plot_to_pdf(ref_atoms, init_atoms, gen_atoms, plot_name, model.use_min_perm_dist)

    @staticmethod
    def _load_dataset_atoms(dataset: OMGTorchDataset, fractional: bool = True) -> Data:
        """
        Load lmdb file atoms.
        """
        all_ref_atoms = []
        for element in dataset:
            assert len(element.species) == element.pos.shape[0]
            assert element.pos.shape[1] == 3
            assert element.cell[0].shape == (3, 3)
            if fractional:
                atoms = Atoms(numbers=element.species, scaled_positions=element.pos, cell=element.cell[0],
                              pbc=(True, True, True))
            else:
                atoms = Atoms(numbers=element.species, positions=element.pos, cell=element.cell[0],
                              pbc=(True, True, True))
            all_ref_atoms.append(atoms)
        return convert_ase_atoms_to_data(all_ref_atoms)

    @staticmethod
    def _plot_to_pdf(reference: Data, initial: Data, generated: Data, plot_name: str, use_min_perm_dist: bool) -> None:
        """
        Plot figures for data analysis/matching between training and generated data.

        :param reference:
            Reference training structures.
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
        """
        fractional_coordinates_corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

        # List of volumes of all training structures.
        ref_vol = []
        # Dictionary mapping atom number to occurrences of that atom number in all training structures.
        ref_nums = {}
        # Dictionary mapping number of unique elements in every training structure to occurrences of that number of
        # unique elements in all training structures.
        ref_n_types = {}
        # Dictionary mapping number of elements in every training structure to occurrences of that number of elements.
        ref_n_atoms = {}

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

        ref_root_mean_square_distances = []
        rand_pos = torch.rand_like(reference.pos)
        # Cell and species are not important here.
        rand_data = Data(pos=rand_pos, cell=reference.cell, species=reference.species, ptr=reference.ptr,
                         n_atoms=reference.n_atoms, batch=reference.batch)
        if use_min_perm_dist:
            correct_for_min_perm_dist(rand_data, reference, fractional_coordinates_corrector)
            rand_pos = rand_data.pos
        rand_pos_prime = fractional_coordinates_corrector.unwrap(reference.pos, rand_pos)
        distances_squared = torch.sum((rand_pos_prime - reference.pos) ** 2, dim=-1)
        for i in range(len(reference.ptr) - 1):
            ds = distances_squared[reference.ptr[i]:reference.ptr[i + 1]]
            ref_root_mean_square_distances.append(float(torch.sqrt(ds.mean())))

        # List of volumes of all generated structures.
        vol = []
        # Dictionary mapping atom number to occurrences of that atom number in all generated structures.
        nums = {}
        # Dictionary mapping number of unique elements in every generated structure to occurrences of that number of
        # unique elements in all generated structures.
        n_types = {}
        # Dictionary mapping number of elements in every generated structure to occurrences of that number of elements.
        n_atoms = {}

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
            correct_for_min_perm_dist(rand_data, generated, fractional_coordinates_corrector)
            rand_pos = rand_data.pos
        rand_pos_prime = fractional_coordinates_corrector.unwrap(generated.pos, rand_pos)
        distances_squared = torch.sum((rand_pos_prime - generated.pos) ** 2, dim=-1)
        for i in range(len(generated.ptr) - 1):
            ds = distances_squared[generated.ptr[i]:generated.ptr[i + 1]]
            root_mean_square_distances.append(float(torch.sqrt(ds.mean())))

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
            plt.plot(x_d, np.exp(log_density_gt), color="darkslategrey", label="Training")
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
                    alpha=0.5, label="Training", color="darkslategrey")
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
                    alpha=0.5, label="Training", color="darkslategrey")
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
            x_d = np.linspace(0.0, (3 * 0.5 * 0.5) ** 0.5, 1000)[:, np.newaxis]
            kde_gt = KernelDensity(kernel='tophat', bandwidth=bandwidth).fit(ref_rmsds)
            log_density_gt = kde_gt.score_samples(x_d)
            kde_gen = KernelDensity(kernel='tophat', bandwidth=bandwidth).fit(rmsds)
            log_density_gen = kde_gen.score_samples(x_d)
            kde_traveled = KernelDensity(kernel='tophat', bandwidth=bandwidth).fit(trmsds)
            log_density_traveled = kde_traveled.score_samples(x_d)
            plt.plot(x_d, np.exp(log_density_gen), color="blueviolet", label="Generated")
            plt.plot(x_d, np.exp(log_density_gt), color="darkslategrey", label="Training")
            plt.plot(x_d, np.exp(log_density_traveled), color="cadetblue", label="Traveled")
            plt.xlabel("Root Mean Square Distance of Fractional Coordinates")
            plt.ylabel("Density")
            plt.legend()
            pdf.savefig()
            plt.close()


class OMGCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, trainer_class=OMGTrainer)

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        d = LightningCLI.subcommands()
        d["visualize"] = {"model", "datamodule"}
        return d
