from typing import Dict, List, Set, Union
from ase import Atoms
from ase.io import read
from lightning.pytorch import Trainer
from lightning.pytorch.cli import LightningCLI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.neighbors import KernelDensity
from omg.omg import OMG
from omg.datamodule.dataloader import OMGDataModule
from omg.globals import MAX_ATOM_NUM


class OMGTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def visualize(self, model: OMG, datamodule: OMGDataModule, xyz_file: str, plot_name: str = "viz.pdf") -> None:
        """
        Compare the distributions of the volume, the element composition, and the number of unique elements per
        structure in the training and generated dataset.

        :param model:
            OMG model (argument required and automatically passed by lightning CLI).
        :type model: OMG
        :param datamodule:
            OMG datamodule (argument required and automatically passed by lightning CLI).
        :param xyz_file: 
            XYZ file containing the generated structures.
            Only this argument has to be set on the command line.
        :type xyz_file: str
        :param plot_name:
            Filename for the plots (defaults to viz.pdf).
            This argument can be optionally set on the command line.
        :type plot_name: str
        """

        # Get atoms
        gen_atoms = self._load_xyz_atoms(xyz_file)
        ref_atoms = self._load_dataset_atoms(datamodule.train_dataset)

        # Plot data
        self._plot_to_pdf(ref_atoms, gen_atoms, plot_name)

    @staticmethod
    def _load_xyz_atoms(xyz_file: str) -> Union[Atoms, List[Atoms]]:
        """
        Load xyz file of atoms.
        """
        atoms = read(xyz_file, index=':')
        return atoms

    @staticmethod
    def _load_dataset_atoms(dataset: str) -> List[Atoms]:
        """
        Load lmdb file atoms.
        """
        ref_atoms = []
        for element in dataset:
            assert len(element.species) == element.pos.shape[0]
            assert element.pos.shape[1] == 3
            assert element.cell[0].shape == (3, 3)
            atom = Atoms(
                element.species,
                positions=element.pos,
                cell=element.cell[0]
            )
            ref_atoms.append(atom)

        return ref_atoms

    @staticmethod
    def _plot_to_pdf(reference: list, generated: list, plot_name: str) -> None:
        """
        Plot figures for data analysis/matching
        between GT and Generated

        :param reference:
            Reference structures
        :type reference: list of ASE atoms
        :param generated:
            Generated structures
        :type generated: list of ASE atoms
        :param plot_name:
            Filename for the plots.
        :type plot_name: str
        """
        # List of volumes of all training structures.
        ref_vol = []
        # Dictionary mapping atom number to occurrences of that atom number in all training structures.
        ref_nums = {}
        # Dictionary mapping number of unique elements in every training structure to occurrences of that number of
        # unique elements in all training structures.
        ref_n_types = {}

        for i in range(1, MAX_ATOM_NUM + 1):
            ref_nums[i] = 0
        for a in reference:
            num = a.numbers
            ref_vol.append(a.get_volume())
            n_type = len(set(num))
            if n_type not in ref_n_types:
                ref_n_types[n_type] = 0
            ref_n_types[n_type] += 1
            for n in num:
                ref_nums[n] += 1
        assert sum(v for v in ref_n_types.values()) == len(reference)

        # List of volumes of all generated structures.
        vol = []
        # Dictionary mapping atom number to occurrences of that atom number in all generated structures.
        nums = {}
        # Dictionary mapping number of unique elements in every generated structure to occurrences of that number of
        # unique elements in all generated structures.
        n_types = {}
        for i in range(1, MAX_ATOM_NUM + 1):
            nums[i] = 0
        for a in generated:
            vol.append(a.get_volume())
            num = a.numbers
            n_type = len(set(num))
            if n_type not in n_types:
                n_types[n_type] = 0
            n_types[n_type] += 1
            for n in num:
                nums[n] += 1
        assert sum(v for v in n_types.values()) == len(generated)

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
            ref_vol = np.array(ref_vol)[:, np.newaxis]
            vol = np.array(vol)[:, np.newaxis]
            min_volume = min(ref_vol.min(), vol.min())
            max_volume = max(ref_vol.max(), vol.max())
            x_d = np.linspace(min_volume - 1.0, max_volume + 1.0, 1000)[:, np.newaxis]
            kde_gt = KernelDensity(kernel="tophat", bandwidth="scott").fit(ref_vol)
            log_density_gt = kde_gt.score_samples(x_d)
            kde_gen = KernelDensity(kernel="tophat", bandwidth="scott").fit(vol)
            log_density_gen = kde_gen.score_samples(x_d)
            plt.plot(x_d, np.exp(log_density_gen), color="blueviolet", label="Generated")
            plt.plot(x_d, np.exp(log_density_gt), color="darkslategrey", label="Training")
            plt.xlabel(r"Volume ($\AA^3$)")
            plt.ylabel("Density")
            plt.title("Volume")
            plt.legend()
            pdf.savefig()
            plt.close()

            # Plot N-ary
            plt.bar([k for k in n_types.keys()], [v / len(generated) for v in n_types.values()], alpha=0.8,
                    label="Generated", color="blueviolet")
            plt.bar([k for k in ref_n_types], [v / len(reference) for v in ref_n_types.values()], alpha=0.5,
                    label="Training", color="darkslategrey")
            plt.title("N-ary")
            plt.xlabel("Unique elements per structure")
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
