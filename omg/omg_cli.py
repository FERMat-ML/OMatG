import torch
import numpy as np
from sklearn.neighbors import KernelDensity
from lightning.pytorch.cli import LightningCLI
import lmdb
import pickle as pkl
from ase import Atoms
from ase.io import read
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, Set
from lightning.pytorch import Trainer

class OMG_Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def visualize(self, xyz_file:str, **kwargs):
        '''
        Visualize dataset vs generated data

        :param xyz_file: 
            xyz file of sampled atoms
        :type xyz_file: str
        '''

        # Get atoms
        gen_atoms = self._load_xyz_atoms(xyz_file)
        ref_atoms = self._load_dataset_atoms(kwargs['datamodule'].train_dataset)

        # Plot data
        self._plot_to_pdf(ref_atoms, gen_atoms)

    def _load_xyz_atoms(self, xyz_file:str):
        '''
        Load xyz file of atoms
        '''
        atoms = read(xyz_file, index=':')
        return atoms

    def _load_dataset_atoms(self, dataset:str):
        '''
        Load lmdb file atoms
        '''
        ref_atoms = []
        for element in dataset:
            atom = Atoms(
                element.species,
                positions=element.pos,
                cell=element.cell[0]
            )
            ref_atoms.append(atom)

        return ref_atoms

    def _plot_to_pdf(self, reference:list, generated:list, plot_name:str='viz.pdf'):
        '''
        Plot figures for data analysis/matching
        between GT and Generated

        :param reference:
            Reference structures
        :type reference: list of ASE atoms
        :param generated:
            Generated structures
        :type generated: list of ASE atoms
        '''

        ref_vol = []
        ref_nums = {}
        ref_n_types = {}
        ref_num_atoms = 0

        for i in range(1,7):
            ref_n_types[i] = 0
        for i in range(1,110):
            ref_nums[i] = 0
        for a in reference:
            ref_num_atoms += len(a)
            num = a.numbers
            ref_vol.append(a.get_volume())
            n_type = len(list(set(num)))
            if n_type > 6:
                n_type = 6
            ref_n_types[n_type] += 1
            for n in num:
                ref_nums[n] +=1

        ref_h = []
        ref_n = []
        for k,v in ref_n_types.items():
            ref_n.append(v)
        for k,v in ref_nums.items():
            ref_h.append(v) 

        # Generated data
        vol = []
        nums = {}
        n_types = {}
        num_atoms = 0
        for i in range(1,7):
            n_types[i] = 0
        for i in range(1,110):
            nums[i] = 0
        for a in generated:
            v = a.get_volume()
            num_atoms += len(a)
            vol.append(a.get_volume())
            num = a.numbers
            n_type = len(list(set(num)))
            if n_type > 6:
                n_type = 6
            n_types[n_type] += 1
            for n in num:
                nums[n] +=1

        h = []
        n = []
        for _,v in n_types.items():
            n.append(v)
        for _,v in nums.items():
            h.append(v)

        # Normalize
        n /= np.sum(n)
        h /= np.sum(h)
        ref_n /= np.sum(ref_n)
        ref_h /= np.sum(ref_h)

        # Plot
        with PdfPages(plot_name) as pdf:

            # Plot Element distribution
            plt.bar(list(range(1,110)), np.array(h)/num_atoms, alpha=0.8, label='Generated', color='blueviolet')
            plt.bar(list(range(1,110)), np.array(ref_h)/num_atoms, alpha=0.5, label='Ground Truth', color='darkslategrey')
            plt.title('Fractional element composition')
            plt.xlabel('Atomic Number')
            plt.ylabel('Density')
            plt.legend()
            pdf.savefig()
            plt.close()
          
            # Plot Volume KDE
            w = np.std(vol) * len(vol) ** (-1/5)
            ref_vol = np.array(ref_vol)[:, np.newaxis]
            vol = np.array(vol)[:, np.newaxis]
            x_d = np.linspace(vol.min() - 1, vol.max() + 1, 1000)[:, np.newaxis]
            kde_gt = KernelDensity(kernel='tophat', bandwidth=w).fit(ref_vol)
            density_gt = kde_gt.score_samples(x_d)
            kde_gen = KernelDensity(kernel='tophat', bandwidth=w).fit(vol)
            density_gen = kde_gen.score_samples(x_d)
            plt.plot(x_d, np.exp(density_gen), color='blueviolet', label='Generated')
            plt.plot(x_d, np.exp(density_gt), color='darkslategrey', label='Ground Truth')
            plt.xlabel(r'Volume ($\AA^3$)')
            plt.ylabel('Density')
            plt.title('Volume')
            plt.legend()
            pdf.savefig()
            plt.close()

            # Plot N-ary
            plt.bar(list(range(1,7)), n, alpha=0.8, label='Generated', color='blueviolet')
            plt.bar(list(range(1,7)), ref_n, alpha=0.5, label="Ground Truth", color='darkslategrey')
            plt.title('N-ary')
            plt.xlabel('Unique Elements/Structure')
            plt.ylabel('Density')
            plt.legend()
            pdf.savefig()
            plt.close()

class OMG_CLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, trainer_class=OMG_Trainer)

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        d = LightningCLI.subcommands()
        d["visualize"] = {'model'}
        return d