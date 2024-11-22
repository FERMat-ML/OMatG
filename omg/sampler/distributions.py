from ase.geometry.cell import cellpar_to_cell
import numpy as np
import scipy
import torch
from torch.distributions import LogNormal

# TODO: !!!Highly WIP!!! from MatterGen for lattice sampling. Not sure it actually works like they say it does!
'''
class NDependentScaledNormal:
    """
    Samples a scaled normal distribution which is scaled according to n, with c and v fixed by the given dataset
    """
    def __init__(self, c):
        self.c = c
        self.rng = np.random.default_rng()

    def __call__(self, n, size):
        mean = (n * self.c) ** (1/3)
        stdev = mean / 8
        val = self.rng.normal(mean, stdev, size)
        l = [val, val, val] * np.identity(3)
        print (l)
        return l
'''


class NDependentGamma:
    def __init__(self, a, loc, scale):
        self.a = a
        self.loc = loc
        self.scale = scale

    def __call__(self, n):
        v = n / scipy.stats.gamma.rvs(self.a, self.scale, self.scale)
        a = v ** (1 / 3)
        cell = [a, a, a] * np.identity(3)
        return cell


class InformedLatticeDistribution(object):
    """
    Informed lattice distribution for different datasets as used by FlowMM (see https://arxiv.org/abs/2406.04713).
    """

    def __init__(self, dataset_name: str) -> None:
        # Numbers taken from
        # https://github.com/facebookresearch/flowmm/blob/main/src/flowmm/rfm/manifolds/lattice_params_stats.yaml
        if dataset_name == "carbon_24":
            self._length_log_means = [0.9852757453918457, 1.3865314722061157, 1.7068126201629639]
            self._length_log_stds = [0.14957907795906067, 0.20431114733219147, 0.2403733879327774]
        elif dataset_name == "mp_20":
            self._length_log_means = [1.575442910194397, 1.7017393112182617, 1.9781638383865356]
            self._length_log_stds = [0.24437622725963593, 0.26526379585266113, 0.3535512685775757]
        elif dataset_name == "mpts_52":
            self._length_log_means = [1.6565313339233398, 1.8407557010650635, 2.1225264072418213]
            self._length_log_stds = [0.2952289581298828, 0.3340013027191162, 0.41885802149772644]
        elif dataset_name == "perov":
            self._length_log_means = [1.419227957725525, 1.419227957725525, 1.419227957725525]
            self._length_log_stds = [0.07268335670232773, 0.07268335670232773, 0.07268335670232773]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        self._length_distribution = LogNormal(torch.tensor(self._length_log_means),
                                              torch.tensor(self._length_log_stds))

    def __call__(self, number_of_atoms: int):
        """
        Sample three three-dimensional lattice vectors in a 3x3 matrix for the given number of atoms.

        This method ignores the given number of atoms.

        :param number_of_atoms:
            Number of atoms.
        :type number_of_atoms: int

        :return:
            A numpy array of shape (size, 3, 3).
        :rtype: numpy.ndarray
        """
        lengths = self._length_distribution.sample().numpy()
        # Generate uniform angles between 60 and 120 degrees.
        # Ase wants angles in degrees.
        angles = ((torch.rand(3) * 60.0) + 60.0).numpy()
        assert lengths.shape == (3,)
        assert angles.shape == (3,)
        return cellpar_to_cell(np.concatenate((lengths, angles)))


class MaskDistribution(object):
    # TODO: The masking token should be stored globally.
    def __init__(self, token=0):
        self.token = token

    def __call__(self, size):
        return np.ones(size) * self.token


class MirrorData(object):
    """
    Distribution that mirrors the given data.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, data: torch.Tensor) -> np.ndarray:
        # TODO: Introduce an abstract base class for all of these distributions.
        # I think all classes should just get the entire pos, species, cell data.
        return data.clone().numpy(force=True)
