from ase.geometry.cell import cellpar_to_cell
import numpy as np
import torch
from torch.distributions import LogNormal
from .abstracts import CellDistribution


class MirrorCell(CellDistribution):
    """
    Base distribution that just mirrors the given cell of a single structure.
    """

    def __init__(self) -> None:
        """Constructor of the MirrorCell class."""
        super().__init__()

    def __call__(self, cell: torch.Tensor) -> np.ndarray:
        """
        Sample a cell from the base distribution given the cell of a single structure.

        This function just returns a clone of the input cell.

        :param cell:
            The cell of a single structure in a tensor of shape (3, 3).
        :type cell: torch.Tensor

        :return:
            A sampled cell from the base distribution in a tensor of shape (3, 3).
        :rtype: np.ndarray
        """
        return cell.detach().clone().cpu().numpy()


class NormalCellDistribution(CellDistribution):
    """
    Base distribution that samples entries of cell vectors from a normal distribution with given mean and standard
    deviation.

    :param loc:
        Mean of the normal distribution.
        Defaults to 0.0.
    :type loc: float
    :param scale:
        Standard deviation of the normal distribution.
        Defaults to 1.0.
    :type scale: float

    :raises ValueError:
        If the scale is less than or equal to 0.
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        """Constructor of the NormalCellDistribution class."""
        super().__init__()
        if scale <= 0.0:
            raise ValueError("Scale must be greater than 0.")
        self._loc = loc
        self._scale = scale

    def __call__(self, cell: torch.Tensor) -> np.ndarray:
        """
        Sample a cell from the base distribution given the cell of a single structure.

        param cell:
            The cell of a single structure in a tensor of shape (3, 3).
        :type cell: torch.Tensor

        :return:
            A sampled cell from the base distribution in a tensor of shape (3, 3).
        :rtype: np.ndarray
        """
        return np.random.normal(loc=self._loc, scale=self._scale, size=cell.shape)


class InformedLatticeDistribution(CellDistribution):
    """
    Informed base lattice distribution for different datasets as used by FlowMM (see https://arxiv.org/abs/2406.04713).

    This distribution samples lattice vectors based on log-normal distributions for the lengths of the lattice vectors
    and uniform distributions for the angles between the lattice vectors. The parameters of the log-normal distributions
    are taken from FlowMM's fits to the training datasets (see also fit_lattice in OMGTrainer).

    :param dataset_name:
        Name of the dataset.
        Can be one of "carbon_24", "mp_20", "mpts_52", "perov_5", and "alex_mp_20".
    :type dataset_name: str

    :raises ValueError:
        If the dataset name is unknown.
    """

    def __init__(self, dataset_name: str) -> None:
        """Constructor of the InformedLatticeDistribution class."""
        super().__init__()
        # Most numbers taken from
        # https://github.com/facebookresearch/flowmm/blob/main/src/flowmm/rfm/manifolds/lattice_params_stats.yaml
        # fit_lattice function in OMGTrainer also yields these values.
        if dataset_name == "carbon_24":
            self._length_log_means = [0.9852757453918457, 1.3865314722061157, 1.7068126201629639]
            self._length_log_stds = [0.14957907795906067, 0.20431114733219147, 0.2403733879327774]
        elif dataset_name == "mp_20":
            self._length_log_means = [1.575442910194397, 1.7017393112182617, 1.9781638383865356]
            self._length_log_stds = [0.24437622725963593, 0.26526379585266113, 0.3535512685775757]
        elif dataset_name == "mpts_52":
            self._length_log_means = [1.6565313339233398, 1.8407557010650635, 2.1225264072418213]
            self._length_log_stds = [0.2952289581298828, 0.3340013027191162, 0.41885802149772644]
        elif dataset_name == "perov_5":
            self._length_log_means = [1.419227957725525, 1.419227957725525, 1.419227957725525]
            self._length_log_stds = [0.07268335670232773, 0.07268335670232773, 0.07268335670232773]
        elif dataset_name == "alex_mp_20":
            self._length_log_means = [1.5808929163076058, 1.74672046352959, 2.065243388307474]
            self._length_log_stds = [0.27284015410437057, 0.2944785731740152, 0.30899526911753017]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        # Use CPU device to align with other distributions.
        self._length_distribution = LogNormal(torch.tensor(self._length_log_means, device="cpu"),
                                              torch.tensor(self._length_log_stds, device="cpu"))

    def __call__(self, cell: torch.Tensor) -> np.ndarray:
        """
        Sample a cell from the informed base distribution given the cell of a single structure.

        :param cell:
            The cell of a single structure in a tensor of shape (3, 3).
        :type cell: torch.Tensor

        :return:
            A sampled cell from the base distribution in a tensor of shape (3, 3).
        :rtype: np.ndarray
        """
        lengths = self._length_distribution.sample().numpy()
        # Generate uniform angles between 60 and 120 degrees.
        # Ase wants angles in degrees.
        angles = ((torch.rand(3, device="cpu") * 60.0) + 60.0).numpy()
        assert lengths.shape == (3,)
        assert angles.shape == (3,)
        return cellpar_to_cell(np.concatenate((lengths, angles)))
