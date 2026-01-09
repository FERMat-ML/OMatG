import torch
from torch_geometric.data import Batch
from omg.datamodule import OMGData, Structure
from omg.sampler import Sampler
from .abstracts import SpeciesDistribution, CellDistribution, PositionDistribution


class IndependentSampler(Sampler):
    """
    Samplers that independently sample species, positions and cells from given base distributions.

    :param species_distribution:
        The species base distribution to sample species from.
    :type species_distribution: SpeciesDistribution
    :param pos_distribution:
        The position base distribution to sample positions from.
    :type pos_distribution: PositionDistribution
    :param cell_distribution:
        The cell base distribution to sample cells from.
    :type cell_distribution: CellDistribution
    """

    def __init__(self, species_distribution: SpeciesDistribution, pos_distribution: PositionDistribution,
                 cell_distribution: CellDistribution) -> None:
        """Constructor for the IndependentSampler class."""
        super().__init__()
        self._species_distribution = species_distribution
        self._pos_distribution = pos_distribution
        self._cell_distribution = cell_distribution

    def sample_p_0(self, x_1: OMGData) -> OMGData:
        """
        Sample the base distribution given a batch of structures from the training data.

        The returned OMGData represents a batch of structures sampled from the base distribution.

        The number of atoms in each sampled structure is the same as in the corresponding structure in the input OMGData
        x_1. For the positions, they choose the same representation (cartesian or fractional) as in the input OMGData
        x_1.

        :param x_1:
            A batch of structures from the training distribution.
        :type x_1: OMGData

        :return:
            A batch of structures sampled from the base distribution.
        :rtype: OMGData
        """
        structures = []
        for i in range(len(x_1.n_atoms)):
            sl = slice(x_1.ptr[i], x_1.ptr[i + 1])
            species = self._species_distribution(x_1.species[sl])
            # Don't wrap back positions explicitly, this should be done by the interpolants.
            pos, pos_is_fractional = self._pos_distribution(x_1.pos[sl], x_1.pos_is_fractional[i])
            cell = self._cell_distribution(x_1.cell[i])
            sampled_structure = Structure(cell=torch.from_numpy(cell).to(x_1.cell.dtype),
                                          atomic_numbers=torch.tensor(species, dtype=torch.int64),
                                          pos=torch.from_numpy(pos).to(x_1.pos.dtype),
                                          pos_is_fractional=pos_is_fractional)
            # Make sure to match the coordinate system of x1.
            if x_1.pos_is_fractional[i]:
                sampled_structure.convert_to_fractional()
            else:
                sampled_structure.convert_to_cartesian()
            structures.append(OMGData(sampled_structure))

        # Assemble batch of OMGData objects.
        # noinspection PyTypeChecker
        return Batch.from_data_list(structures)
