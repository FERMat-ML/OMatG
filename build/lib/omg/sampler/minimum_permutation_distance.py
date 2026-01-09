from scipy.optimize import linear_sum_assignment
import torch
from omg.datamodule import OMGData
from omg.si.corrector import Corrector


def correct_for_minimum_permutation_distance(x_0: OMGData, x_1: OMGData, corrector: Corrector,
                                             switch_species: bool = False) -> None:
    """
    For every configuration in the batch, permute the fractional coordinates (and species) in x_0 so that it minimizes
    the distance with respect to the corresponding configuration in x_1.

    This function modifies x_0 in place.

    Switching both the species and the fractional coordinations will keep the original configuration intact and just
    change the order of atoms. This order is relevant for the pairings in the stochastic interpolants during training.

    However, in crystal-structure prediction, the species should not be permuted so that stochastic interpolants are
    constructed between the same species.

    In de-novo generation, we either use fully masked species or uniformly distributed species, so that keeping the
    species or not does not yield any difference.

    :param x_0:
        Batch of initial configurations stored in a OMGData object.
    :type x_0: OMGData
    :param x_1:
        Batch of final configurations stored in a OMGData object.
    :type x_1: OMGData
    :param corrector:
        Corrector that corrects the distances (for instance, to consider periodic boundary conditions).
    :type corrector: Corrector
    :param switch_species:
        Whether to switch species as well to keep the original configuration intact.
    :type switch_species: bool
    """
    assert torch.all(x_0.ptr == x_1.ptr)
    assert x_0.pos.shape == x_1.pos.shape
    assert x_0.species.shape == x_1.species.shape
    assert x_0.cell.shape == x_1.cell.shape

    # Pointers to start and end of each configuration.
    ptr = x_0.ptr
    # TODO: This could be trivially parallelized.
    for i in range(len(ptr) - 1):
        # Find minimum permutation for configuration.
        p1 = x_0.pos[ptr[i]:ptr[i + 1]]  # Shape (n, 3).
        p2 = x_1.pos[ptr[i]:ptr[i + 1]]  # Shape (n, 3).
        distance_matrix = _distance_matrix(p1, p2, corrector)  # Shape (n, n)
        row, col = linear_sum_assignment(distance_matrix.cpu())
        # For square cost matrices, the row indices are sorted.
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html.
        assert torch.all(row == torch.arange(len(row)))
        # Reassign
        x_0.pos[ptr[i]:ptr[i + 1]] = x_0.pos[ptr[i]:ptr[i + 1]][col]
        if switch_species:
            x_0.species[ptr[i]:ptr[i + 1]] = x_0.species[ptr[i]:ptr[i + 1]][col]


def _distance_matrix(x_0: torch.Tensor, x_1: torch.Tensor, corrector: Corrector) -> torch.Tensor:
    """
    Compute the distance matrix between two sets of n positions.

    The distance matrix is a matrix of shape (n, n) where the element (i, j) is the distance between the i-th position
    in x_0 and the j-th position in x_1.

    :param x_0:
        First set of positions of shape (n, 3).
    :type x_0: torch.Tensor
    :param x_1:
        Second set of positions of shape (n, 3).
    :type x_1: torch.Tensor
    :param corrector:
        Corrector that corrects the distances (for instance, to consider periodic boundary conditions).
    :type corrector: Corrector

    :return:
        Distance matrix of shape (n, n).
    :rtype distance: torch.Tensor
    """
    assert x_0.shape == x_1.shape
    assert len(x_0.shape) == 2
    assert x_0.shape[1] == 3
    # Unwrap all x_1 configurations with respect to all x_0 configurations.
    # Use broadcasting: Shape (1, n, 3) - (n, 1, 3) = (n, n, 3).
    x_1_prime = corrector.unwrap(x_0[None, :, :], x_1[:, None, :])  # Shape (n, n, 3).
    # Use norm along last dimension.
    return torch.norm(x_1_prime - x_0[None, :, :], dim=-1)
