import torch
from scipy.optimize import linear_sum_assignment

def compute_com(x):
    '''
    Function to compute center of mass
    :param x:
        Configuration
    :type x: torch.tensor
    :return com:
        Center of mass
    :rtype com: torch.tensor
    '''
    
    # Compute COM
    N = x.shape[0]
    return torch.sum(x, dim=0) / N

def min_perm_dist(x_0, x_1):
    '''
    Compute the minimum permutational distance between two configurations

    :param x_0:
        Initial configuration.
    :type x_0: torch.tensor
    :param x_1:
        Final configuration.
    :type x_1: torch.tensor
    '''

    # Align center of mass
    x_0 = (x_0 - compute_com(x_0)) % 1.
    x_1 = (x_1 - compute_com(x_1)) % 1.

    # Compute distance matrix
    distance_matrix = torch.norm(x_0[:, None, :] - x_1[None, :, :], dim=-1)
    row, col = linear_sum_assignment(distance_matrix)
    return x_0[row], x_1