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
    com = []
    for i in range(x.shape[-1]):
        dimension_max = 1.0
        theta = (x[:,i] / dimension_max) * 2 * torch.pi
        xi = torch.cos(theta)
        zeta = torch.sin(theta)
        N = x.shape[0]
        xi_bar = (1 / N) * torch.sum(xi, dim=-1)
        zeta_bar = (1 / N) * torch.sum(zeta, dim=-1)
        theta_bar = torch.atan2(-zeta_bar, -xi_bar) + torch.pi
        dimension_com = dimension_max * (theta_bar / (2 * torch.pi))
        com.append(dimension_com)

    # Return
    return torch.tensor(com)

def min_perm_dist(x_0, x_1, distance):
    '''
    Compute the minimum permutational distance between two configurations

    :param x_0:
        Initial configuration.
    :type x_0: torch.tensor
    :param x_1:
        Final configuration.
    :type x_1: torch.tensor
    :return row, col:
        Indices of minimum permutation
    :rtype row, col: (torch.tensor, torch.tensor)
    '''

    # Get data
    ptr = x_0.ptr
    for i in range(len(ptr)-1):

        # Shuffle
        assert x_0.pos.shape == x_1.pos.shape
        assert x_0.species.shape == x_1.species.shape

        # Minimum Perm
        row, col = min_perm_dist(x_0.pos[ptr[i]:ptr[i+1]], x_1.pos[ptr[i]:ptr[i+1]], periodic_distance)

        # Reassign
        x_0.pos[ptr[i]:ptr[i+1]] = x_0.pos[ptr[i]:ptr[i+1]][row]
        x_0.species[ptr[i]:ptr[i+1]] = x_0.species[ptr[i]:ptr[i+1]][row]
        x_1.pos[ptr[i]:ptr[i+1]] = x_1.pos[ptr[i]:ptr[i+1]][col]
        x_1.species[ptr[i]:ptr[i+1]] = x_1.species[ptr[i]:ptr[i+1]][col]

    # Compute distance matrix
    distance_matrix = distance(x_0[:, None, :], x_1[None, :, :])
    row, col = linear_sum_assignment(distance_matrix)
    return x_0, x_1

def periodic_distance(x_0, x_1):
    '''
    Compute Periodic distance between points
    '''
    dist = x_0 - x_1 - torch.floor(x_0 - x_1)
    dist = torch.min(dist, 1.0-dist)
    return torch.norm(dist, dim=-1)

def euclidian_distance(x_0, x_1):
    '''
    Compute Euclidian distance between points
    '''
    dist = x_1 - x_0
    return torch.norm(dist, dim=-1)