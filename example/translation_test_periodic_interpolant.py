import torch
from torch_scatter import scatter_mean
from omg.si.corrector import PeriodicBoundaryConditionsCorrector
from omg.si.interpolants import PeriodicLinearInterpolant
torch.manual_seed(1)


def main():
    interpolant = PeriodicLinearInterpolant()
    n_atoms = 2  # Changing this to 10 messes most of the values up.
    dims = 3
    shift_range = 0.01  # Shift goes from [-shift_range, shift_range].
    batch_indices = torch.repeat_interleave(torch.arange(2), n_atoms)
    times = torch.zeros((2 * n_atoms, dims))
    times[:n_atoms, :] = 0.3
    times[n_atoms:, :] = 0.7

    x_0 = torch.rand((2 * n_atoms, dims))
    x_1 = torch.rand((2 * n_atoms, dims))
    shift = torch.zeros_like(x_0)
    shift[:n_atoms] = (torch.rand(dims) * 2.0 * shift_range - shift_range).repeat(n_atoms, 1)
    shift[n_atoms:] = (torch.rand(dims) * 2.0 * shift_range - shift_range).repeat(n_atoms, 1)

    shifted_x_0 = (x_0 + shift) % 1.0
    shifted_x_1 = (x_1 + shift) % 1.0

    reference = interpolant.interpolate_derivative(times, x_0, x_1)
    # This is the correction from FlowMM.
    reference = reference - torch.index_select(scatter_mean(reference, batch_indices, dim=0), 0, batch_indices)
    shifted_zero = interpolant.interpolate_derivative(times, shifted_x_0, x_1)
    shifted_zero = shifted_zero - torch.index_select(scatter_mean(shifted_zero, batch_indices, dim=0), 0, batch_indices)
    shifted_one = interpolant.interpolate_derivative(times, x_0, shifted_x_1)
    shifted_one = shifted_one - torch.index_select(scatter_mean(shifted_one, batch_indices, dim=0), 0, batch_indices)
    shifted_two = interpolant.interpolate_derivative(times, shifted_x_0, shifted_x_1)
    shifted_two = shifted_two - torch.index_select(scatter_mean(shifted_two, batch_indices, dim=0), 0, batch_indices)

    print(torch.count_nonzero(torch.logical_not(torch.isclose(reference, shifted_zero, atol=1e-7))))
    print(torch.count_nonzero(torch.logical_not(torch.isclose(reference, shifted_one, atol=1e-7))))
    print(torch.count_nonzero(torch.logical_not(torch.isclose(reference, shifted_two, atol=1e-7))))


if __name__ == '__main__':
    main()
