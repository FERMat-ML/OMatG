import torch
from omg.si.interpolants import PeriodicLinearInterpolant


def main():
    interpolant = PeriodicLinearInterpolant()
    n_atoms = 2  # Changing this to 10 messes most of the values up.
    dims = 3
    batch_pointer = torch.tensor([0, n_atoms, 2 * n_atoms])
    times = torch.zeros((2 * n_atoms, dims))
    times[:n_atoms, :] = 0.3
    times[n_atoms:, :] = 0.7

    x_0 = torch.rand((2 * n_atoms, dims))
    x_1 = torch.rand((2 * n_atoms, dims))
    shift = torch.zeros_like(x_0)
    shift[:n_atoms] = (torch.rand(dims) - 0.5).repeat(n_atoms, 1)
    shift[n_atoms:] = (torch.rand(dims) - 0.5).repeat(n_atoms, 1)

    shifted_x_0 = (x_0 + shift) % 1.0
    shifted_x_1 = (x_1 + shift) % 1.0

    print(interpolant.interpolate_derivative(times, x_0, x_1, batch_pointer))
    print(interpolant.interpolate_derivative(times, shifted_x_0, x_1, batch_pointer))
    print(interpolant.interpolate_derivative(times, x_0, shifted_x_1, batch_pointer))


if __name__ == '__main__':
    main()
