import torch
from omg.si.interpolants import PeriodicLinearInterpolant


def main():
    interpolant = PeriodicLinearInterpolant()
    n_atoms = 4  # Changing this to 10 messes most of the values up.
    dims = 3
    batch_pointer = torch.tensor([0, n_atoms])
    times = torch.full((n_atoms, dims), 0.3)

    x_0 = torch.rand((n_atoms, dims))
    x_1 = torch.rand((n_atoms, dims))
    shift = (torch.rand(dims) - 0.5).repeat(n_atoms, 1)

    shifted_x_0 = (x_0 + shift) % 1.0
    shifted_x_1 = (x_1 + shift) % 1.0

    print(interpolant.interpolate_derivative(times, x_0, x_1, batch_pointer))
    print(interpolant.interpolate_derivative(times, shifted_x_0, x_1, batch_pointer))
    print(interpolant.interpolate_derivative(times, x_0, shifted_x_1, batch_pointer))


if __name__ == '__main__':
    main()
