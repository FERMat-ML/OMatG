import matplotlib.pyplot as plt
import numpy as np
import torch
from omg.si.gamma import LatentGammaSqrt
from omg.si.interpolants import LinearInterpolant, PeriodicLinearInterpolant
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant


seed = 3


def main():
    linear_interpolant_without_gamma = SingleStochasticInterpolant(
        interpolant=LinearInterpolant(), gamma=None, epsilon=None, differential_equation_type="ODE")
    linear_interpolant_with_gamma = SingleStochasticInterpolant(
        interpolant=LinearInterpolant(), gamma=LatentGammaSqrt(1.0), epsilon=None, differential_equation_type="ODE")
    periodic_linear_interpolant_without_gamma = SingleStochasticInterpolant(
        interpolant=PeriodicLinearInterpolant(), gamma=None, epsilon=None,
        differential_equation_type="ODE")
    periodic_linear_interpolant_with_gamma = SingleStochasticInterpolant(
        interpolant=PeriodicLinearInterpolant(), gamma=LatentGammaSqrt(1.0), epsilon=None,
        differential_equation_type="ODE")

    x_0 = torch.tensor([[0.1, 0.2]])
    x_1 = torch.tensor([[0.9, 0.8]])
    omega = 2.0 * torch.pi * (x_1 - x_0)
    diff = torch.atan2(torch.sin(omega), torch.cos(omega)) / (2.0 * torch.pi)
    x_1_prime = x_0 + diff  # Image of x_1 closest to x_0.
    batch_pointer = torch.tensor([0, 1])
    times = torch.linspace(0.0, 1.0, 1000)

    x_t_path_linear_without_gamma = []
    x_t_path_linear_with_gamma = []
    x_t_path_periodic_without_gamma = []
    x_t_path_periodic_with_gamma = []
    for t in times:
        full_t = torch.tensor([[t, t]])  # Make time of same shape as x_0 and x_1.

        x_t, _ = linear_interpolant_without_gamma.interpolate(full_t, x_0, x_1_prime, batch_pointer)
        x_t_path_linear_without_gamma.append(x_t[0].numpy())

        torch.manual_seed(seed)  # Enforce always drawing the same z.
        x_t, _ = linear_interpolant_with_gamma.interpolate(full_t, x_0, x_1_prime, batch_pointer)
        x_t_path_linear_with_gamma.append(x_t[0].numpy())

        x_t, _ = periodic_linear_interpolant_without_gamma.interpolate(full_t, x_0, x_1, batch_pointer)
        x_t_path_periodic_without_gamma.append(x_t[0].numpy())

        torch.manual_seed(seed)  # Enforce always drawing the same z.
        x_t, _ = periodic_linear_interpolant_with_gamma.interpolate(full_t, x_0, x_1, batch_pointer)
        x_t_path_periodic_with_gamma.append(x_t[0].numpy())

    x_t_path_linear_without_gamma = np.array(x_t_path_linear_without_gamma)
    x_t_path_linear_with_gamma = np.array(x_t_path_linear_with_gamma)
    x_t_path_periodic_without_gamma = np.array(x_t_path_periodic_without_gamma)
    x_t_path_periodic_with_gamma = np.array(x_t_path_periodic_with_gamma)

    plt.figure()
    plt.plot(x_t_path_periodic_without_gamma[:, 0], x_t_path_periodic_without_gamma[:, 1], color="C1",
             linestyle="none", marker=".", markersize=1)
    plt.plot(x_t_path_periodic_with_gamma[:, 0], x_t_path_periodic_with_gamma[:, 1], color="C1",
             linestyle="none", marker=".")
    plt.plot(x_t_path_linear_without_gamma[:, 0], x_t_path_linear_without_gamma[:, 1], color="C0",
             linestyle="dashed")
    plt.plot(x_t_path_linear_with_gamma[:, 0], x_t_path_linear_with_gamma[:, 1], color="C0")
    plt.axhline(y=0.0, color="black")
    plt.axvline(x=0.0, color="black")
    plt.axhline(y=1.0, color="black")
    plt.axvline(x=1.0, color="black")
    plt.gca().set_aspect("equal")
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
