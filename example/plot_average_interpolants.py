import matplotlib.pyplot as plt
import numpy as np
import torch
from omg.si.gamma import LatentGammaSqrt
from omg.si.interpolants import LinearInterpolant
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant


tries = 1000


def main():
    linear_interpolant_without_gamma = SingleStochasticInterpolant(
        interpolant=LinearInterpolant(), gamma=None, epsilon=None, differential_equation_type="ODE")
    linear_interpolant_with_gamma = SingleStochasticInterpolant(
        interpolant=LinearInterpolant(), gamma=LatentGammaSqrt(1.0), epsilon=None, differential_equation_type="ODE")

    x_0 = torch.tensor([[0.1, 0.2]])
    x_1 = torch.tensor([[0.9, 0.8]])
    omega = 2.0 * torch.pi * (x_1 - x_0)
    diff = torch.atan2(torch.sin(omega), torch.cos(omega)) / (2.0 * torch.pi)
    x_1_prime = x_0 + diff  # Image of x_1 closest to x_0.
    batch_pointer = torch.tensor([0, 1])
    times = torch.linspace(0.0, 1.0, 1000)

    plt.figure()

    x_t_path_linear_without_gamma = []
    for t in times:
        full_t = torch.tensor([[t, t]])  # Make time of same shape as x_0 and x_1.
        x_t, _ = linear_interpolant_without_gamma.interpolate(full_t, x_0, x_1_prime, batch_pointer)
        x_t_path_linear_without_gamma.append(x_t[0].numpy())
    x_t_path_linear_without_gamma = np.array(x_t_path_linear_without_gamma)
    plt.plot(x_t_path_linear_without_gamma[:, 0], x_t_path_linear_without_gamma[:, 1], color="k",
             linestyle="dashed")

    x_t_paths_linear_with_gamma = []
    for i in range(tries):
        if i % 100 == 0:
            print(i)
        x_t_path_linear_with_gamma = []
        for t in times:
            full_t = torch.tensor([[t, t]])
            torch.manual_seed(i)  # Always choose a different (fixed) seed.
            x_t, _ = linear_interpolant_with_gamma.interpolate(full_t, x_0, x_1_prime, batch_pointer)
            x_t_path_linear_with_gamma.append(x_t[0].numpy())
        x_t_path_linear_with_gamma = np.array(x_t_path_linear_with_gamma)
        plt.plot(x_t_path_linear_with_gamma[:, 0], x_t_path_linear_with_gamma[:, 1], color="C0", alpha=0.05)
        x_t_paths_linear_with_gamma.append(x_t_path_linear_with_gamma)

    x_t_paths_linear_with_gamma = np.array(x_t_paths_linear_with_gamma)
    mean_x_t_path_linear_with_gamma = np.mean(x_t_paths_linear_with_gamma, axis=0)

    print(np.abs(mean_x_t_path_linear_with_gamma - x_t_path_linear_without_gamma).max())
    plt.plot(mean_x_t_path_linear_with_gamma[:, 0], mean_x_t_path_linear_with_gamma[:, 1], color="C1")

    plt.axhline(y=0.0, color="black")
    plt.axvline(x=0.0, color="black")
    plt.axhline(y=1.0, color="black")
    plt.axvline(x=1.0, color="black")
    plt.gca().set_aspect("equal")
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
