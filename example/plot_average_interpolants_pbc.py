import matplotlib.pyplot as plt
import numpy as np
import torch
from omg.si.gamma import LatentGammaSqrt
from omg.si.interpolants import PeriodicLinearInterpolant
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant


tries = 10000
time_steps = 1000


def pbc_mean(x):
    # assuming pbcs from 0 to 1
    x_ref = x[0]
    dists = np.abs(x - x_ref) 
    x_prime = np.where(dists >=0.5, x + np.sign(x_ref - 0.5), x)
    return np.average(x_prime) % 1.


def main():
    linear_interpolant_without_gamma = SingleStochasticInterpolant(
        interpolant=PeriodicLinearInterpolant(), gamma=None, epsilon=None, differential_equation_type="ODE")
    linear_interpolant_with_gamma = SingleStochasticInterpolant(
        interpolant=PeriodicLinearInterpolant(), gamma=LatentGammaSqrt(1.0), epsilon=None,
        differential_equation_type="ODE")

    x_0 = torch.tensor([[0.1, 0.2]])
    x_1 = torch.tensor([[0.9, 0.2]])
    diff = torch.abs(x_1 - x_0)
    x_1_prime = torch.where(diff >= 0.5, x_1 + torch.sign(x_0 - 0.5), x_1)
    batch_pointer = torch.tensor([0, 1])
    times = torch.linspace(0.0, 1.0, time_steps)

    plt.figure()

    x_t_path_linear_without_gamma = []
    for t in times:
        full_t = torch.tensor([[t, t]])  # Make time of same shape as x_0 and x_1.
        x_t, _ = linear_interpolant_without_gamma.interpolate(full_t, x_0, x_1_prime, batch_pointer)
        x_t_path_linear_without_gamma.append(x_t[0].numpy())
    x_t_path_linear_without_gamma = np.array(x_t_path_linear_without_gamma)
    plt.scatter(x_t_path_linear_without_gamma[:, 0], x_t_path_linear_without_gamma[:, 1], color="k", marker=".", s=0.5)

    x_t_paths_linear_with_gamma = []
    for i in range(tries):
        seed = torch.seed()  # Always choose a different (fixed) seed.
        if i % 100 == 0:
            print(i)
        x_t_path_linear_with_gamma = []
        for t in times:
            full_t = torch.tensor([[t, t]])
            torch.manual_seed(seed)
            x_t, _ = linear_interpolant_with_gamma.interpolate(full_t, x_0, x_1_prime, batch_pointer)
            x_t_path_linear_with_gamma.append(x_t[0].numpy())
        x_t_path_linear_with_gamma = np.array(x_t_path_linear_with_gamma)
        plt.scatter(x_t_path_linear_with_gamma[:, 0], x_t_path_linear_with_gamma[:, 1], color="C0", alpha=0.05,
                    marker=".", s=0.5)
        x_t_paths_linear_with_gamma.append(x_t_path_linear_with_gamma)

    x_t_paths_linear_with_gamma = np.array(x_t_paths_linear_with_gamma)
    mean_x_t_path_linear_with_gamma = np.array([[pbc_mean(x_t_paths_linear_with_gamma[:, i, 0]), pbc_mean(x_t_paths_linear_with_gamma[:, i, 1])] for i in range(x_t_paths_linear_with_gamma.shape[1])])

    plt.scatter(mean_x_t_path_linear_with_gamma[:, 0], mean_x_t_path_linear_with_gamma[:, 1], color="C1", marker=".")


    plt.axhline(y=0.0, color="black")
    plt.axvline(x=0.0, color="black")
    plt.axhline(y=1.0, color="black")
    plt.axvline(x=1.0, color="black")
    plt.gca().set_aspect("equal")
    #plt.show()
    plt.savefig('plot_average_interpolants_pbc.png')
    #plt.close()


if __name__ == '__main__':
    main()
