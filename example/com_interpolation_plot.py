import matplotlib.pyplot as plt
import numpy as np
import torch
from omg.si.corrector import PeriodicBoundaryConditionsCorrector
from omg.si.interpolants import PeriodicLinearInterpolant, PeriodicScoreBasedDiffusionModelInterpolant


def main():
    n_atoms = 10  # Changing this to 10 messes most of the values up.
    dims = 2
    time_steps = 1000
    batch_indices = torch.repeat_interleave(torch.arange(1), n_atoms)
    corrector = PeriodicBoundaryConditionsCorrector(0.0, 1.0)

    interpolant = PeriodicLinearInterpolant()
    # interpolant = PeriodicScoreBasedDiffusionModelInterpolant()

    x_0 = torch.rand((n_atoms, dims))
    x_1 = torch.rand((n_atoms, dims))

    x_0_com = corrector.compute_center_of_mass(x_0, batch_indices)
    x_1_com = corrector.compute_center_of_mass(x_1, batch_indices)

    shifted_x_0 = corrector.correct(x_0 - x_0_com)
    shifted_x_1 = corrector.correct(x_1 - x_1_com)

    print(x_0_com[0])
    print(corrector.compute_center_of_mass(shifted_x_0, batch_indices)[0])
    print(x_1_com[0])
    print(corrector.compute_center_of_mass(shifted_x_1, batch_indices)[0])

    times = torch.linspace(0, 1, time_steps)

    x_t_path = [x_0.numpy()]
    x_t_com_path = [x_0_com[0].numpy()]  # COM same for every atom.
    shifted_x_t_path = [shifted_x_0.numpy()]
    shifted_x_t_com_path = [corrector.compute_center_of_mass(shifted_x_0, batch_indices)[0].numpy()]

    for i, t in enumerate(times):
        full_t = torch.full_like(x_0, t)
        x_t = interpolant.interpolate(full_t, x_0, x_1)
        x_t_path.append(x_t.numpy())
        x_t_com_path.append(corrector.compute_center_of_mass(x_t, batch_indices)[0].numpy())
        shifted_x_t = interpolant.interpolate(full_t, shifted_x_0, shifted_x_1)
        shifted_x_t_path.append(shifted_x_t.numpy())
        shifted_x_t_com_path.append(corrector.compute_center_of_mass(shifted_x_t, batch_indices)[0].numpy())

    x_t_path = np.array(x_t_path)
    x_t_com_path = np.array(x_t_com_path)
    shifted_x_t_path = np.array(shifted_x_t_path)
    shifted_x_t_com_path = np.array(shifted_x_t_com_path)

    print(shifted_x_t_com_path)
    print(np.max(shifted_x_t_com_path))

    plt.figure()
    plt.scatter(x_t_com_path[:, 0], x_t_com_path[:, 1], color="C0", s=1.0, label="COM path unshifted")
    plt.scatter(shifted_x_t_com_path[:, 0], shifted_x_t_com_path[:, 1], color="C1", s=1.0, label="COM path shifted")
    plt.axhline(0.0, color="k")
    plt.axhline(1.0, color="k")
    plt.axvline(0.0, color="k")
    plt.axvline(1.0, color="k")
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
