from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from omg.si.corrector import IdentityCorrector, PeriodicBoundaryConditionsCorrector
from omg.si.gamma import LatentGammaSqrt
from omg.si.interpolants import (LinearInterpolant, PeriodicLinearInterpolant,
                                 ScoreBasedDiffusionModelInterpolant,
                                 PeriodicScoreBasedDiffusionModelInterpolant)
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant, torch


time_steps = 1000
Interpolant = ScoreBasedDiffusionModelInterpolant
Interpolant = LinearInterpolant
PeriodicInterpolant = PeriodicScoreBasedDiffusionModelInterpolant
PeriodicInterpolant = PeriodicLinearInterpolant


def mean(x_t_paths, reference=None, distance_corrector=IdentityCorrector(), position_corrector=IdentityCorrector()):
    if reference is None:
        return np.mean(x_t_paths, axis=0)
    if isinstance(reference, int):
        reference = x_t_paths[reference].copy()

    correct_distance = lambda x: distance_corrector.correct(torch.tensor(x)).numpy()
    correct_position = lambda x: position_corrector.correct(torch.tensor(x)).numpy()

    distances = np.array([correct_distance(x - reference) for x in x_t_paths])
    corrected_x = np.array([reference + distance for distance in distances])
    return correct_position(np.mean(corrected_x, axis=0))


mean_method = partial(mean, reference=None, distance_corrector=IdentityCorrector(),
                      position_corrector=IdentityCorrector())
mean_method = partial(mean, reference=np.zeros((time_steps, 2)), distance_corrector=IdentityCorrector(),
                      position_corrector=IdentityCorrector())
mean_method = partial(mean, reference=np.random.uniform((time_steps, 2)), distance_corrector=IdentityCorrector(),
                      position_corrector=IdentityCorrector())
mean_method = partial(mean, reference=1, distance_corrector=IdentityCorrector(), position_corrector=IdentityCorrector())

mean_method_pbc = partial(mean, reference=None, distance_corrector=None, position_corrector=None)
mean_method_pbc = partial(mean, reference=np.zeros((time_steps, 2)),
                          distance_corrector=PeriodicBoundaryConditionsCorrector(min_value=-0.5, max_value=0.5),
                          position_corrector=PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0))
mean_method_pbc = partial(mean, reference=np.full((time_steps, 2), 0.5),
                          distance_corrector=PeriodicBoundaryConditionsCorrector(min_value=-0.5, max_value=0.5),
                          position_corrector=PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0))
mean_method_pbc = partial(mean, reference=np.random.uniform((time_steps, 2)),
                          distance_corrector=PeriodicBoundaryConditionsCorrector(min_value=-0.5, max_value=0.5),
                          position_corrector=PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0))
mean_method_pbc = partial(mean, reference=0,
                          distance_corrector=PeriodicBoundaryConditionsCorrector(min_value=-0.5, max_value=0.5),
                          position_corrector=PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0))


def main():
    linear_interpolant_without_gamma = SingleStochasticInterpolant(
        interpolant=Interpolant(), gamma=None, epsilon=None, differential_equation_type="ODE")
    linear_interpolant_without_gamma_periodic = SingleStochasticInterpolant(
        interpolant=PeriodicInterpolant(), gamma=None, epsilon=None, differential_equation_type="ODE")
    linear_interpolant_with_gamma = SingleStochasticInterpolant(
        interpolant=Interpolant(), gamma=LatentGammaSqrt(1.0), epsilon=None, differential_equation_type="ODE")
    linear_interpolant_with_gamma_periodic = SingleStochasticInterpolant(
        interpolant=PeriodicInterpolant(), gamma=LatentGammaSqrt(1.0), epsilon=None,
        differential_equation_type="ODE")

    x_0 = torch.tensor([[0.1, 0.2]])
    x_1 = torch.tensor([[0.7, 0.9]])
    torch.manual_seed(9468949097199408570)
    #print(torch.seed())
    random_z = torch.randn_like(x_0)
    omega = 2.0 * torch.pi * (x_1 - x_0)
    diff = torch.atan2(torch.sin(omega), torch.cos(omega)) / (2.0 * torch.pi)
    x_1_prime = x_0 + diff  # Image of x_1 closest to x_0.
    batch_pointer = torch.tensor([0, 1])
    times = torch.linspace(0.0, 1.0, time_steps)

    plt.figure()

    x_t_path_linear_without_gamma_periodic = []
    for t in times:
        full_t = torch.tensor([[t, t]])  # Make time of same shape as x_0 and x_1.
        x_t, _ = linear_interpolant_without_gamma_periodic.interpolate(full_t, x_0, x_1_prime, batch_pointer)
        x_t_path_linear_without_gamma_periodic.append(x_t[0].numpy())
    x_t_path_linear_without_gamma_periodic = np.array(x_t_path_linear_without_gamma_periodic)
    plt.scatter(x_t_path_linear_without_gamma_periodic[:, 0], x_t_path_linear_without_gamma_periodic[:, 1], color="k",
                marker=".", s=0.5)

    x_t_path_linear_without_gamma = []
    for t in times:
        full_t = torch.tensor([[t, t]])  # Make time of same shape as x_0 and x_1.
        x_t, _ = linear_interpolant_without_gamma.interpolate(full_t, x_0, x_1_prime, batch_pointer)
        x_t_path_linear_without_gamma.append(x_t[0].numpy())
    x_t_path_linear_without_gamma = np.array(x_t_path_linear_without_gamma)
    plt.scatter(x_t_path_linear_without_gamma[:, 0], x_t_path_linear_without_gamma[:, 1], color="k",
                marker=".", s=0.5)

    x_t_paths_linear_with_gamma_periodic = []
    x_t_paths_linear_with_gamma = []

    x_t_p_path_linear_with_gamma_periodic = []
    for t in times:
        full_t = torch.tensor([[t, t]])
        torch.randn_like = lambda _: random_z
        x_t, _ = linear_interpolant_with_gamma_periodic.interpolate(full_t, x_0, x_1_prime, batch_pointer)
        x_t_p_path_linear_with_gamma_periodic.append(x_t[0].numpy())
    x_t_p_path_linear_with_gamma_periodic = np.array(x_t_p_path_linear_with_gamma_periodic)
    plt.scatter(x_t_p_path_linear_with_gamma_periodic[:, 0], x_t_p_path_linear_with_gamma_periodic[:, 1], color="C0",
                alpha=0.5, marker=".", s=0.5)
    x_t_paths_linear_with_gamma_periodic.append(x_t_p_path_linear_with_gamma_periodic)

    x_t_p_path_linear_with_gamma = []
    for t in times:
        full_t = torch.tensor([[t, t]])
        torch.randn_like = lambda _: random_z
        x_t, _ = linear_interpolant_with_gamma.interpolate(full_t, x_0, x_1_prime, batch_pointer)
        x_t_p_path_linear_with_gamma.append(x_t[0].numpy())
    x_t_p_path_linear_with_gamma = np.array(x_t_p_path_linear_with_gamma)
    plt.scatter(x_t_p_path_linear_with_gamma[:, 0], x_t_p_path_linear_with_gamma[:, 1], color="C0",
                alpha=0.5, marker=".", s=0.5)
    x_t_paths_linear_with_gamma.append(x_t_p_path_linear_with_gamma)

    x_t_m_path_linear_with_gamma_periodic = []
    for t in times:
        full_t = torch.tensor([[t, t]])
        torch.randn_like = lambda _: -random_z
        x_t, _ = linear_interpolant_with_gamma_periodic.interpolate(full_t, x_0, x_1_prime, batch_pointer)
        x_t_m_path_linear_with_gamma_periodic.append(x_t[0].numpy())
    x_t_m_path_linear_with_gamma_periodic = np.array(x_t_m_path_linear_with_gamma_periodic)
    plt.scatter(x_t_m_path_linear_with_gamma_periodic[:, 0], x_t_m_path_linear_with_gamma_periodic[:, 1], color="C0",
                alpha=0.5, marker=".", s=0.5)
    x_t_paths_linear_with_gamma_periodic.append(x_t_m_path_linear_with_gamma_periodic)

    x_t_m_path_linear_with_gamma = []
    for t in times:
        full_t = torch.tensor([[t, t]])
        torch.randn_like = lambda _: -random_z
        x_t, _ = linear_interpolant_with_gamma.interpolate(full_t, x_0, x_1_prime, batch_pointer)
        x_t_m_path_linear_with_gamma.append(x_t[0].numpy())
    x_t_m_path_linear_with_gamma = np.array(x_t_m_path_linear_with_gamma)
    plt.scatter(x_t_m_path_linear_with_gamma[:, 0], x_t_m_path_linear_with_gamma[:, 1], color="C0",
                alpha=0.5, marker=".", s=0.5)
    x_t_paths_linear_with_gamma.append(x_t_m_path_linear_with_gamma)

    x_t_paths_linear_with_gamma_periodic = np.array(x_t_paths_linear_with_gamma_periodic)
    x_t_paths_linear_with_gamma = np.array(x_t_paths_linear_with_gamma)

    mean_x_t_path_linear_with_gamma_periodic = mean_method_pbc(x_t_paths_linear_with_gamma_periodic)
    correct_mean_x_t_path_linear_with_gamma_periodic = mean(
        x_t_paths_linear_with_gamma_periodic, reference=x_t_path_linear_without_gamma_periodic,
        distance_corrector=PeriodicBoundaryConditionsCorrector(min_value=-0.5, max_value=0.5),
        position_corrector=PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0))
    #random_index = np.random.randint(0, time_steps)
    #reference_point = x_t_path_linear_without_gamma_periodic[random_index]
    #reference_point = np.tile(reference_point, (time_steps, 1))
    #correct_mean_x_t_path_linear_with_gamma_periodic = mean(
    #    x_t_paths_linear_with_gamma_periodic, reference=reference_point,
    #    distance_corrector=PeriodicBoundaryConditionsCorrector(min_value=-0.5, max_value=0.5),
    #    position_corrector=PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0))
    mean_x_t_path_linear_with_gamma = mean_method(x_t_paths_linear_with_gamma)

    print(np.abs(correct_mean_x_t_path_linear_with_gamma_periodic - x_t_path_linear_without_gamma_periodic).max())
    print(np.abs(mean_x_t_path_linear_with_gamma_periodic - x_t_path_linear_without_gamma_periodic).max())
    print(np.abs(mean_x_t_path_linear_with_gamma - x_t_path_linear_without_gamma).max())

    plt.scatter(mean_x_t_path_linear_with_gamma[:, 0], mean_x_t_path_linear_with_gamma[:, 1],
                color="C2", marker=".", s=0.5, label="geodesic")
    plt.scatter(correct_mean_x_t_path_linear_with_gamma_periodic[:, 0],
                correct_mean_x_t_path_linear_with_gamma_periodic[:, 1],
                color="C3", marker=".", s=0.5, label="geodesic reference mean")
    plt.scatter(mean_x_t_path_linear_with_gamma_periodic[:, 0], mean_x_t_path_linear_with_gamma_periodic[:, 1],
                color="C1", marker=".", s=0.5, label="wrong reference mean")

    plt.axvline(x=0.0, color="k")
    plt.axvline(x=1.0, color="k")
    plt.axhline(y=0.0, color="k")
    plt.axhline(y=1.0, color="k")
    plt.legend()

    plt.gca().set_aspect("equal")
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
