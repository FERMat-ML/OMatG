import pytest
import torch
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.corrector import PeriodicBoundaryConditionsCorrector
from omg.si.interpolants import *
from omg.si.gamma import *
from omg.si.epsilon import *
from omg.globals import SMALL_TIME, BIG_TIME

# Testing parameters/objects
stol = 6.5e-2
eps = 1e-3
times = torch.linspace(SMALL_TIME+eps, BIG_TIME-eps, 100)
nrep = 10000
ptr = torch.arange(nrep) * 10

# Interpolants
interpolants = [
    LinearInterpolant(),
    TrigonometricInterpolant(),
    PeriodicLinearInterpolant(),
    EncoderDecoderInterpolant(),
    MirrorInterpolant(),
    ScoreBasedDiffusionModelInterpolant()
]

# Interpolant arguments
gammas = [
    LatentGammaEncoderDecoder(),
    LatentGammaSqrt(0.1)
]

# Epsilons
epsilons = [
    VanishingEpsilon(c=0.1),
    ConstantEpsilon(c=0.1)
]

def get_name(obj):
    return obj.__class__.__name__ if obj is not None else "None"

@pytest.mark.parametrize(
    "gamma, interpolant, epsilon", 
    [(gamma, interpolant, epsilon) for gamma in gammas for interpolant in interpolants for epsilon in epsilons],
    ids=[
        f"gamma={get_name(gamma)}, interpolant={get_name(interpolant)}, epsilon={get_name(epsilon)}"
        for gamma in gammas for interpolant in interpolants for epsilon in epsilons
    ]
)
def test_sde_integrator(interpolant, gamma, epsilon):
    '''
    Test interpolant integrator
    '''
    # Initialize
    corr = None
    x_init = torch.ones(size=(10, nrep)) * 0.10
    x_final = (torch.rand(size=(10,))).unsqueeze(-1).expand(10, nrep)
    batch_pointer = torch.tensor([0, 4, 7, 10]).unsqueeze(-1).expand(4, nrep)
    if isinstance(interpolant, PeriodicLinearInterpolant):
        corr = PeriodicBoundaryConditionsCorrector(min_value=0, max_value=1)
    if isinstance(interpolant, MirrorInterpolant):
        x_init = x_final

    # Design interpolant
    interpolant = SingleStochasticInterpolant(
        interpolant=interpolant, gamma=gamma,epsilon=epsilon,
        differential_equation_type='SDE', corrector=corr,
        integrator_kwargs={'method':'srk'}
    )

    # ODE function
    def velo(t, x):
        z = torch.randn(x_init.shape)
        return interpolant._interpolate_derivative(torch.tensor(t), x_init, x_final, z=z, batch_pointer=batch_pointer), z
    
    # Integrate
    x = x_init
    for i in range(1,len(times)):

        # Get time
        t_i = times[i-1]
        dt = times[i] - t_i
        x_interp_mean = interpolant.interpolate(times[i], x_init, x_final, batch_pointer)[0].mean(dim=-1)
        x_mean = interpolant._sde_integrate(velo, x, t_i, dt, batch_pointer).mean(dim=-1)
        x = x_mean.unsqueeze(-1).expand(10, nrep)

        # Assertion test
        assert x_mean == pytest.approx(x_interp_mean, abs=stol)
