import pytest
import torch
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.interpolants import *
from omg.si.gamma import *
from omg.si.epsilon import *
from omg.globals import SMALL_TIME, BIG_TIME

# Testing parameters/objects
stol = 6.5e-2
eps = 1e-3
times = torch.linspace(SMALL_TIME+eps, BIG_TIME-eps, 100)
nrep = 10000
ptr = torch.arange(nrep+1) * 10

# Interpolants
interpolants = [
    LinearInterpolant(),
    TrigonometricInterpolant(),
    PeriodicLinearInterpolant(),
    EncoderDecoderInterpolant(),
    MirrorInterpolant(),
    ScoreBasedDiffusionModelInterpolant(),
    PeriodicScoreBasedDiffusionModelInterpolant()
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
    x_init = torch.ones(size=(10, nrep)) * 0.10
    x_final = (torch.rand(size=(10,))).unsqueeze(-1).expand(10, nrep)
    if isinstance(interpolant, MirrorInterpolant):
        x_init = x_final

    if isinstance(interpolant, (PeriodicLinearInterpolant, PeriodicScoreBasedDiffusionModelInterpolant)):
        pbc_flag = True
    else:
        pbc_flag = False

    # Design interpolant
    interpolant = SingleStochasticInterpolant(
        interpolant=interpolant, gamma=gamma,epsilon=epsilon,
        differential_equation_type='SDE',
        integrator_kwargs={'method':'srk'}
    )

    # ODE function
    def velo(t, x):
        z = torch.randn(x_init.shape)
        return interpolant._interpolate_derivative(torch.tensor(t), x_init, x_final, z=z, batch_pointer=ptr), z
    
    # Integrate
    x = x_init
    for i in range(1,len(times)):

        # Get time
        t_i = times[i-1]
        dt = times[i] - t_i

        # Assertion test
        if pbc_flag:
            # assume pbc is from 0 - 1
            x_interp = interpolant.interpolate(times[i], x_init, x_final, ptr)[0]
            x = interpolant._sde_integrate(velo, x, t_i, dt, ptr)
            diff = torch.abs(x_interp - x)
            x_interp_prime = torch.where(diff >= 0.5, x_interp + torch.sign(x - 0.5), x_interp)
            assert x.mean(dim=-1) == pytest.approx(x_interp_prime.mean(dim=-1), abs=stol)
        else:
            x_interp_mean = interpolant.interpolate(times[i], x_init, x_final, ptr)[0].mean(dim=-1)
            x = interpolant._sde_integrate(velo, x, t_i, dt, ptr)
            assert x.mean(dim=-1) == pytest.approx(x_interp_mean, abs=stol)
