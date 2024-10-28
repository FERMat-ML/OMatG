import pytest
import torch
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.corrector import PeriodicBoundaryConditionsCorrector
from omg.si.interpolants import *
from omg.si.gamma import *
from omg.si.epsilon import *
from omg.globals import SMALL_TIME, BIG_TIME

# Testing parameters/objects
tol = 1e-2
stol = 6e-2
times = torch.linspace(SMALL_TIME, BIG_TIME, 200)
nrep = 10000

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
    None,
    LatentGammaEncoderDecoder(),
    LatentGammaSqrt(.1)
]

def get_name(obj):
    return obj.__class__.__name__ if obj is not None else "None"

@pytest.mark.parametrize(
    "gamma, interpolant", 
    [(gamma, interpolant) for gamma in gammas for interpolant in interpolants],
    ids=[
        f"gamma={get_name(gamma)}, interpolant={get_name(interpolant)}"
        for gamma in gammas for interpolant in interpolants
    ]
)
def test_ode_integrator(interpolant, gamma):
    '''
    Test interpolant integrator
    '''
    # Initialize
    corr = None
    x_init = torch.ones(size=(10,)) * 0.1
    batch_pointer = torch.tensor([0, 10])

    x_final = torch.rand(size=(10,))
    if isinstance(interpolant, PeriodicLinearInterpolant):
        corr = PeriodicBoundaryConditionsCorrector(min_value=0, max_value=1)
    if isinstance(interpolant, MirrorInterpolant):
        x_init = x_final

    if isinstance(gamma, LatentGammaSqrt) or isinstance(gamma, LatentGammaEncoderDecoder):
        lat_flag = True
        x_init = x_init.unsqueeze(-1).expand(10, nrep)
        x_final = x_final.unsqueeze(-1).expand(10, nrep)
    else:
        lat_flag=False

    # Design interpolant
    interpolant = SingleStochasticInterpolant(
        interpolant=interpolant, gamma=gamma,epsilon=None,
        differential_equation_type='ODE', corrector=corr,
        integrator_kwargs={'method':'rk4'}
    )

    # ODE function
    def velo(t, x):
        return (interpolant._interpolate_derivative(torch.tensor(t), x_init, x_final, z=torch.randn(x_init.shape),
                                                    batch_pointer=batch_pointer), torch.tensor(torch.nan))
    
    # Integrate
    x = x_init
    for i in range(1,len(times)):

        # Get time
        t_i = times[i-1]
        dt = times[i] - t_i

        # If stochastic element
        if lat_flag:
            x_interp_mean = interpolant.interpolate(times[i], x_init, x_final, batch_pointer)[0].mean(dim=-1)
            x_mean = interpolant._ode_integrate(velo, x, t_i, dt, batch_pointer).mean(dim=-1)
            x = x_mean.unsqueeze(-1).expand(10, nrep)

            # Assertion test
            assert x_mean == pytest.approx(x_interp_mean, abs=stol)

        # If all deterministic
        else:
       
            # Interpolate
            x_interp, z = interpolant.interpolate(times[i], x_init, x_final, batch_pointer)
            x_new = interpolant._ode_integrate(velo, x, t_i, dt, batch_pointer)
            x = x_new

            # Test for equality
            assert x == pytest.approx(x_interp, abs=tol)
