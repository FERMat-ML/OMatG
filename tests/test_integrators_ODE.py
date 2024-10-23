import pytest
import torch
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.corrector import PeriodicBoundaryConditionsCorrector
from omg.si.interpolants import *
from omg.si.gamma import *
from omg.si.epsilon import *
from omg.globals import SMALL_TIME, BIG_TIME

# Testing parameters/objects
tol = 5e-2
ptr = None
eps = 1e-3
times = torch.linspace(SMALL_TIME+eps, BIG_TIME-eps, 100)
nrep = 5000

# Interpolants
interpolants = [
    LinearInterpolant(),
    TrigonometricInterpolant(),
    #PeriodicLinearInterpolant(),
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
def test_integrator(interpolant, gamma):
    '''
    Test interpolant integrator
    '''
    # Initialize
    corr = None
    x_init = torch.zeros(size=(2, 2))
    x_final = torch.rand(size=(2, 2))
    if isinstance(interpolant, PeriodicLinearInterpolant):
        corr = PeriodicBoundaryConditionsCorrector(min_value=0, max_value=1)
    if isinstance(interpolant, MirrorInterpolant):
        x_init = x_final

    if isinstance(gamma, LatentGammaSqrt) or isinstance(gamma, LatentGammaEncoderDecoder):
        lat_flag = True
        x_init = x_init.unsqueeze(2).expand(2, 2, nrep)
        x_final = x_final.unsqueeze(2).expand(2, 2, nrep)
    else:
        lat_flag=False

    # Design interpolant
    interpolant = SingleStochasticInterpolant(
        interpolant=interpolant, gamma=gamma,epsilon=None,
        differential_equation_type='ODE', corrector=corr,
        integrator_kwargs={'method':'euler'}
    )

    # ODE function
    def velo(t, x):
        return [interpolant._interpolate_derivative(torch.tensor(t), x_init, x_final, z=torch.randn(x_init.shape), batch_pointer=None)]
    
    # Integrate
    x = x_init
    for i in range(1,len(times)):

        # Get time
        t_i = times[i-1]
        dt = times[i] - t_i

        # If stochastic element
        if lat_flag:
            x_interp_mean = interpolant.interpolate(times[i], x_init, x_final, ptr)[0].mean(dim=-1)
            x_mean = interpolant._ode_integrate(velo, x, t_i, dt).mean(dim=-1)
            x = x_mean.unsqueeze(2).expand(2, 2, nrep)

            # Assertion test
            assert x_mean == pytest.approx(x_interp_mean, abs=tol)

        # If all deterministic
        else:
       
            # Interpolate
            x_interp, z = interpolant.interpolate(times[i], x_init, x_final, ptr)
            x_new = interpolant._ode_integrate(velo, x, t_i, dt)
            x = x_new

            # Test for equality
            assert x == pytest.approx(x_interp, abs=tol)
