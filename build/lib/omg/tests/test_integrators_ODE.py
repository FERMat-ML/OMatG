import pytest
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.interpolants import *
from omg.si.gamma import *
from omg.si.epsilon import *
from omg.si.tau import TauConstantSchedule
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
    ScoreBasedDiffusionModelInterpolantVP(tau=TauConstantSchedule()),
    PeriodicScoreBasedDiffusionModelInterpolantVP(tau=TauConstantSchedule()),
    PeriodicTrigonometricInterpolant(),
    PeriodicEncoderDecoderInterpolant(),
]

# Interpolant arguments
gammas = [
    None,
    LatentGammaEncoderDecoder(),
    LatentGammaSqrt(1.0)
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
    x_init = torch.ones(size=(10,)) * 0.1
    batch_indices = torch.tensor([0 for _ in range(10)])
    x_final = torch.rand(size=(10,))
    if isinstance(interpolant, MirrorInterpolant):
        x_init = x_final.clone()

    if isinstance(interpolant, (PeriodicLinearInterpolant, PeriodicScoreBasedDiffusionModelInterpolantVP,
                                PeriodicTrigonometricInterpolant, PeriodicEncoderDecoderInterpolant)):
        pbc_flag = True
        interpolant_geodesic = SingleStochasticInterpolant(
            interpolant=interpolant, gamma=None,epsilon=None,
            differential_equation_type='ODE',
            integrator_kwargs={'method':'rk4'})
    else:
        pbc_flag = False

    if isinstance(gamma, (LatentGammaSqrt, LatentGammaEncoderDecoder)):
        lat_flag = True
        x_init = x_init.unsqueeze(-1).expand(10, nrep)
        x_final = x_final.unsqueeze(-1).expand(10, nrep)
    else:
        lat_flag=False

    # Design interpolant
    interpolant = SingleStochasticInterpolant(
        interpolant=interpolant, gamma=gamma,epsilon=None,
        differential_equation_type='ODE',
        integrator_kwargs={'method':'rk4'}
    )

    # ODE function
    def velo(t, x):
        return (interpolant._interpolate_derivative(torch.tensor(t), x_init, x_final, z=torch.randn(x_init.shape)),
                torch.tensor(torch.nan))
    
    def pbc_mean(x, x_ref):
        # assuming pbcs from 0 to 1
        dists = torch.abs(x - x_ref) 
        x_prime = torch.where(dists >=0.5, x + torch.sign(x_ref - 0.5), x)
        return x_prime.mean(dim=-1) % 1.
    
    # Integrate
    x = x_init
    for i in range(1,len(times)):

        # Get time
        t_i = times[i-1]
        dt = times[i] - t_i

        # If stochastic element
        if lat_flag:
            x_interp = interpolant.interpolate(times[i], x_init, x_final, batch_indices)[0]
            x_new = interpolant._ode_integrate(velo, x, t_i, dt, batch_indices)

            if pbc_flag:
                x_new_geodesic = interpolant_geodesic.interpolate(times[i], x_init, x_final, batch_indices)[0]
                # Use COM and PBCs to find average x_new
                x_ref = x_new_geodesic
                x_new_mean = pbc_mean(x_new, x_ref)
                x_interp_mean = pbc_mean(x_interp, x_ref)
                x = x_new_mean.unsqueeze(-1).expand(10, nrep)
                
                # Find closest images of the means of x_new and x_interp
                diff = torch.abs(x_interp_mean - x_new_mean)
                x_interp_mean_prime = torch.where(diff >= 0.5, x_interp_mean + torch.sign(x_new_mean - 0.5), x_interp_mean)
                assert x_new_mean == pytest.approx(x_interp_mean_prime, abs=stol)
            else:
                # Set every x to the mean of the batch.
                x = x_new.mean(dim=-1).unsqueeze(-1).expand(10, nrep)
                # Take mean across batches.
                x_interp_mean = x_interp.mean(dim=-1)
                x_new_mean = x_new.mean(dim=-1)
                assert x_new_mean == pytest.approx(x_interp_mean, abs=stol)

        # If all deterministic
        else:
            # Interpolate
            x_interp = interpolant.interpolate(times[i], x_init, x_final, batch_indices)[0]
            x_new = interpolant._ode_integrate(velo, x, t_i, dt, batch_indices)

            # Test for equality
            if pbc_flag:
                x = x_new
                # assume pbc is from 0 - 1
                diff = torch.abs(x_interp - x_new)
                x_interp_prime = torch.where(diff >= 0.5, x_interp + torch.sign(x_new - 0.5), x_interp)
                assert x_new == pytest.approx(x_interp_prime, abs=tol)
            else:
                x = x_new
                assert x_new == pytest.approx(x_interp, abs=tol)
