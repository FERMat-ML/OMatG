import pytest
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.interpolants import *
from omg.si.gamma import *
from omg.si.epsilon import *
from omg.si.tau import TauConstantSchedule
from omg.globals import SMALL_TIME, BIG_TIME

# Testing parameters/objects
stol = 6.5e-2
eps = 1e-3
times = torch.linspace(SMALL_TIME+eps, BIG_TIME-eps, 100)
nrep = 10000
indices = torch.repeat_interleave(torch.arange(nrep), 10)

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
    LatentGammaEncoderDecoder(),
    LatentGammaSqrt(0.1)
]

# Epsilons
epsilons = [
    VanishingEpsilon(c=0.05),
    ConstantEpsilon(c=0.05)
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
        x_init = x_final.clone().detach()

    if isinstance(interpolant, (PeriodicLinearInterpolant, PeriodicScoreBasedDiffusionModelInterpolantVP,
                                PeriodicTrigonometricInterpolant, PeriodicEncoderDecoderInterpolant)):
        pbc_flag = True
        interpolant_geodesic = SingleStochasticInterpolant(
            interpolant=interpolant, gamma=None,epsilon=None,
            differential_equation_type='ODE',
            integrator_kwargs={'method':'rk4'}
            )
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
        return interpolant._interpolate_derivative(torch.tensor(t), x_init, x_final, z=z), z
    
    def pbc_mean(x, x_ref):
        # assuming pbcs from 0 to 1

        # find distances to arbitrary element (0th) in nreps
        dists = torch.abs(x - x_ref) 
        x_prime = torch.where(dists >=0.5, x + torch.sign(x_ref - 0.5), x)
        return x_prime.mean(dim=-1) % 1.

    # Integrate
    x = x_init
    for i in range(1,len(times)):

        # Get time
        t_i = times[i-1]
        dt = times[i] - t_i

        x_new = interpolant._sde_integrate(velo, x, t_i, dt, indices)

        # Assertion test
        if pbc_flag:
            x_interp = interpolant.interpolate(times[i], x_init, x_final, indices)[0]
            x_new_geodesic = interpolant_geodesic.interpolate(times[i], x_init, x_final, indices)[0]
            
            # assume pbc is from 0 - 1
            x_ref = x_new_geodesic
            x_new_mean = pbc_mean(x_new, x_ref)
            x_interp_mean = pbc_mean(x_interp, x_ref)
            x = x_new_mean.unsqueeze(-1).expand(10, nrep)
            
            # Find closest images of x_new and x_interp
            diff = torch.abs(x_interp_mean - x_new_mean)
            x_interp_mean_prime = torch.where(diff >= 0.5, x_interp_mean + torch.sign(x_new_mean - 0.5), x_interp_mean)
            assert x_new_mean == pytest.approx(x_interp_mean_prime, abs=stol)

        else:
            x_interp_mean = interpolant.interpolate(times[i], x_init, x_final, indices)[0].mean(dim=-1)
            x = x_new
            assert x_new.mean(dim=-1) == pytest.approx(x_interp_mean, abs=stol)
