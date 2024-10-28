import pytest
import torch
import torch.nn.functional as functional
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.corrector import PeriodicBoundaryConditionsCorrector
from omg.si.discrete_flow_matching_mask import DiscreteFlowMatchingMask
from omg.si.discrete_flow_matching_uniform import DiscreteFlowMatchingUniform
from omg.globals import SMALL_TIME, BIG_TIME, MAX_ATOM_NUM

# Testing parameters/objects
#times = torch.linspace(0, 1, 100)
ptr = None


# Interpolants
def test_discrete_mask_integrator():

    # Initialize
    x_final = torch.randint(size=(10,), low=1, high=MAX_ATOM_NUM).long()
    x_init = torch.zeros((10,)).long()
    interp = DiscreteFlowMatchingMask(noise=0.)

    # ODE function
    def velo(x, t):
        x1 = functional.one_hot(x_final, num_classes=MAX_ATOM_NUM).float()
        x1[x1 == 0] = -float("INF")
        return x1, None
    
    # Integrate
    x = x_init
    t = 0.0
    dt = 0.001
    while t < 1.0:

        # Get time
        x = interp.integrate(velo, x, t, dt, batch_pointer=None)
        t += dt

    # Assertion test
    print(x)
    print(x_final)
    assert torch.all(x == x_final)

# Interpolants
def test_discrete_uniform_integrator():

    # Initialize
    x_final = torch.randint(size=(10,), low=1, high=MAX_ATOM_NUM).long()
    x_init = torch.randint(size=(10,), low=1, high=MAX_ATOM_NUM).long()
    interp = DiscreteFlowMatchingUniform(noise=0.0)

    # ODE function
    def velo(x, t):
        x1 = functional.one_hot(x_final, num_classes=MAX_ATOM_NUM).float()
        x1[x1 == 0] = -float("INF")
        return x1, None
    
    # Integrate
    x = x_init
    t = 0.0
    dt = 0.001
    while t < 1.0:

        # Get time
        x = interp.integrate(velo, x, t, dt, batch_pointer=None)
        t += dt

    # Assertion test
    assert torch.all(x == x_final)
