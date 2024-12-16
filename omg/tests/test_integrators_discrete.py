import torch
import torch.nn.functional as functional
from omg.si.discrete_flow_matching_mask import DiscreteFlowMatchingMask
from omg.si.discrete_flow_matching_uniform import DiscreteFlowMatchingUniform
from omg.globals import SMALL_TIME, BIG_TIME, MAX_ATOM_NUM

# Testing parameters/objects
indices = torch.tensor([0 for _ in range(10)])


# Interpolants
def test_discrete_mask_integrator():

    # Initialize
    x_final = torch.randint(size=(10,), low=1, high=MAX_ATOM_NUM + 1).long()
    x_init = torch.zeros((10,)).long()
    interp = DiscreteFlowMatchingMask(noise=0.)

    # ODE function
    def velo(t, x):
        # Model does not know about masking index.
        x1 = functional.one_hot(x_final - 1, num_classes=MAX_ATOM_NUM).float()
        # One-hot encoding will be passed through softmax. Ensure that wrong types are getting zero.
        x1[x1 == 0] = -float("INF")
        return x1, None
    
    # Integrate
    x = x_init
    times = torch.linspace(SMALL_TIME, BIG_TIME, 100)
    for t_index in range(1, len(times)):
        t = times[t_index - 1]
        dt = times[t_index] - times[t_index - 1]
        x = interp.integrate(velo, x, t, dt, batch_indices=indices)

    # Assertion test
    assert torch.all(x == x_final)


# Interpolants
def test_discrete_uniform_integrator():

    # Initialize
    x_final = torch.randint(size=(10,), low=1, high=MAX_ATOM_NUM + 1).long()
    x_init = torch.randint(size=(10,), low=1, high=MAX_ATOM_NUM + 1).long()
    interp = DiscreteFlowMatchingUniform(noise=0.0)

    # ODE function
    def velo(t, x):
        x1 = functional.one_hot(x_final - 1, num_classes=MAX_ATOM_NUM).float()
        # One-hot encoding will be passed through softmax. Ensure that wrong types are getting zero.
        x1[x1 == 0] = -float("INF")
        return x1, None
    
    # Integrate
    x = x_init
    times = torch.linspace(SMALL_TIME, BIG_TIME, 100)
    for t_index in range(1, len(times)):
        t = times[t_index - 1]
        dt = times[t_index] - times[t_index - 1]
        x = interp.integrate(velo, x, t, dt, batch_indices=indices)

    # Assertion test
    assert torch.all(x == x_final)
