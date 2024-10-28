import pytest
import torch
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.stochastic_interpolants import StochasticInterpolants
from omg.si.corrector import PeriodicBoundaryConditionsCorrector
from omg.si.interpolants import *
from omg.si.gamma import *
from omg.si.epsilon import *
from omg.si.discrete_flow_matching_mask import DiscreteFlowMatchingMask
from torch_geometric.data import Data
import torch.nn.functional as functional
from omg.globals import SMALL_TIME, BIG_TIME, MAX_ATOM_NUM

# Testing parameters/objects
stol = 6.5e-2
tol = 1e-2
eps = 1e-6
times = torch.linspace(SMALL_TIME, BIG_TIME, 100)
nrep = 1000
ptr = torch.arange(nrep+1) * 10
n_atoms = torch.ones(size=(nrep,)) * 10

def test_coupled_integrator():
    '''
    Test interpolant integrator
    '''
    # Initialize three interoplants
    ode_interp = SingleStochasticInterpolant(
        interpolant=LinearInterpolant(), gamma=None, epsilon=None, 
        differential_equation_type='ODE',corrector=None, integrator_kwargs={'method':'rk4'}
    )
    sde_interp = SingleStochasticInterpolant(
        interpolant=LinearInterpolant(), gamma=LatentGammaSqrt(0.1), epsilon=VanishingEpsilon(c=0.1),
        differential_equation_type='SDE', corrector=None, integrator_kwargs={'method':'srk'}
    )
    discrete_interp = DiscreteFlowMatchingMask(noise=0.0)

    # Sequence
    interp_seq = [ode_interp, sde_interp, discrete_interp]
    coupled_interp = StochasticInterpolants(
        stochastic_interpolants=interp_seq, data_fields=['pos', 'cell', 'species'], 
        integration_time_steps=100
    )

    # Set up data dictionary
    x_0 = Data(pos=torch.rand(size=(10,)).unsqueeze(-1).expand(10, nrep), cell=torch.rand(size=(10,)).unsqueeze(-1).expand(10, nrep), species=torch.zeros(size=(10,)).long().unsqueeze(-1).expand(10, nrep).reshape((-1,)), ptr=ptr, n_atoms=n_atoms)
    x_1 = Data(pos=torch.rand(size=(10,)).unsqueeze(-1).expand(10, nrep), cell=torch.zeros(size=(10,)).unsqueeze(-1).expand(10, nrep), species=torch.randint(size=(10,), low=1, high=MAX_ATOM_NUM).long().unsqueeze(-1).expand(10, nrep).reshape((-1,)), ptr=ptr, n_atoms=n_atoms)

    # ODE function
    def velo(t, x):

        # Velocities
        z = torch.randn(size=(10, nrep))
        pos_b = ode_interp._interpolate_derivative(t, x_0.pos, x_1.pos, z=None, batch_pointer=ptr)
        cell_b = sde_interp._interpolate_derivative(t, x_0.cell, x_1.cell, z=z, batch_pointer=ptr)
        x1_spec = functional.one_hot(x_1.species, num_classes=MAX_ATOM_NUM).float()
        x1_spec[x1_spec == 0] = -float("INF")
        species_b = x1_spec

        # Stochastic variable
        cell_eta = z

        # Return
        return Data(pos_b=pos_b, pos_eta=pos_b, cell_b=cell_b, cell_eta=cell_eta, species_b=species_b, species_eta=species_b, ptr=ptr)
    
    # Integrate
    x, inter = coupled_interp.integrate(x_0, velo, save_intermediate=True)

    # Loop over times for average
    for i in range(0, len(times)):

        # Get average
        cell_avg = inter[i].cell.mean(dim=-1)

        # True value
        cell_true = sde_interp.interpolate(times[i], x_0.cell, x_1.cell, batch_pointer=ptr)[0].mean(dim=-1)
        pos_true = ode_interp.interpolate(times[i], x_0.pos, x_1.pos, batch_pointer=ptr)[0]

        # Check approximation
        assert inter[i].pos == pytest.approx(pos_true, abs=tol)
        assert cell_avg == pytest.approx(cell_true, abs=stol)
        print(i)

    # Check at the end for discrete
    assert torch.all(x.species == x_1.species)