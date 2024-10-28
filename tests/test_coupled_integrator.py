import pytest
import torch
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.stochastic_interpolants import StochasticInterpolants
from omg.si.corrector import PeriodicBoundaryConditionsCorrector
from omg.si.interpolants import *
from omg.si.gamma import *
from omg.si.epsilon import *
from omg.si.discrete_flow_matching_mask import DiscreteFlowMatchingMask
from omg.utils import reshape_t, DataField
from torch_geometric.data import Data
import torch.nn.functional as functional
from omg.globals import SMALL_TIME, BIG_TIME, MAX_ATOM_NUM
from omg.utils import reshape_t, DataField

# Testing parameters/objects
stol = 6.5e-2
tol = 1e-2
eps = 1e-6
times = torch.linspace(SMALL_TIME, BIG_TIME, 100)
nrep = 1000
ptr = torch.arange(nrep+1) * 3
n_atoms = torch.ones(size=(nrep,)) * 3

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
    x_0_pos = torch.rand(size=(1, 3)).repeat(nrep, 1)
    x_1_pos = torch.rand(size=(1, 3)).repeat(nrep, 1)
    x_0_cell = torch.rand(size=(1, 3, 3)).repeat(nrep, 1, 1)
    x_1_cell = torch.zeros(size=(1, 3, 3)).repeat(nrep, 1, 1)
    x_0_spec = torch.zeros(size=(3 * nrep,)).long()
    x_1_spec = torch.randint(size=(3 * nrep,), low=1, high=MAX_ATOM_NUM + 1).long()
    x_0 = Data(pos=x_0_pos, cell=x_0_cell, species=x_0_spec, ptr=ptr, n_atoms=n_atoms)
    x_1 = Data(pos=x_1_pos, cell=x_1_cell, species=x_1_spec, ptr=ptr, n_atoms=n_atoms)

    # ODE function
    def velo(x, t):

        # Velocities
        z_x = torch.randn_like(x.pos)
        z_cell = torch.randn_like(x.cell)
        t_pos = reshape_t(t, n_atoms.long(), DataField.pos)
        t_cell = reshape_t(t, n_atoms.long(), DataField.cell)
        pos_b = ode_interp._interpolate_derivative(t_pos, x_0.pos, x_1.pos, z=z_x, batch_pointer=ptr)
        cell_b = sde_interp._interpolate_derivative(t_cell, x_0.cell, x_1.cell, z=z_cell, batch_pointer=ptr)
        x1_spec = functional.one_hot(x_1.species - 1, num_classes=MAX_ATOM_NUM).float()
        x1_spec[x1_spec == 0] = -float("INF")
        species_b = x1_spec

        # Stochastic variable
        cell_eta = z_cell

        # Return
        return Data(pos_b=pos_b, pos_eta=pos_b, cell_b=cell_b, cell_eta=cell_eta, species_b=species_b, species_eta=species_b, ptr=ptr)
    
    # Integrate
    x, inter = coupled_interp.integrate(x_0, velo, save_intermediate=True)

    # Loop over times for average
    for i in range(0, len(times)):

        # Get average
        cell_avg = inter[i].cell.mean(dim=0)

        # True value
        cell_true = sde_interp.interpolate(times[i], x_0.cell, x_1.cell, batch_pointer=ptr)[0].mean(dim=0)
        pos_true = ode_interp.interpolate(times[i], x_0.pos, x_1.pos, batch_pointer=ptr)[0]

        # Check approximation
        assert inter[i].pos == pytest.approx(pos_true, abs=tol)
        assert cell_avg == pytest.approx(cell_true, abs=stol)

    # Check at the end for discrete
    print(x.species[torch.where(x.species != x_1.species)])
    assert torch.all(x.species == x_1.species)