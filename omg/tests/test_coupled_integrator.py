import pytest
import torch
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.stochastic_interpolants import StochasticInterpolants
from omg.si.interpolants import *
from omg.si.gamma import *
from omg.si.epsilon import *
from omg.si.discrete_flow_matching_mask import DiscreteFlowMatchingMask
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
indices = torch.repeat_interleave(torch.arange(nrep), 3)
n_atoms = torch.ones(size=(nrep,)) * 3

def test_coupled_integrator():
    '''
    Test interpolant integrator
    '''
    # Initialize three interpolants
    pos_interp = SingleStochasticInterpolant(
        interpolant=PeriodicLinearInterpolant(), gamma=None, epsilon=None, 
        differential_equation_type='ODE', integrator_kwargs={'method':'rk4'}
    )
    cell_interp = SingleStochasticInterpolant(
        interpolant=LinearInterpolant(), gamma=LatentGammaSqrt(0.1), epsilon=VanishingEpsilon(c=0.1),
        differential_equation_type='SDE', integrator_kwargs={'method':'srk'}
    )
    discrete_interp = DiscreteFlowMatchingMask(noise=0.0)

    # Sequence
    interp_seq = [pos_interp, cell_interp, discrete_interp]
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
    x_0 = Data(pos=x_0_pos, cell=x_0_cell, species=x_0_spec, batch=indices, n_atoms=n_atoms)
    x_1 = Data(pos=x_1_pos, cell=x_1_cell, species=x_1_spec, batch=indices, n_atoms=n_atoms)

    # ODE function
    def velo(x, t):

        # Velocities
        z_x = torch.randn_like(x.pos)
        z_cell = torch.randn_like(x.cell)
        t_pos = reshape_t(t, n_atoms.long(), DataField.pos)
        t_cell = reshape_t(t, n_atoms.long(), DataField.cell)
        pos_b = pos_interp._interpolate_derivative(t_pos, x_0.pos, x_1.pos, z=z_x)
        cell_b = cell_interp._interpolate_derivative(t_cell, x_0.cell, x_1.cell, z=z_cell)
        x1_spec = functional.one_hot(x_1.species - 1, num_classes=MAX_ATOM_NUM).float()
        x1_spec[x1_spec == 0] = -float("INF")
        species_b = x1_spec

        # Stochastic variable
        cell_eta = z_cell

        # Return
        return Data(pos_b=pos_b, pos_eta=pos_b, cell_b=cell_b, cell_eta=cell_eta, species_b=species_b, species_eta=species_b, batch=indices)
    
    # Integrate
    x, inter = coupled_interp.integrate(x_0, velo, save_intermediate=True)

    # Loop over times for average
    for i in range(0, len(times)):

        # Get average
        cell_avg = inter[i].cell.mean(dim=0)

        # True value
        cell_true = cell_interp.interpolate(times[i], x_0.cell, x_1.cell, batch_indices=indices)[0].mean(dim=0)
        pos_true = pos_interp.interpolate(times[i], x_0.pos, x_1.pos, batch_indices=indices)[0]

        # Check approximation
        pos_diff = torch.abs(pos_true - inter[i].pos)
        pos_true_prime = torch.where(pos_diff >= 0.5, pos_true + torch.sign(inter[i].pos - 0.5), pos_true)
        assert inter[i].pos == pytest.approx(pos_true_prime, abs=tol)
        assert cell_avg == pytest.approx(cell_true, abs=stol)

    # Check at the end for discrete
    print(x.species[torch.where(x.species != x_1.species)])
    assert torch.all(x.species == x_1.species)