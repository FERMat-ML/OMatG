from omg.datamodule.dataloader import OMGTorchDataset, OMGDataModule
from omg.datamodule.datamodule import DataModule
from omg.sampler.sample_from_rng import SampleFromRNG
from omg.sampler.distributions import NDependentGamma
import torch
from omg.si.stochastic_interpolants import StochasticInterpolants
from omg.si.discrete_flow_matching_uniform import DiscreteFlowMatchingUniform
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.interpolants import PeriodicLinearInterpolant
from omg.si.corrector import PeriodicBoundaryConditionsCorrector
from omg.si.interpolants import LinearInterpolant
from omg.model.model import Model
from omg.model.encoders.cspnet_full import CSPNetFull
from omg.model.heads.pass_through import PassThrough
from omg.model.model_utils import SinusoidalTimeEmbeddings
from functools import partial
from omg.utils import xyz_saver
import sys
from torch_geometric.data import Data
from omg.globals import SMALL_TIME, BIG_TIME

# Set up dataloader. Make sure to run from repo base directory
ds = OMGTorchDataset(DataModule(['data/mp_20/val.lmdb']))
dl = OMGDataModule(train_dataset=ds, batch_size=1)

# Set up sampler. This will use the new gamma distribution for lattice and defaults for the others
sampler = SampleFromRNG(cell_distribution=NDependentGamma(a=6.950090673417738,loc=0.0011311460889336065,scale=0.008141385751601667))

STEPS = 10
# Set up SI class. Essentially same as that constructed from conf file
si = StochasticInterpolants(stochastic_interpolants=[DiscreteFlowMatchingUniform(number_integration_steps=STEPS, noise=0),SingleStochasticInterpolant(interpolant=PeriodicLinearInterpolant(),gamma=None, epsilon=None,differential_equation_type="ODE",corrector=PeriodicBoundaryConditionsCorrector(min_value=0, max_value=1)),SingleStochasticInterpolant(interpolant=LinearInterpolant(),gamma=None, epsilon=None,differential_equation_type="ODE") ], data_fields=['species', 'pos', 'cell'], integration_time_steps=STEPS)

# Sample p_1 and p_0
x_1 = next(iter(dl.train_dataloader()))
x_0 = sampler.sample_p_0(x_1)

# Print x_0 and x_1 info. Comment out certain values if you only want to look at one type at a time
print ('=========X0=========')
print (x_0.species)
print (x_0.pos)
print (x_0.cell)
print ('=========X1=========')
print (x_1.species)
print (x_1.pos)
print (x_1.cell)
print ()
print ()
# Interpolate
times = torch.linspace(SMALL_TIME, BIG_TIME, STEPS)
for time in times:
    print (f'========={time}=========')
    x_t, _ = si._interpolate(torch.tensor([time]), x_0, x_1)
    print (x_t.species)
    print (x_t.pos)
    print (x_t.cell)
    print()
    print()
# Check that x_t t->1 is roughly equal to x_1
print (f'=========Check Differences=========')
print (x_1.species - x_t.species)
print (x_1.pos - x_t.pos)
print (x_1.cell - x_t.cell)
