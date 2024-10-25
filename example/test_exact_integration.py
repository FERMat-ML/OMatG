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
from omg.globals import SMALL_TIME, BIG_TIME, MAX_ATOM_NUM
from torch.nn import functional

# Set up dataloader. Make sure to run from repo base directory
ds = OMGTorchDataset(DataModule(['data/mp_20/val.lmdb']))
dl = OMGDataModule(train_dataset=ds, batch_size=1)

# Set up sampler. This will use the new gamma distribution for lattice and defaults for the others
sampler = SampleFromRNG(cell_distribution=NDependentGamma(a=6.950090673417738,loc=0.0011311460889336065,scale=0.008141385751601667))

STEPS = 10
# Set up SI classes. Will test each integration separately. 
# TODO: Need to look at how to do this with discrete species without too much of a hack
pos_si = SingleStochasticInterpolant(interpolant=PeriodicLinearInterpolant(),gamma=None, epsilon=None,differential_equation_type="ODE",corrector=PeriodicBoundaryConditionsCorrector(min_value=0, max_value=1)) 

cell_si = SingleStochasticInterpolant(interpolant=LinearInterpolant(),gamma=None, epsilon=None,differential_equation_type="ODE")

species_si = DiscreteFlowMatchingUniform(noise=0)

# Sample p_1 and p_0
x_1 = next(iter(dl.train_dataloader()))
x_0 = sampler.sample_p_0(x_1)

# Set up some "model" functions which just return the exact b field
# These models need take x_t and t as arguments. But for pos and cell ground truth b field calculation is independent of both
def pos_model(x_t, t):
    gt_b = pos_si._interpolant.interpolate_derivative(torch.tensor([times[0]]), x_0.pos, x_1.pos, x_0.ptr)
    return gt_b

def cell_model(x_t, t):
    gt_b = cell_si._interpolant.interpolate_derivative(torch.tensor([times[0]]), x_0.cell, x_1.cell, x_0.ptr)
    return gt_b

def species_model(x_t, t):
    gt = functional.one_hot(x_t - 1, MAX_ATOM_NUM).double()
    gt[gt == 0.] = -float("INF")
    return gt # softmax of this is equivalent to one hot from above

def pos_model_wrapper(t, x): # adapted from model_prediction_fn in si.stochastic_interpolants 
    t = torch.tensor(t)
    x = torch.tensor(x)
    t = t.repeat(1,)
    b = pos_model(x, t)
    b = b.reshape((-1,))
    return b, None

def cell_model_wrapper(t, x): # adapted from model_prediction_fn in si.stochastic_interpolants 
    t = torch.tensor(t)
    x = torch.tensor(x)
    t = t.repeat(1,)
    b = cell_model(x, t)
    b = b.reshape((-1,))
    return b, None

def species_model_wrapper(t, x): # adapted from model_prediction_fn in si.stochastic_interpolants 
    t = t.repeat(1,)
    b = species_model(x, t)
    b = b.reshape((-1,))
    return b, None

# Define times
STEPS=10
times = torch.linspace(SMALL_TIME, BIG_TIME, STEPS)


# Check integration of pos
pos = x_0.pos
print ('=========Positions=========')
for i in range(1,len(times)):
    print (f'========={times[i]}=========')
    x_t_true, _ = pos_si.interpolate(times[i], x_0.pos, x_1.pos, x_0.ptr)
    print ('---------Interpolated Value---------')
    print (x_t_true)
    pos = pos_si._ode_integrate(pos_model_wrapper, pos, (float(times[i - 1]), float(times[i])))
    print ('---------Integrated Value---------')
    print (pos)
    print ('---------Difference---------')
    print (x_t_true - pos)
    print ()
    print ()
# Check integration of cell
cell = x_0.cell
print ('=========Cell=========')
for i in range(1,len(times)):
    print (f'========={times[i]}=========')
    x_t_true, _ = cell_si.interpolate(times[i], x_0.cell, x_1.cell, x_0.ptr)
    print ('---------Interpolated Value---------')
    print (x_t_true)
    cell = cell_si._ode_integrate(cell_model_wrapper, cell, (float(times[i - 1]), float(times[i])))
    print ('---------Integrated Value---------')
    print (cell)
    print ('---------Difference---------')
    print (x_t_true - cell)
    print ()
    print ()

# Check integration of species
print ('=========Species=========')
for i in range(1,len(times)):
    print (f'========={times[i]}=========')
    x_t_true, _ = species_si.interpolate(times[i], x_0.species, x_1.species, x_0.ptr)
    print ('---------Interpolated Value---------')
    print (x_t_true)
    species = x_t_true.clone()
    species = species_si.integrate(species_model_wrapper, species, (float(times[i - 1]), float(times[i])))
    print ('---------Integrated Value---------')
    print (species)
    print ('---------Difference---------')
    print (x_t_true - species)
    print ()
    print ()
