import torch

from omg.sampler import SampleFromRNG, SampleFromDataset
from omg.datamodule import DataModule
#import numpy as np
from functools import partial

torch.manual_seed(0)

dm = DataModule(["./example.lmdb"])

# Sample from the dataset
sampler = SampleFromDataset(dm, batch_size=3)
print("Sample 1: ", sampler.sample_p_0())

# Sample from the dataset with fractional coordinates
sampler = SampleFromDataset(dm, convert_to_fractional=True, batch_size=3)
print("Sample 2: ", sampler.sample_p_0())

# Sample from random distributions with fixed number of particles
sampler = SampleFromRNG(n_particle_sampler=4)
print("Random Sample: ", sampler.sample_p_0())

# Sample from random distributions with a random number of particles
# rng = np.random.default_rng()

def n_particle_sampler():
    return torch.randint(10,15, size=(1,))

sampler = SampleFromRNG(n_particle_sampler=n_particle_sampler, batch_size=3)
print("Random Sample: ", sampler.sample_p_0())

x1 = sampler.sample_p_0()
print("x1:", x1)
print("Random Sample from x1: ", sampler.sample_p_0(x1))

# Sample from random distributions with a random number of particles and fractional coordinates
sampler = SampleFromRNG(n_particle_sampler=n_particle_sampler, convert_to_fractional=True)
print("Random Sample: ", sampler.sample_p_0())
