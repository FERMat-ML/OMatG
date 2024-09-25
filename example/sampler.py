from omg.sampler import SampleFromRNG, SampleFromDataset
from omg.datamodule import DataModule
import numpy as np
from functools import partial

np.random.seed(0)


dm = DataModule.from_lmdb("./example.lmdb")

# Sample from the dataset
sampler = SampleFromDataset(dm)
print("Sample 1: ", sampler.sample_p_0())

# Sample from the dataset with fractional coordinates
sampler.set_frac_coords(False)
print("Sample 2: ", sampler.sample_p_0())

# Sample from random distributions with fixed number of particles
sampler = SampleFromRNG(n_particle_sampler=4)
print("Random Sample: ", sampler.sample_p_0())

# Sample from random distributions with a random number of particles
rng = np.random.default_rng()
n_particle_sampler = partial(rng.integers, low=10, high=15)

sampler = SampleFromRNG(n_particle_sampler=n_particle_sampler)
print("Random Sample: ", sampler.sample_p_0())

# Sample from random distributions with a random number of particles and fractional coordinates
sampler = SampleFromRNG(n_particle_sampler=n_particle_sampler, fractional_coordinates=True)
print("Random Sample: ", sampler.sample_p_0())
