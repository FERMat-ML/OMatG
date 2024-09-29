import numpy as np
import scipy
# from MatterGen for lattice sampling
class NDependentScaledNormal:
    """
    Samples a scaled normal distribution which is scaled according to n, with c and v fixed by the given dataset
    """
    def __init__(self, c):
        self.c = c
        self.rng = np.random.default_rng()

    def __call__(self, n, size):
        mean = (n * self.c) ** (1/3)
        stdev = mean / 8
        val = self.rng.normal(mean, stdev, size)
        l = [val, val, val] * np.identity(3)
        print (l)
        return l

class NDependentGamma:
    def __init__(self, a, loc, scale):
        self.a = a
        self.loc = loc
        self.scale = scale
    def __call__(self, n):
        v = n/scipy.stats.gamma.rvs(self.a, self.scale, self.scale)
        a = v ** (1/3)
        cell = [a,a,a] * np.identity(3)
        return cell        

if __name__ == "__main__":
    from omg.sampler.sample_from_rng import SampleFromRNG
    from functools import partial
    import torch
    N = 10
    cell_sampler = NDependentGamma(6.950090673417738, 0.0011311460889336065, 0.008141385751601667)
    rng = np.random.default_rng()
    sampler = SampleFromRNG(cell_distribution=cell_sampler,n_particle_sampler=N)
    vol = []
    items = []
    for i in range (1000):
        #print ((sampler.sample_p_0().cell[0,:,:]))
        sample = sampler.sample_p_0()
        v = torch.abs(torch.linalg.det(sample.cell[0,:,:])).numpy()
        for i in range(3):
            items.append(sample.cell[0,:,:][i,i])
        vol.append(v)
    print (np.mean(vol))
    print (np.mean(items))


