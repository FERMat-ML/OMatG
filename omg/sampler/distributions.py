import numpy as np
import scipy
# TODO: !!!Highly WIP!!! from MatterGen for lattice sampling. Not sure it actually works like they say it does!
'''
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
'''
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

class MaskDistribution:
    def __init__(self, token=0):
        self.token = token
    def __call__(self, size):
        return np.ones(size) * self.token