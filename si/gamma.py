# Import libraries
import abc
import torch

#########
# GAMMA #
#########
class Gamma(abc.ABC):
    '''
    Abstract class for defining an interpolant
    '''

    def __init__(self, a=None):
        '''
        Construct interpolant
        '''
        
        # Construct
        super().__init__()
        self.a = a

    @abc.abstractmethod
    def __call__(t:torch.tensor, a=None):
        '''
        Function call to interpolant
        '''
        pass

    @abc.abstractmethod
    def compute_dt(t:torch.tensor, a=None):
        '''
        Call for time derivative of interpolant
        '''
        pass

'''
This file contains some predefined gamma functions
for tuning the stochasticity of the stochastic interpolant.
Other gamma functions can be defined by the user
'''
class GammaSqrt(Gamma):

    def __init__(self, a):
        '''
        Construct gamma that goes as sqrt(a*t*(1-t)) 
        '''
        super().__init__(a)

    def __call__(self, t:torch.tensor):
        '''
        Gamma term
        '''

        # Compute gamma
        return torch.sqrt(self.a * t * (1 - t))

    def compute_dt(self, t:torch.tensor):
        '''
        dGamma/dt
        '''

        # Compute dG/dt
        return (self.a * (1 - 2 * t)) / (2 * torch.sqrt(self.a * t * (1 - t)))

class GammaEncoderDecoder(Gamma):

    def __init__(self):
        '''
        Construct gamma that goes as sin^2(pi*t) 
        '''
        super().__init__(None)

    def __call__(self, t:torch.tensor):
        '''
        Gamma term
        '''

        # Compute gamma
        return torch.sin(torch.pi * t) ** 2

    def compute_dt(self, t:torch.tensor):
        '''
        dGamma/dt
        '''

        # Compute dG/dt
        return 2 * torch.sin(torch.pi * t) * torch.pi * torch.cos(torch.pi * t)
 
class GammaNull(Gamma):

    def __init__(self):
        '''
        Construct gamma that goes as sin^2(pi*t) 
        '''
        super().__init__(None)

    def __call__(self, t:torch.tensor):
        '''
        Gamma term
        '''

        # Compute gamma
        return 0

    def compute_dt(self, t:torch.tensor):
        '''
        dGamma/dt
        '''

        # Compute dG/dt
        return 0
 
     