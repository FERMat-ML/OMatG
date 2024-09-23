# Import libraries
import abc
import torch

################
# INTERPOLANTS #
################
class Interpolant(abc.ABC):
    '''
    Abstract class for defining an interpolant
    '''

    def __init__(self, switch=None):
        '''
        Construct interpolant
        '''
        
        # Construct
        super().__init__()
        self.switch = switch

    @abc.abstractmethod
    def __call__(t:torch.tensor, x_0, x_1):
        '''
        Function call to interpolant
        '''
        pass

    @abc.abstractmethod
    def compute_dt(t:torch.tensor, x_0, x_1):
        '''
        Call for time derivative of interpolant
        '''
        pass

'''
This file contains some predefined interpolants for the SI framework. Other
interpolants can be defined by the user.
'''
class LinearInterpolant(Interpolant):
    
    def __init__(self):
        '''
        Construct linear interpolant
        '''
        super().__init__()

    def __call__(self, t:torch.tensor, x_0, x_1):
        '''
        Linear interpolant
        @param t : time in [0,1]
        @param x_0 : point from p_0
        @param x_1 : point from p_1
        @return interpolated value
        '''

        # Compute interpolated value
        return (1 - t) * x_0 + t * x_1

    def compute_dt(self, t:torch.tensor, x_0, x_1):
        '''
        Linear interpolant dI/dt
        '''
        
        # Compute dI/dt
        return x_1 - x_0

class TrigonometricInterpolant(Interpolant):

    def __init__(self):
        '''
        Construct trigonometric interpolant
        '''
        super().__init__()

    def __call__(self, t:torch.tensor, x_0, x_1):
        '''
        Trigonometric interpolant
        @param t : time in [0,1]
        @param x_0 : point from p_0
        @param x_1 : point from p_1
        @return interpolated value
        '''

        # Compute interpolated value
        return torch.cos(torch.pi * t / 2) * x_0 + torch.sin(torch.pi * t / 2) * x_1

    def compute_dt(self, t:torch.tensor, x_0, x_1):
        '''
        Trigonometric interpolant dI/dt
        '''

        # Compute dI/dt
        return - (torch.pi / 2) * torch.sin(torch.pi * t / 2) * x_0 + (torch.pi / 2) * torch.cos(torch.pi * t / 2) * x_1

class EncoderDecoderInterpolant(Interpolant):

    def __init__(self, switch):
        '''
        Construct encoder-decoder interpolant with switch 
        '''
        super().__init__(switch)
        self.switch = switch

    def __call__(self, t:torch.tensor, x_0, x_1):
        '''
        Encoder-decoder interpolant
        @param t : time in [0,1]
        @param x_0 : point from p_0
        @param x_1 : point from p_1
        @return interpolated value
        '''

        # Compute interpolated value
        if self.switch.lower() == "two-sided":
            if t < 1/2:
                return (torch.cos(torch.pi * t) ** 2) * x_0
            else:
                return (torch.cos(torch.pi * t) ** 2) * x_1

        elif self.switch.lower() == "one_sided":
            return torch.sqrt(1 - (t ** 2)) * x_0 + t * x_1

        else:
            raise ValueError("Interpolation type not supported!")

    def compute_dt(self, t:torch.tensor, x_0, x_1):
        '''
        Encoder-decoder interpolant dI/dt
        '''

        # Compute interpolated value
        if self.switch.lower() == "two-sided":
            if t < 1/2:
                return - 2 * torch.cos(torch.pi * t) * torch.pi * torch.sin(torch.pi * t) * x_0
            else:
                return - 2 * torch.cos(torch.pi * t) * torch.pi * torch.sin(torch.pi * t) * x_0

        elif self.switch.lower() == "one_sided":
            return - 0.5 * ((1 - (t ** 2)) ** -0.5) * 2 * t * x_0 + x_1

        else:
            raise ValueError("Interpolation type not supported!") 

class MirrorInterpolant(Interpolant):
     
    def __init__(self):
        '''
        Construct mirror interpolant
        '''
        super().__init__()

    def __call__(self, t:torch.tensor, x_0, x_1):
        '''
        Mirror interpolant
        @param t : time in [0,1]
        @param x_0 : point from p_0
        @param x_1 : point from p_1
        @return interpolated value
        '''

        # Compute interpolated value
        return x_1

    def compute_dt(self, t:torch.tensor, x_0, x_1):
        '''
        Mirror interpolant dI/dt
        '''
        
        # Compute dI/dt
        return 0

class VPInterpolant(Interpolant):

    def __init__(self):
        '''
        Construct VP interpolant
        '''
        super().__init__()

    def __call__(self, t:torch.tensor, x_0, x_1):
        '''
        VP interpolant
        @param t : time in [0,1]
        @param x_0 : point from p_0
        @param x_1 : point from p_1
        @return interpolated value
        '''

        # Compute interpolated value
        return torch.sqrt(1 - (t ** 2)) * x_0 + t * x_1

    def compute_dt(self, t:torch.tensor, x_0, x_1):
        '''
        VP interpolant dI/dt
        '''
        
        # Compute dI/dt
        return - ((1 - (t ** 2)) ** -0.5) * t * x_0 + x_1