# Import libraries
import torch

class StochasticInterpolant:
    '''
    Base class for a stochastic interpolant
    '''

    def __init__(self, interpolant, gamma, eps:float):
        '''
        General initialization for a stochastic interpolant
        @param interpolant : Must be a FUNCTION. This provides flexibility in the interpolaters one can use but certain end conditions must hold
        @param gamma : keyword for gamma 
        @param eps : control for level of stochasticity
        '''

        # Initialize interpolant
        self.I = interpolant
        self.gamma = gamma

        # Set stochasticity
        self.eps = eps

    def __call__(self, t:torch.Tensor, x_0:torch.Tensor, x_1:torch.Tensor, z:torch.Tensor):
        '''
        Evaluate stochastic interpolant on a trio of time and two data points
        @param t : time
        @param x_0 : data point from p_0
        @param x_1 : data point from p_1
        '''

        # Assert size equality
        assert x_0.shape == x_1.shape

        # Return evaluation
        return self.I(t, x_0, x_1) + self.eps * self.gamma(t) * torch.rand_like(x_0) * z 

    def compute_time_derivative(self, t:torch.Tensor, x_0:torch.Tensor, x_1:torch.Tensor, z:torch.Tensor):
        '''
        Evaluate stochastic interpolant on a trio of time and two data points
        @param t : time
        @param x_0 : data point from p_0
        @param x_1 : data point from p_1
        '''

        # Assert size equality
        assert x_0.shape == x_1.shape

        # Return evaluation
        return self.I.compute_dt(t, x_0, x_1) + self.gamma.compute_dt(t) * z