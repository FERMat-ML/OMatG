from omg.si.stochastic_interpolants import StochasticInterpolants
import lightning as L
import torch
from typing import Sequence

class OMG(L.LightningModule):
    """
    Main module on which we call Trainer.fit and generate structures via generate function.
    """
    
    # TODO: specify argument types
    def __init__(self, si: StochasticInterpolants, sampler, model):
        self.si = si 
        self.sampler = sampler
        self.model = model

    def forward(self, x_t: Sequence[torch.Tensor], t: torch.Tensor) -> Sequence[Sequence[torch.Tensor]]:
        """
        Calls encoder + head stack

        :param x_t:
            Sequence of torch.tensors corresponding to batched species, fractional coordinates and lattices.
        :type x_t: Sequence[torch.Tensor]

        :param t:
            Sampled times
        :type t: torch.Tensor

        :return:
            Predicted b and etas for species, coordinates and lattices, respectively.
        :rtype: Sequence[Sequence[torch.Tensor]]
        """
        x = self.model(x, t)
        return x

    # TODO: specify argument types, determine what we want to return
    def training_step(self, x_1):
        """
        Performs one training step given a batch of x_1
        """

        x_0 = self.sampler.sample_p_0() # this might need x_1 as input so number of atoms are consistent 
        
        # sample t uniformly for each structure`
        t = torch.rand(len(x_1.n_atoms))

        x_t = self.si.interpolate(t, x_0, x_1)
        
        pred = self.model(x_t, t)
        
        # record loss
        loss = si.loss(pred, t, x_0, x_1) 
       
        return loss

    def validation_step(self, x_1)
        """
        Performs one validation step given a batch of x_1
        """

        x_0 = self.sampler.sample_p_0() # this might need x_1 as input so number of atoms are consistent 
        
        # sample t uniformly for each structure`
        t = torch.rand(len(x_1.n_atoms)) 
        
        x_t = self.si.interpolate(t, x_0, x_1)
        
        pred = self.model(x_t, t)
        
        # record loss
        loss = si.loss(pred, t, x_0, x_1) 

        return loss

    def predict(self)
        x_0 = self.sampler.sample_p_0()
        gen = self.si.integrate(x_0, self.model)
        return gen



