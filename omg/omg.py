from omg.si.stochastic_interpolants import StochasticInterpolants
import lightning as L
import torch
import torch.nn as nn
from torch import optim
from typing import Optional, Sequence
from omg.sampler.sampler import Sampler
from omg.utils import xyz_saver 

class OMG(L.LightningModule):
    """
    Main module which is fit and and used to generate structures using Lightning CLI.
    """
    
    # TODO: specify argument types
    def __init__(self, si: StochasticInterpolants, sampler: Sampler, model: nn.Module,
                 relative_si_costs: Sequence[float], load_checkpoint: Optional[str] = None) -> None:
        super().__init__()
        self.si = si 
        self.sampler = sampler
        model = model.double()
        self.model = model
        print (self.state_dict)
        if not len(relative_si_costs) == len(self.si):
            raise ValueError("The number of stochastic interpolants and costs must be equal.")
        if not all(cost >= 0.0 for cost in relative_si_costs):
            raise ValueError("All cost factors must be non-negative.")
        if not abs(sum(relative_si_costs) - 1.0) < 1e-10:
            raise ValueError("The sum of all cost factors must be approximately equal to 1.")
        self._relative_si_costs = relative_si_costs
        if load_checkpoint:
            checkpoint = torch.load(load_checkpoint, map_location=self.device)
            self.load_state_dict(checkpoint['state_dict'])

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

    # TODO: specify argument types
    def training_step(self, x_1) -> torch.Tensor:
        """
        Performs one training step given a batch of x_1

        :return:
            Loss from training step
        :rtype: torch.Tensor
        """
        x_0 = self.sampler.sample_p_0(x_1).to(self.device) 

        # sample t uniformly for each structure
        t = torch.rand(len(x_1.n_atoms)).to(self.device)

        losses = self.si.losses(self.model, t, x_0, x_1)

        total_loss = torch.tensor(0.0, device=self.device)

        for cost, loss_key in zip(self._relative_si_costs, losses):
            losses[loss_key] = cost * losses[loss_key]
            total_loss += losses[loss_key]

        assert "loss_total" not in losses
        losses["loss_total"] = total_loss

        self.log_dict(
            losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return total_loss

    def validation_step(self, x_1) -> torch.Tensor:
        """
        Performs one validation step given a batch of x_1
        """

        x_0 = self.sampler.sample_p_0(x_1).to(self.device) 

        # sample t uniformly for each structure
        t = torch.rand(len(x_1.n_atoms)).to(self.device)

        losses = self.si.losses(self.model, t, x_0, x_1)

        total_loss = torch.tensor(0.0, device=self.device)

        for cost, loss_key in zip(self._relative_si_costs, losses):
            losses[f"val_{loss_key}"] = cost * losses[loss_key]
            total_loss += losses[loss_key]
            losses.pop(loss_key)

        assert "loss_total" not in losses
        losses["val_loss_total"] = total_loss

        self.log_dict(
            losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return total_loss
    # TODO: what do we want to return
    def predict_step(self, x):
        """
        Performs generation
        """
        x_0 = self.sampler.sample_p_0()
        gen = self.si.integrate(x_0, self.model)
        # probably want to turn structure back into some other object that's easier to work with
        xyz_saver(gen)
        return gen

    #TODO allow for YAML config
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



