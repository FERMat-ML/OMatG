from omg.si.stochastic_interpolants import StochasticInterpolants
import lightning as L
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
                 relative_si_costs: Sequence[float], load_checkpoint: Optional[str] = None,
                 learning_rate: Optional[float] = 1.e-3,
                 # TODO: make adjustable
                 lr_scheduler: Optional[bool] = False) -> None:
        super().__init__()
        self.si = si 
        self.sampler = sampler
        model = model.double()
        self.learning_rate = learning_rate
        self.model = model
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
        # TODO: hardcoded normalization for losses
        self.loss_norm = {}
        self.loss_norm['loss_species'] = 0.43 
        self.loss_norm['loss_pos'] = 0.020
        self.loss_norm['loss_cell'] = 0.022
        self.lr_scheduler = lr_scheduler

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

    def on_fit_start(self):
        if self.learning_rate:
            # Overwrite learning rate after running LearningRateFinder
            for optimizer in self.trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["learning_rate"] = self.learning_rate

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
            losses[loss_key] = cost * losses[loss_key] # Don't normalize here so we can inspect the losses
            total_loss += losses[loss_key] / self.loss_norm[loss_key] # normalize weights TODO: Look at how SDE losses are combined

        assert "loss_total" not in losses
        losses["loss_total"] = total_loss

        self.log_dict(
            losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
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
            total_loss += losses[f"val_{loss_key}"] / self.loss_norm[loss_key]
            losses.pop(loss_key)

        assert "loss_total" not in losses
        losses["val_loss_total"] = total_loss

        self.log_dict(
            losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss
    # TODO: what do we want to return
    def predict_step(self, x):
        """
        Performs generation
        """
        x_0 = self.sampler.sample_p_0(x).to(self.device)
        gen, inter = self.si.integrate(x_0, self.model, save_intermediate=True)
        # probably want to turn structure back into some other object that's easier to work with
        xyz_saver(gen.to('cpu'))
        return gen

    #TODO allow for YAML config
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.lr_scheduler:
            lr_scheduler = ReduceLROnPlateau(optimizer,patience=40)
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss_total",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
            }
            return {"optimizer": optimizer,
                    "lr_scheduler": lr_scheduler_config
                    }
        else:
            return optimizer



