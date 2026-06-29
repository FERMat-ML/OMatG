from pathlib import Path
from typing import Optional, Union
from lightning.pytorch import Trainer
import matplotlib.pyplot as plt
import torch
from omg.datamodule import OMGDataModule
from omg.globals import BIG_TIME, SMALL_TIME
from omg.omg_irl import OMGIRLScale


class OMGIRLTrainer(Trainer):
    """
    Trainer for the Open Materials Generation with Inference-time Reinforcement Learning (OMatG-IRL) framework.

    This trainer extends the PyTorch Lightning Trainer and provides additional functionality specific to OMG-IRL, such
    as plotting the learned noise schedules over time.

    :param args:
        Positional arguments to pass to the PyTorch Lightning Trainer constructor.
    :param kwargs:
        Keyword arguments to pass to the PyTorch Lightning Trainer constructor.
    """

    def __init__(self, *args, **kwargs):
        """Constructor of the OMGIRLTrainer class."""
        super().__init__(*args, **kwargs)

    def plot_schedule(self, model: OMGIRLScale, datamodule: OMGDataModule,
                      ckpt_path: Optional[Union[str, Path]] = None, plot_filename: str = "schedule.pdf") -> None:
        """
        Plot the learned velocity-annealing schedule of the velocity-annealing OMatG-IRL model.

        :param model:
           OMGIRLScale model (argument required and automatically passed by Lightning CLI).
        :type model: OMGIRLScale
        :param datamodule:
            OMGDataModule (argument required and automatically passed by Lightning CLI).
        :type datamodule: OMGDataModule
        :param ckpt_path:
            Path to the checkpoint file to load. If None, no checkpoint is loaded.
            This argument can be optionally set on the command line.
            Defaults to None.
        :type ckpt_path: Optional[Union[str, Path]]
        :param plot_filename:
            Filename to save the plot of the learned schedule.
            Defaults to "schedule.pdf".
        :type plot_filename: str

        :raises ValueError:
            If the model is not an instance of OMGIRLScale.
        """
        if not isinstance(model, OMGIRLScale):
            raise ValueError("Schedule plotting is only supported for OMGIRLScale models.")

        if ckpt_path is not None:
            print(f"Restoring states from the checkpoint path at {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["state_dict"])
            print(f"Loaded model weights from the checkpoint at {ckpt_path}")

        # noinspection PyTypeChecker
        times = torch.linspace(SMALL_TIME, BIG_TIME, model.integration_time_steps)
        schedules = model.scale_model(times)

        plt.figure()
        plt.plot(times.numpy(), schedules["pos_s"].squeeze(-1).numpy(force=True), label="atomic positions")
        plt.plot(times.numpy(), schedules["cell_s"].squeeze(-1).numpy(force=True), label="lattice vectors")
        plt.xlabel("time")
        plt.ylabel("scale factor")
        plt.legend()
        plt.savefig(plot_filename)
        plt.close()
