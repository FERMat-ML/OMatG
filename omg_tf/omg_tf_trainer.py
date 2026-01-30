from pathlib import Path
from typing import Optional, Union
from lightning.pytorch import Trainer
import matplotlib.pyplot as plt
import torch
from omg.datamodule import OMGDataModule
from omg.globals import BIG_TIME, SMALL_TIME
from omg_tf.omg_tf_lightning_ppo import OMGTFLightningPPO


class OMGTFTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        """Constructor of the OMGTFTrainer class."""
        super().__init__(*args, **kwargs)

    def plot_schedule(self, model: OMGTFLightningPPO, datamodule: OMGDataModule,
                      ckpt_path: Optional[Union[str, Path]] = None) -> None:
        if model.residual_mode != OMGTFLightningPPO.ResidualMode.SCALE:
            raise ValueError("Schedule plotting is only supported for SCALE residual mode.")

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["state_dict"])

        times = torch.linspace(SMALL_TIME, BIG_TIME, model.integration_time_steps)
        schedules = model.residual_model(None, times.unsqueeze(-1))

        plt.figure()
        plt.plot(times.numpy(), schedules["pos_s"].squeeze(-1).numpy(force=True), label="atomic positions")
        plt.plot(times.numpy(), schedules["cell_s"].squeeze(-1).numpy(force=True), label="lattice vectors")
        plt.legend()
        plt.show()
        plt.close()