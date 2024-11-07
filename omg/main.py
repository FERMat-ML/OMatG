from lightning.pytorch.cli import LightningCLI
from omg.omg import OMG
from omg.datamodule.dataloader import OMGDataModule

def main():
    cli = LightningCLI(OMG, OMGDataModule, save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    main()

