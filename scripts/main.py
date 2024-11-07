#from lightning.pytorch.cli import LightningCLI
from omg.omg_cli import OMG_CLI
from omg.omg import OMG
from omg.datamodule.dataloader import OMGDataModule

def main():
    cli = OMG_CLI(OMG, OMGDataModule, save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    main()

