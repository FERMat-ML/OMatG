from omg.omg import OMG
from omg.omg_cli import OMGCLI
from omg.datamodule.dataloader import OMGDataModule


def main():
    OMGCLI(model_class=OMG, datamodule_class=OMGDataModule, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
