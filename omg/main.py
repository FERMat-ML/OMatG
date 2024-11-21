from omg.omg_lightning import OMGLightning
from omg.omg_cli import OMGCLI
from omg.datamodule.dataloader import OMGDataModule


def main():
    OMGCLI(model_class=OMGLightning, datamodule_class=OMGDataModule, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
