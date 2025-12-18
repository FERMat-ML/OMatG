from typing import Optional
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning import LightningDataModule


base_datamodule: Optional[LightningDataModule] = None


class OMGTFCLI(LightningCLI):
    def __init__(self, *args, **kwargs) -> None:
        """Constructor of the OMGTFCLI class."""
        super().__init__(*args, **kwargs)

    def after_instantiate_classes(self) -> None:
        """Hook to modify instantiated classes after instantiation."""
        super().after_instantiate_classes()
        if base_datamodule is None:
            raise ValueError("Base datamodule must be set globally before instantiating OMGTFCLI.")
        self.datamodule = base_datamodule
