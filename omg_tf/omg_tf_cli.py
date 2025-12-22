from lightning.pytorch.cli import LightningCLI
from omg_tf.base_modules import base_modules


class OMGTFCLI(LightningCLI):
    def __init__(self, *args, **kwargs) -> None:
        """Constructor of the OMGTFCLI class."""
        super().__init__(*args, **kwargs)

    def after_instantiate_classes(self) -> None:
        """Hook to modify instantiated classes after instantiation."""
        super().after_instantiate_classes()
        if base_modules["datamodule"] is None:
            raise ValueError("Base datamodule must be set globally before instantiating OMGTFCLI.")
        self.datamodule = base_modules["datamodule"]
