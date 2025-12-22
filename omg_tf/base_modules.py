from typing import Optional, TypedDict
from lightning import LightningDataModule
from omg.omg_lightning import OMGLightning


class BaseModules(TypedDict):
    """TypedDict to hold the base model and datamodule."""
    model: Optional[OMGLightning]
    datamodule: Optional[LightningDataModule]


# Store the OMG base model and datamodule globally to exclude them from lightning's parameter tracking.
# Store in a mutable dictionary to allow for modifications from other modules with a shared state.
base_modules: BaseModules = {
    "model": None,
    "datamodule": None,
}
