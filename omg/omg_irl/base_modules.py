from typing import Optional, TypedDict
from omg.datamodule import OMGDataModule
from omg.omg_lightning import OMGLightning


class BaseModules(TypedDict):
    """TypedDict to hold the base model and datamodule."""
    model: Optional[OMGLightning]
    datamodule: Optional[OMGDataModule]


# Store the OMatG base model and datamodule globally to exclude them from lightning's parameter tracking.
# Store in a mutable dictionary to allow for modifications from other modules with a shared state.
base_modules: BaseModules = {
    "model": None,
    "datamodule": None,
}
