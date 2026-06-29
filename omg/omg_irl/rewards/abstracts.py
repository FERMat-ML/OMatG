from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Sequence
import numpy as np
from omg.datamodule import OMGDataset, Structure


class ComputeStage(Enum):
    """
    Enum for the possible stages of reward computation.
    """

    TRAIN = auto()
    """
    Training stage.
    """
    VAL = auto()
    """
    Validation stage.
    """
    PRED = auto()
    """
    Prediction stage.
    """


class Reward(ABC):
    """
    Abstract base class for reward functions.

    Reward functions can be initialized with the training, validation, and prediction datasets before compute is called.
    This allows reward functions to pre-compute any necessary data structures for efficient reward computation.
    """

    def set_train_dataset(self, train_dataset: OMGDataset) -> None:
        """
        Set the training dataset for reward computation.

        This method should be called before compute is called. Subclasses can override this method to pre-compute any
        necessary data structures for efficient reward computation.

        :param train_dataset:
            Training dataset.
        :type train_dataset: OMGDataset
        """
        pass

    def set_val_dataset(self, val_dataset: OMGDataset) -> None:
        """
        Set the validation dataset for reward computation.

        This method should be called before compute is called. Subclasses can override this method to pre-compute any
        necessary data structures for efficient reward computation.

        :param val_dataset:
            Validation dataset.
        :type val_dataset: OMGDataset
        """
        pass

    def set_pred_dataset(self, pred_dataset: OMGDataset) -> None:
        """
        Set the prediction dataset for reward computation.

        This method should be called before compute is called. Subclasses can override this method to pre-compute any
        necessary data structures for efficient reward computation.

        :param pred_dataset:
            Prediction dataset.
        :type pred_dataset: OMGDataset
        """
        pass

    @abstractmethod
    def compute(self, structures: Sequence[Structure], stage: ComputeStage,
                enable_progress_bar: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Compute rewards for a batch of structures.

        Some reward functions may require access to a reference dataset for computing rewards (e.g., to compute
        similarity to known stable structures). The stage parameter indicates whether the reward is being computed
        for training, validation, or prediction, allowing the reward function to use the appropriate precomputed data
        from the set_datasets method.

        Since rewards should always be maximized, the returned rewards might be transformed versions of the actual
        property of interest. For logging purposes, additional information (e.g., the untransformed property values) can
        be returned in the info dictionary. This dictionary maps string keys to numpy arrays of shape (batch_size,).

        :param structures:
            Sequence of Structure objects representing generated structures.
        :type structures: Sequence[Structure]
        :param stage:
            Stage of the reward computation.
        :type stage: ComputeStage
        :param enable_progress_bar:
            Whether to enable a progress bar during reward computation.
        :type enable_progress_bar: bool

        :return:
            (List of rewards per structure, info dictionary).
        :rtype: tuple[np.ndarray, dict[str, np.ndarray]]
        """
        raise NotImplementedError
