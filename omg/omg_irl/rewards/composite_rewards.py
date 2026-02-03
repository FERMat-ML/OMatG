from typing import Sequence
import numpy as np
from omg.datamodule import OMGDataset, Structure
from .abstracts import ComputeStage, Reward


class CompositeRewards(Reward):
    """
    Composite reward that combines multiple rewards with different weights.

    Rewards with zero weight are skipped during training but are still computed during validation and prediction.

    :param rewards:
        List of reward functions to combine.
    :type rewards: Sequence[Reward]
    :param weights:
        Weights for each reward function (must sum to 1).
    :type weights: Sequence[float]

    :raises ValueError:
        If the number of reward functions does not match the number of weights.
        If any weight is negative.
        If the weights do not sum to 1.
    """

    def __init__(self, rewards: Sequence[Reward], weights: Sequence[float]) -> None:
        """Constructor for CompositeReward."""
        super().__init__()
        if len(rewards) != len(weights):
            raise ValueError("Number of reward functions must match number of weights.")
        if not all(w >= 0.0 for w in weights):
            raise ValueError("Weights must be positive.")
        if not abs(sum(weights) - 1.0) < 1e-6:
            raise ValueError("Weights must sum to 1.0")
        self._reward_functions = rewards
        self._weights = weights

    def set_train_dataset(self, train_dataset: OMGDataset) -> None:
        """
        Set the training dataset for all reward functions.

        :param train_dataset:
            Training dataset.
        :type train_dataset: OMGDataset
        """
        for reward_function in self._reward_functions:
            reward_function.set_train_dataset(train_dataset)

    def set_val_dataset(self, val_dataset: OMGDataset) -> None:
        """
        Set the validation dataset for all reward functions.

        :param val_dataset:
            Validation dataset.
        :type val_dataset: OMGDataset
        """
        for reward_function in self._reward_functions:
            reward_function.set_val_dataset(val_dataset)

    def set_pred_dataset(self, pred_dataset: OMGDataset) -> None:
        """
        Set the prediction dataset for all reward functions.

        :param pred_dataset:
            Prediction dataset.
        :type pred_dataset: OMGDataset
        """
        for reward_function in self._reward_functions:
            reward_function.set_pred_dataset(pred_dataset)

    def compute(self, structures: Sequence[Structure], stage: ComputeStage,
                enable_progress_bar: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Compute rewards for a batch of structures.

        If a reward has zero weight, it is skipped during training but still computed during validation and prediction.

        :param structures:
            Sequence of Structure objects representing generated structures.
        :type structures: Sequence[Structure]
        :param stage:
            Stage of the reward computation.
        :type stage: ComputeStage
        :param enable_progress_bar:
            Whether to enable the progress bar for this computation.
        :type enable_progress_bar: bool

        :return:
            (List of rewards per structure, info dictionary).
        :rtype: tuple[np.ndarray, dict[str, np.ndarray]]
        """
        total_rewards = np.zeros(len(structures))
        total_dict = {}
        for reward_function, weight in zip(self._reward_functions, self._weights):
            if stage != ComputeStage.TRAIN or weight > 0.0:
                rewards, info_dict = reward_function.compute(structures, stage, enable_progress_bar)
                total_rewards += weight * rewards
                for key, value in info_dict.items():
                    assert key not in total_dict
                    total_dict[key] = value
        return total_rewards, total_dict
