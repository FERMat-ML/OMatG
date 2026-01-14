from abc import ABC, abstractmethod
from enum import auto, Enum
from typing import Sequence
import numpy as np
import torch
from tqdm import trange
from omg.datamodule import OMGData, OMGDataset, Structure
from omg.globals import BIG_TIME, SMALL_TIME
from omg.model.model import Model
from omg.si import (SingleStochasticInterpolant, SingleStochasticInterpolantIdentity, SingleStochasticInterpolantOS,
                    DifferentialEquationType)
from omg.si.abstracts import TimeChecker
from omg.utils import DataField
from omg_tf.base_modules import base_modules


class Combiner(ABC):
    """
    Abstract base class for combining an ODE-based model with a residual model that learns and applies residual
    velocities in a reinforcement learning setting.

    TODO:
    - Add noise scale scheduling.
    - Add species integration support.
    - Add velocity annealing like scaling.
    - Allow to apply only to subset of fields.
    """

    def __init__(self, noise_scales: dict[str, float], integrate_noisy: bool = False) -> None:
        """
        Constructor of the Combiner class.

        :param noise_scales:
            Dictionary mapping data field names to noise scales for stochastic policy.
        :type noise_scales: dict[str, float]
        :param integrate_noisy:
            Whether to integrate noisy residuals.
        :type integrate_noisy: bool
        """
        super().__init__()

        base_model = base_modules["model"]
        if base_model is None:
            raise ValueError("Base model must be set globally before initializing OMGTFLightning.")
        # Freeze base model parameters.
        base_model.freeze()

        self._integrated_data_fields = []
        pos_interpolant = base_model.si.get_stochastic_interpolant(DataField.pos.name)
        self._integrate_pos = not isinstance(pos_interpolant, SingleStochasticInterpolantIdentity)
        if self._integrate_pos:
            if isinstance(pos_interpolant, SingleStochasticInterpolant):
                if not pos_interpolant.differential_equation_type == DifferentialEquationType.ODE:
                    raise ValueError("Residual integration for positions is only supported for ODE-based stochastic "
                                     "interpolants.")
                if pos_interpolant.velocity_annealing_factor != 0.0:
                    raise ValueError("Residual integration for positions is not supported with velocity annealing.")
            elif isinstance(pos_interpolant, SingleStochasticInterpolantOS):
                if not pos_interpolant.differential_equation_type == DifferentialEquationType.ODE:
                    raise ValueError("Residual integration for positions is only supported for ODE-based stochastic "
                                     "interpolants.")
                if not pos_interpolant.predict_velocity:
                    raise ValueError("Residual integration for positions is only supported when predicting velocity.")
                if pos_interpolant.velocity_annealing_factor != 0.0:
                    raise ValueError("Residual integration for positions is not supported with velocity annealing.")
            else:
                raise ValueError("Unsupported stochastic interpolant for position residual integration.")
            self._integrated_data_fields.append(DataField.pos)
            if DataField.pos.name not in noise_scales:
                raise ValueError("Noise scale for position residuals must be provided when integrating positions.")
        else:
            if DataField.pos.name in noise_scales:
                raise ValueError("Noise scale for position residuals provided but positions are not integrated.")

        cell_interpolant = base_model.si.get_stochastic_interpolant(DataField.cell.name)
        self._integrate_cell = not isinstance(cell_interpolant, SingleStochasticInterpolantIdentity)
        if self._integrate_cell:
            if isinstance(cell_interpolant, SingleStochasticInterpolant):
                if not cell_interpolant.differential_equation_type == DifferentialEquationType.ODE:
                    raise ValueError("Residual integration for cell is only supported for ODE-based stochastic "
                                     "interpolants.")
                if cell_interpolant.velocity_annealing_factor != 0.0:
                    raise ValueError("Residual integration for cell is not supported with velocity annealing.")
            elif isinstance(cell_interpolant, SingleStochasticInterpolantOS):
                if not cell_interpolant.differential_equation_type == DifferentialEquationType.ODE:
                    raise ValueError("Residual integration for cell is only supported for ODE-based stochastic "
                                     "interpolants.")
                if not cell_interpolant.predict_velocity:
                    raise ValueError("Residual integration for cell is only supported when predicting velocity.")
                if cell_interpolant.velocity_annealing_factor != 0.0:
                    raise ValueError("Residual integration for cell is not supported with velocity annealing.")
            else:
                raise ValueError("Unsupported stochastic interpolant for cell residual integration.")
            self._integrated_data_fields.append(DataField.cell)
            if DataField.cell.name not in noise_scales:
                raise ValueError("Noise scale for cell residuals must be provided when integrating cell.")
        else:
            if DataField.cell.name in noise_scales:
                raise ValueError("Noise scale for cell residuals provided but cell is not integrated.")

        species_interpolant = base_model.si.get_stochastic_interpolant(DataField.species.name)
        self._integrate_species = not isinstance(species_interpolant, SingleStochasticInterpolantIdentity)
        if self._integrate_species:
            raise ValueError("Residual integration for species is not supported yet.")
        else:
            if DataField.species.name in noise_scales:
                raise ValueError("Noise scale for species residuals provided but species are not integrated.")

        try:
            self._noise_scales = {DataField[key]: value for key, value in noise_scales.items()}
        except KeyError as e:
            raise ValueError(f"Invalid data field key in noise scales: {e}")

        if not len(self._integrated_data_fields) > 0:
            raise ValueError("At least one of position, cell, or species must be integrated.")

        self._integration_time_steps = base_model.si.integration_time_steps
        self._integrate_noisy = integrate_noisy

    def integrated_data_fields(self) -> list[DataField]:
        """
        Get the data fields that are integrated with residuals.

        :return:
            List of data fields integrated with residuals.
        :rtype: list[DataField]
        """
        return self._integrated_data_fields

    @torch.no_grad()
    def integrate(self, residual_model: Model, x_0: OMGData) -> OMGData:
        """
        Integrate the structures x_0 from time 0 to 1 with an Euler integration scheme relying on the added velocities
        of the base and residual models.

        :param residual_model:
            The residual model predicting residual velocities.
        :type residual_model: Model
        :param x_0:
            Initial structures at time 0.
        :type x_0: OMGData

        :return:
            Structures at final time 1 after integration with residuals.
        :rtype: OMGData
        """
        if self._integrate_noisy:
            x_t, _, _ = self.training_integrate(residual_model, x_0)
            return x_t

        base_model = base_modules["model"].model
        assert base_model is not None
        batch_size = len(x_0.n_atoms)
        times = torch.linspace(SMALL_TIME, BIG_TIME, self._integration_time_steps, device=x_0.pos.device)
        x_t = x_0.clone()
        for t_index in trange(1, len(times), desc="Integrating with residuals", position=1, leave=False):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            time = t.repeat(batch_size)
            base_model_output = base_model(x_t, time)
            residual_output = residual_model(x_t, time)
            if self._integrate_pos:
                pos_b = base_model_output[DataField.pos.name + "_b"] + residual_output[DataField.pos.name + "_b"]
                x_t.pos = x_t.pos + pos_b * dt
            if self._integrate_cell:
                cell_b = base_model_output[DataField.cell.name + "_b"] + residual_output[DataField.cell.name + "_b"]
                x_t.cell = x_t.cell + cell_b * dt
        return x_t

    @abstractmethod
    def training_integrate(
            self,
            residual_model: Model,
            x_0: OMGData
    ) -> tuple[OMGData, dict[DataField, torch.Tensor], dict[DataField, torch.Tensor]]:
        """
        Integrate the structures x_0 from time 0 to 1 with an Euler integration scheme relying on the added velocities
        of the base and residual models.

        This method should typically perform an Euler integration similar to integrate_with_residual_means. However,
        it should randomize the residual velocities from the residual model (e.g., by adding noise) and return the log
        probabilities of the applied residuals for each integrated data field. These can then be used for policy
        gradient updates.

        In addition, this method should also return the mean squared residuals per integrated data field for
        regularization purposes.

        The implementation of this method in subclasses decides how the residuals are randomized.

        :param residual_model:
            The residual model predicting residual velocities.
        :type residual_model: Model
        :param x_0:
            Initial structures at time 0.
        :type x_0: OMGData

        :return:
            (Structures at final time 1 after integration with residuals,
             Batch-wise log probabilities of the applied residuals for each integrated data field,
             Batch-wise mean squared residuals for each integrated data field).
        :rtype: tuple[OMGData, dict[DataField, torch.tensor], dict[DataField, torch.tensor]]
        """
        raise NotImplementedError


class Reward(ABC):
    """
    Abstract base class for reward functions.

    Reward functions must be initialized with the training and validation datasets before compute is called.
    This allows reward functions to pre-compute any necessary data structures for efficient reward computation.

    TODO: SHOULD I MAKE SURE THAT REWARDS ARE NOT SCALED WITH NUMBER OF ATOMS?
    """
    class ComputeStage(Enum):
        """
        Enum for the possible stages of reward computation.
        """

        TRAIN = auto()
        """
        Train stage.
        """
        VAL = auto()
        """
        Validation stage.
        """
        PRED = auto()
        """
        Prediction stage.
        """

    def set_train_dataset(self, train_dataset: OMGDataset) -> None:
        """
        Set the training dataset for reward computation.

        This method is called in the OMGTFLightning class before compute is called. Subclasses can override this method
        to pre-compute any necessary data structures for efficient reward computation.

        :param train_dataset:
            Training dataset.
        :type train_dataset: OMGDataset
        """
        pass

    def set_val_dataset(self, val_dataset: OMGDataset) -> None:
        """
        Set the validation dataset for reward computation.

        This method is called in the OMGTFLightning class before compute is called. Subclasses can override this method
        to pre-compute any necessary data structures for efficient reward computation.

        :param val_dataset:
            Validation dataset.
        :type val_dataset: OMGDataset
        """
        pass

    def set_pred_dataset(self, pred_dataset: OMGDataset) -> None:
        """
        Set the prediction dataset for reward computation.

        This method is called in the OMGTFLightning class before compute is called. Subclasses can override this method
        to pre-compute any necessary data structures for efficient reward computation.

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
        for training or validation, allowing the reward function to use the appropriate precomputed data from the
        set_datasets method.

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


class NoiseSchedule(ABC, TimeChecker):
    """
    Abstract base class for defining a noise schedule sigma(t).
    """

    def __init__(self) -> None:
        """Constructor of the NoiseSchedule class."""
        super().__init__()

    @abstractmethod
    def noise(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get the noise level at times t.

        :param t:
            Times in [0, 1].
        :type t: torch.Tensor

        :return:
            Noise levels sigma(t).
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def learnable(self) -> bool:
        """
        Return whether the noise schedule has learnable parameters.

        :return:
            True if the noise schedule has learnable parameters, False otherwise.
        :rtype: bool
        """
        raise NotImplementedError
