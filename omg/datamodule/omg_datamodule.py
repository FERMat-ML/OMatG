import copy
from typing import Any, Optional
from torch_geometric.loader import DataLoader
from torch_geometric.data.lightning import LightningDataset
from omg.datamodule import OMGDataset, StructureDataset


class OMGDataModule(LightningDataset):
    def __init__(self, train_dataset: StructureDataset, val_dataset: StructureDataset, pred_dataset: StructureDataset,
                 train_batch_size: Optional[int] = None, val_batch_size: Optional[int] = None,
                 pred_batch_size: Optional[int] = None, **kwargs: Any) -> None:
        """
        Lightning DataModule that wraps StructureDataset objects for training, validation, and prediction into
        OMGDataset objects that are compatible with Pytorch Geometric.

        Note that this class extends the LightningDataset class from Pytorch Geometric which in turn extends the
        LightningDataModule class from Pytorch Lightning.

        This class enforces that the datasets used for training, validation, and prediction are all set.

        Also, it is possible to specify specific batch sizes for training, validation, and prediction dataloaders that
        differ from the default batch size specified in the kwargs.

        :param train_dataset:
            StructureDataset for training.
        :type train_dataset: StructureDataset
        :param val_dataset:
            StructureDataset for validation.
        :type val_dataset: StructureDataset
        :param pred_dataset:
            StructureDataset for prediction.
        :type pred_dataset: StructureDataset
        :param train_batch_size:
            Optional batch size for the training dataloader. If None, the default batch size from kwargs is used.
        :type train_batch_size: Optional[int]
        :param val_batch_size:
            Optional batch size for the validation dataloader. If None, the default batch size from kwargs is used.
        :type val_batch_size: Optional[int]
        :param pred_batch_size:
            Optional batch size for the prediction dataloader. If None, the default batch size from kwargs is used.
        :type pred_batch_size: Optional[int]
        :param kwargs:
            Additional keyword arguments to pass to the LightningDataset constructor which are in turn passed to the
            PyTorch Geometric and PyTorch data loaders.
        :type kwargs: Any
        """
        train_omg_dataset = OMGDataset(train_dataset)
        val_omg_dataset = OMGDataset(val_dataset)
        pred_omg_dataset = OMGDataset(pred_dataset)

        # Not necessary but allows type checkers to know the OMGDataset types of the datasets.
        self.train_dataset =  train_omg_dataset
        self.val_dataset = val_omg_dataset
        self.pred_dataset = pred_omg_dataset
        self.test_dataset = None

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.pred_batch_size = pred_batch_size

        _ = DataLoader(train_omg_dataset, **kwargs)  # This allows Lightning CLI to infer the allowed kwargs.
        super().__init__(train_dataset=train_omg_dataset, val_dataset=val_omg_dataset, pred_dataset=pred_omg_dataset,
                         **kwargs)

    def train_dataloader(self) -> DataLoader:
        """
        Create the training dataloader.

        If a specific training batch size was provided, it is used here. Otherwise, the default batch size from kwargs
        is used.

        :return:
            The training dataloader.
        :rtype: DataLoader
        """
        if self.train_batch_size is not None:
            original_kwargs = copy.copy(self.kwargs)
            self.kwargs['batch_size'] = self.train_batch_size
            dataloader = super().train_dataloader()
            self.kwargs = original_kwargs  # Restore original kwargs.
            return dataloader
        return super().train_dataloader()

    def val_dataloader(self) -> DataLoader:
        """
        Create the validation dataloader.

        If a specific validation batch size was provided, it is used here. Otherwise, the default batch size from kwargs
        is used.

        :return:
            The validation dataloader.
        :rtype: DataLoader
        """
        if self.val_batch_size is not None:
            original_kwargs = copy.copy(self.kwargs)
            self.kwargs['batch_size'] = self.val_batch_size
            dataloader = super().val_dataloader()
            self.kwargs = original_kwargs  # Restore original kwargs.
            return dataloader
        return super().val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        """
        Create the prediction dataloader.

        If a specific prediction batch size was provided, it is used here. Otherwise, the default batch size from kwargs
        is used.

        :return:
            The prediction dataloader.
        :rtype: DataLoader
        """
        if self.pred_batch_size is not None:
            original_kwargs = copy.copy(self.kwargs)
            self.kwargs['batch_size'] = self.pred_batch_size
            dataloader = super().predict_dataloader()
            self.kwargs = original_kwargs  # Restore original kwargs.
            return dataloader
        return super().predict_dataloader()
