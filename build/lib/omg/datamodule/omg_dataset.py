from torch_geometric.data import Dataset
from omg.datamodule import OMGData, StructureDataset


class OMGDataset(Dataset):
    """
    Dataset wrapper around a StructureDataset that is compatible with Pytorch Geometric.

    This class converts each Structure in the StructureDataset to an OMGData object when accessed via the get method.

    Note that this class can also be passed to torch_geometric.data.lightning.LightningDataset which in turn can be used
    as a LightningDataModule for training with PyTorch Lightning.

    :param dataset:
        StructureDataset object containing the structures to be converted to OMGData objects.
    :type dataset: StructureDataset
    """

    def __init__(self, dataset: StructureDataset):
        """Constructor for the OMGDataset class."""
        super().__init__()
        self._dataset = dataset

    def len(self) -> int:
        """
        Get the number of structures within the dataset.

        :return:
            Number of structures in the dataset.
        :rtype: int
        """
        return len(self._dataset)

    def get(self, idx: int) -> OMGData:
        """
        Return the structure at the given index.

        :param idx:
            Index of the structure to return.
        :type idx: int

        :return:
            The structure at the given index.
        :rtype: Structure
        """
        return OMGData(self._dataset[idx])


if __name__ == '__main__':
    from torch_geometric.data.lightning import LightningDataset

    dataset = StructureDataset("data/mp_20/test.lmdb", lazy_storage=False, property_keys=["band_gap"])
    print(dataset[0].atomic_numbers)
    print(dataset[0].pos)
    print(dataset[0].cell)
    print(dataset[0].property_dict)
    print(dataset[0].pos_is_fractional)
    print(dataset[1].atomic_numbers)
    print(dataset[1].pos)
    print(dataset[1].cell)
    print(dataset[1].property_dict)
    print(dataset[1].pos_is_fractional)

    # Specifying the sampler switches of the random shuffling in the dataloader.
    lightning_data_module = LightningDataset(
        OMGDataset(StructureDataset("data/mp_20/test.lmdb", lazy_storage=False, property_keys=["band_gap"])),
        batch_size=2, pin_memory=False, num_workers=2, sampler=range(len(dataset)))

    for batch in lightning_data_module.train_dataloader():
        print("n_atoms: ", batch.n_atoms, batch.n_atoms.shape)
        print("batch: ", batch.batch, batch.batch.shape)
        print("species: ", batch.species, batch.species.shape)
        print("cell: ", batch.cell, batch.cell.shape)
        print("pos: ", batch.pos, batch.pos.shape)
        print("ptr: ", batch.ptr, batch.ptr.shape)
        print("pos_is_fractional: ", batch.pos_is_fractional, batch.pos_is_fractional.shape)
        print("property: ", batch.property)
        break
