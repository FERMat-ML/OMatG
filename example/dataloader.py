import torch_geometric.loader

from omg.datamodule import DataModule
from omg.datamodule.dataloader import OMGData, OMGTorchDataset

ds = DataModule.from_ase("./example.xyz")

torch_ds = OMGTorchDataset(ds)

print(torch_ds[0])

dl = torch_geometric.loader.DataLoader(torch_ds, batch_size=3)

batch = next(iter(dl))

print(batch.species, batch.cell, batch.pos, batch.batch)
