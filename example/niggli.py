from omg.datamodule.utils import niggli_reduce_configuration, niggli_reduce_data
from omg.datamodule.datamodule import DataModule
import torch

ds = DataModule(["./../data/mp_20/train.lmdb"])

transform_matrix = [[1, 0, 0], [0, 1, 0], [1., 0, 1]]
transform_matrix = torch.tensor(transform_matrix, dtype=ds[0].cell.dtype) * 2.0

config = ds[0]
# config._cell = torch.mm(config.cell, transform_matrix)

# print(torch.mm(ds[0].cell, transform_matrix))

print(config.cell)
niggli_reduce_configuration(config)
print(config.cell)
config = ds[1]
print(config.cell, config.coords)
cell,pos = niggli_reduce_data(config.species, config.coords, config.cell)
print(cell, pos)

