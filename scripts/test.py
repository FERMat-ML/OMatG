from omg.datamodule import DataModule
from omg.datamodule.dataloader import OMGTorchDataset
from torch_geometric.loader import DataLoader
import torch
from omg.model.encoders.cspnet_full import CSPNet_Full


dataset = DataModule.from_lmdb(["./../data/mp_20/train.lmdb"])

example_data = OMGTorchDataset(dataset)

loader = DataLoader(example_data, batch_size=10)

for batch in loader:
    t = torch.rand(1)
    print(t,batch)
    break  

t = t.repeat(10,1)


model = CSPNet_Full(latent_dim=1)
model = model.double()

md = model(batch,t = t)
