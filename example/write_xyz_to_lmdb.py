import lmdb
from ase.io import read
import random
from tqdm import tqdm
import numpy as np
import pickle as pkl
import torch

random.seed(123)

env = lmdb.open("alexandria_val.lmdb", map_size=int(1e12), subdir=False)
txn = env.begin(write=True)
atoms = read('alexandria_20.xyz', index=':')
random.shuffle(atoms)
split = int(len(atoms)*.8)
for i,a in tqdm(enumerate(atoms[split:])):
    pos = a.get_positions()
    cell = np.array(a.get_cell()[:,:])
    atomic_numbers = np.array(a.get_atomic_numbers(), dtype=np.int32)
    property_dict = {}
    pbc = torch.tensor([1, 1, 1], dtype=torch.int32)

    txn.put(f"{i}".encode(), pkl.dumps({"pos": torch.from_numpy(pos),
                                            "cell": torch.from_numpy(cell),
                                            "atomic_numbers": torch.from_numpy(atomic_numbers),
                                            "pbc": pbc} | property_dict))
txn.commit()
env.close()
