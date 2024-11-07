from omg.datamodule.dataloader import OMGTorchDataset
from omg.datamodule.datamodule import DataModule
from ase import Atoms
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
dm = DataModule('data/mp_20/train.lmdb')
ds = OMGTorchDataset(dm,convert_to_fractional=True,niggli=True)

vol = {}
vols = []
dens = []
nums = {}
n_types = {}
num_atoms = 0
#for j in range(1,21):
#    nums[j]={}
for i in range(1,110):
        nums[i] = 0
for i in range(1,7):
    n_types[i] = 0
for it in tqdm(range(len(ds))):
    try:
        i = ds[it]
        a = Atoms(numbers=i.species, scaled_positions=i.pos, cell=i.cell[0,:,:])
        num_atoms += len(a)
        num = a.numbers
        n_type = len(list(set(num)))
        if n_type > 6:
            n_type = 6
        n_types[n_type] += 1
        t = len(num)
        for n in num:
            nums[n] += 1
    #vol[len(a)].append(a.get_volume())
        vols.append(a.get_volume())
    except:
        pass
    #dens.append(len(a)/a.get_volume())
#print (np.mean(vols),np.mean(dens))
#print (np.mean(vol[10]),np.percentile(vol[4],[1,25,50,75,99]))
#fit_alpha, fit_loc, fit_beta=stats.gamma.fit(dens)
#print(fit_alpha, fit_loc, fit_beta)
h = []
for k,v in nums.items():
    h.append(v)
plt.bar(list(range(1,110)),np.array(h)/num_atoms)
plt.title('MP20 Training Element Distribution')
plt.ylabel('Compostion Fraction')
plt.xlabel('Atomic Number')
#plt.ylim(0,.07)
plt.savefig('nums.png')
plt.close()
plt.hist(vols, bins=50)
plt.ylabel('Count')
plt.title('MP20 Training Volume Distribution')
plt.xlabel(r'Volume (Angstrom$^3$)')
plt.savefig('vol_niggli.png')
plt.close()
h = []
for k,v in n_types.items():
    h.append(v)
plt.bar(list(range(1,7)), h)
plt.ylabel('Count')
plt.xlabel('Number of Unique Element Types')
plt.title(r'MP20 Training $N$-ary Distribution')
plt.savefig('ntypes.png')
plt.close()
#with open('vol.pkl', 'wb') as f:
#    pickle.dump(vol,f)
