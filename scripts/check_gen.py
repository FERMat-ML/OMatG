from ase.io import read
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
import time

def main(f):
    atoms = read(f, index=':')

    vol = []
    nums = {}
    n_types = {}
    num_atoms = 0
    for i in range(1,7):
        n_types[i] = 0
    for i in range(1,110):
        nums[i] = 0
    for a in atoms:
        v = a.get_volume()
        num_atoms += len(a)
        vol.append(a.get_volume())
        num = a.numbers
        n_type = len(list(set(num)))
        if n_type > 6:
            n_type = 6
        n_types[n_type] += 1
        for n in num:
            nums[n] +=1



    h = []
    n = []
    for k,v in n_types.items():
        n.append(v)
    for k,v in nums.items():
        h.append(v)
    plt.bar(list(range(1,110)), np.array(h)/num_atoms)
    plt.title('Fractional element composition')
    plt.savefig(f'elements_{time.strftime("%Y%m%d-%H%M%S")}.png')
    plt.close()
    plt.hist(vol, bins = 50)
    plt.title('Volume')
    plt.savefig(f'volume_{time.strftime("%Y%m%d-%H%M%S")}.png')
    plt.close()
    plt.bar(list(range(1,7)),n)
    plt.title('Nary')
    plt.savefig(f'nary_{time.strftime("%Y%m%d-%H%M%S")}.png')

if __name__ == "__main__":
    import sys
    f = sys.argv[1]
    main(f)
