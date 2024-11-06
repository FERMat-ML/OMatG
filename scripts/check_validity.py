from ase.io import read

import numpy as np

import smact
from smact.screening import pauling_test

from collections import Counter
import itertools
import sys

# Adapted from CDVAE https://github.com/txie-93/cdvae/blob/main/scripts/eval_utils.py

def structural_validity(file):
    atoms = read(file, index=':')
    validities = {'True': 0, 'False': 0}
    for a in atoms:
        a.pbc = True
        dis_mat = a.get_all_distances(mic=True)
        # make diagonals larger
        dis_mat += 5. * np.identity(len(a))
        # all interatomic distances should be > 0.5 A 
        valid = np.all(dis_mat > 0.5)
        if valid:
            validities['True'] += 1
        else:
            validities['False'] += 1

    return validities

def smact_validity(file, use_pauling_test=True, include_alloys=True):
    atoms = read(file, index=':')
    validities = {'True': 0, 'False': 0}

    def get_composition(a):
        elem_counter = Counter(a.get_chemical_symbols())
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        elems = elems
        comps = tuple(counts.astype('int').tolist())
        return elems, comps

    for a in atoms: 
        elem_symbols, count = get_composition(a)
        space = smact.element_dictionary(elem_symbols)
        smact_elems = [e[1] for e in space.items()]
        electronegs = [e.pauling_eneg for e in smact_elems]
        ox_combos = [e.oxidation_states for e in smact_elems]
        if len(set(elem_symbols)) == 1:
            validities['True'] += 1
            continue
        if include_alloys:
            is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
            if all(is_metal_list):
                validities['True'] += 1
                continue

        threshold = np.max(count)
        compositions = []
        for ox_states in itertools.product(*ox_combos):
            stoichs = [(c,) for c in count]
            # Test for charge balance
            cn_e, cn_r = smact.neutral_ratios(
                ox_states, stoichs=stoichs, threshold=threshold)
            # Electronegativity test
            if cn_e:
                if use_pauling_test:
                    try:
                        electroneg_OK = pauling_test(ox_states, electronegs)
                    except TypeError:
                        # if no electronegativity data, assume it is okay
                        electroneg_OK = True
                else:
                    electroneg_OK = True
                if electroneg_OK:
                    for ratio in cn_r:
                        compositions.append(
                            tuple([elem_symbols, ox_states, ratio]))
        compositions = [(i[0], i[2]) for i in compositions]
        compositions = list(set(compositions))
        if len(compositions) > 0:
            #print (elem_symbols,stoichs, ox_combos, compositions)
            validities['True'] += 1
            continue
        else:
            #print (elem_symbols,stoichs, ox_combos, compositions)
            #print (a)
            validities['False'] += 1
    return validities

if __name__ == "__main__":
    f = sys.argv[1]
    s = structural_validity(f)
    print (s,s['True']/(s['True']+s['False']))
    c = smact_validity(f)
    print (c,c['True']/(c['True']+c['False']))

