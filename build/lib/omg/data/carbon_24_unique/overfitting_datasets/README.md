### Overfitting datasets

The two sets of carbon overfitting datasets are among the highest-frequency duplicate structures found in the original 
*Carbon-24* dataset. They are intended to be used as toy datasets to benchmark specific model capabilities.

- The *Carbon-NXL* dataset contains 353 duplicate representations of a single carbon crystal structure, which have a 
  different number of atoms per unit cell (*N*), different fractional coordinates (*X*), and different lattice shapes 
  and rotations (*L*).
- The *Carbon-X* file contains 480 duplicate representations of a single carbon crystal structure, with the same number 
  of atoms per unit cell (*N*=6) and the same lattice shape (*L*), but different fractional coordinates (*X*). The 
  lattices have the same dimensions but do not have fixed orientation.

The datasets are included in this folder are split into a training and validation set.

- For the *Carbon-NXL* dataset, the validation dataset contains 3 unit cells, which have size *N*=6, 8, 10, and the 
  train dataset contains 350 unit cells ranging in size from *N*=6–16.
- For the *Carbon-X* dataset, the validation dataset contains 1 unit cell, and the train dataset contains 479 unit 
  cells. All unit cells have size *N*=6.

A tutorial notebook providing a crystallography primer is available on 
[Kaggle](https://www.kaggle.com/code/mayamartirossyan/crystal-representations-primer) utilizing *Carbon-NXL*.