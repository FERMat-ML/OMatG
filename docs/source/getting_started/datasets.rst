========
Datasets
========

OMatG supports multiple datasets for training and evaluation. This guide covers included datasets,
downloadable datasets, and data formats.

Included Datasets
=================

Several standard datasets are included in the repository under ``omg/data/``:

MP-20
-----

* **Size**: 45,229 structures
* **Source**: `Materials Project <https://materialsproject.org/>`__
* **Characteristics**: Maximum 20 atoms per structure
* **Location**: ``omg/data/mp_20/``
* **Use case**: General crystal structure modeling

.. code-block:: python

   data:
     train_dataset:
       class_path: omg.datamodule.StructureDataset
       init_args:
         file_path: "data/mp_20/train.lmdb"

MPTS-52
-------

* **Size**: 40,476 structures
* **Source**: Chronological split of Materials Project
* **Characteristics**: Up to 52 atoms per structure
* **Reference**: `MatBench Discovery <https://joss.theoj.org/papers/10.21105/joss.05618>`__
* **Location**: ``omg/data/mpts_52/``
* **Use case**: Temporal generalization testing

Perov-5
-------

* **Size**: 18,928 structures
* **Source**: `Perovskite dataset <https://pubs.rsc.org/en/content/articlelanding/2012/ee/c2ee22341d>`__
* **Characteristics**: All structures have 5 atoms (ABX₃ formula)
* **Location**: ``omg/data/perov_5/``
* **Use case**: Perovskite-specific modeling

Carbon-24
---------

* **Size**: 10,153 structures
* **Characteristics**: Pure carbon, up to 24 atoms
* **Source**: `Carbon allotropes <https://arxiv.org/abs/2110.06197>`__
* **Location**: ``omg/data/carbon_24/``
* **Use case**: Single-element structure exploration

Benchmark Datasets
==================

New benchmark datasets from the NeurIPS 2025 paper:

Polymorph-Split Datasets
-------------------------

Datasets split such that polymorphs (same composition, different structure) are in the same split:

* **Perov-5-polymorph-split**: ``omg/data/perov_5_ps/``
* **MP-20-polymorph-split**: ``omg/data/mp_20_ps/``

These ensure fair evaluation by preventing test set "leakage" through polymorphs.

Carbon-24-Unique Variants
--------------------------

Duplicate-pruned versions of Carbon-24:

* **Carbon-24-unique**: 4,250 unique structures (random split)
* **Carbon-24-unique-N-split**: Split by atom count

  * Low-to-high: Train 6-10 atoms, val 12-14, test 16-24
  * High-to-low: Train 10-24 atoms, val 8, test 6

* **Location**: ``omg/data/carbon_24_unique/``

Overfitting Test Datasets
--------------------------

Small datasets for testing symmetry handling:

* **Carbon-X**: 480 duplicates, varying only fractional coordinates
* **Carbon-NXL**: 353 duplicates, varying coordinates, cell shape, and atom count
* **Location**: ``omg/data/carbon_24_unique/overfitting_datasets/``

Downloadable Datasets
=====================

Large datasets available on `Hugging Face <https://huggingface.co/OMatG/datasets>`__:

Listing Available Datasets
---------------------------

.. code-block:: bash

   omg_load --list

Downloading Datasets
--------------------

.. code-block:: bash

   # Download a specific dataset
   omg_load Alex-MP-20_Polymorph_Split

   # Downloads to ./Alex-MP-20_Polymorph_Split/

Available Downloads
-------------------

Alex-MP-20
^^^^^^^^^^

* **Size**: 675,204 structures
* **Sources**: Alexandria + MP-20
* **Reference**: `Consolidated dataset <https://www.nature.com/articles/s41586-025-08628-5>`__
* **Note**: 10% held out for test set (differs from MatterGen split)

.. code-block:: bash

   omg_load Alex-MP-20

Alex-MP-20-Polymorph-Split
^^^^^^^^^^^^^^^^^^^^^^^^^^

Polymorph-aware split of Alex-MP-20:

.. code-block:: bash

   omg_load Alex-MP-20_Polymorph_Split

Carbon-24-Unique-With-Enantiomorphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Size**: 4,330 structures
* **Feature**: Enantiomorph pairs labeled and retained
* **Note**: PyMatgen's StructureMatcher cannot distinguish chiral pairs

.. code-block:: bash

   omg_load Carbon-24-Unique-With-Enantiomorphs

Carbon-Enantiomorphs
^^^^^^^^^^^^^^^^^^^^

* **Size**: 160 chiral structures (80 pairs)
* **Use case**: Testing chiral structure generation

.. code-block:: bash

   omg_load Carbon-Enantiomorphs

Data Formats
============

LMDB Format
-----------

OMatG's native format. Each record contains:

**Key**: Encoded string (arbitrary identifier)

**Value**: Pickled dictionary with:

.. code-block:: python

   {
       'pos': torch.Tensor,      # Shape: (N, 3) - Cartesian coordinates
       'cell': torch.Tensor,     # Shape: (3, 3) - Lattice vectors
       'atomic_numbers': torch.Tensor,  # Shape: (N,) - Atomic numbers
   }

CSV Format
----------

CSV files with a ``cif`` column containing CIF-formatted structures:

.. code-block:: python

   data:
     train_dataset:
       class_path: omg.datamodule.StructureDataset
       init_args:
         file_path: "data/my_structures.csv"

The dataset automatically parses CIF strings to extract cell, positions, and atomic numbers.

Parquet Format
--------------

Datasets from Hugging Face are in Parquet format, directly usable:

.. code-block:: python

   data:
     train_dataset:
       class_path: omg.datamodule.StructureDataset
       init_args:
         file_path: "Alex-MP-20/train.parquet"

Lazy Loading
============

For large datasets, use lazy loading to reduce memory usage:

.. code-block:: yaml

   data:
     train_dataset:
       class_path: omg.datamodule.StructureDataset
       init_args:
         file_path: "data/large_dataset.lmdb"
         lazy_storage: true

CSV files are automatically converted to LMDB for fast lazy access.

Data Preprocessing
==================

Niggli Reduction
----------------

Apply Niggli reduction to standardize unit cells:

.. code-block:: yaml

   data:
     train_dataset:
       class_path: omg.datamodule.StructureDataset
       init_args:
         file_path: "data/mp_20/train.lmdb"
         niggli_reduce: true

Creating Custom Datasets
=========================

From CIF Files
--------------

Create a CSV with a ``cif`` column:

.. code-block:: python

   import pandas as pd
   from pymatgen.core import Structure

   structures = [Structure.from_file(f) for f in cif_files]
   cifs = [s.to(fmt="cif") for s in structures]

   df = pd.DataFrame({'cif': cifs})
   df.to_csv('my_dataset.csv', index=False)

From ASE Atoms
--------------

Convert ASE Atoms to OMatG format:

.. code-block:: python

   import torch
   from ase.io import read
   import lmdb
   import pickle

   atoms = read('structure.cif')

   data = {
       'pos': torch.tensor(atoms.get_positions()),
       'cell': torch.tensor(atoms.get_cell().array),
       'atomic_numbers': torch.tensor(atoms.get_atomic_numbers()),
   }

   env = lmdb.open('my_dataset.lmdb', map_size=1099511627776)
   with env.begin(write=True) as txn:
       txn.put(b'0', pickle.dumps(data))
   env.close()

Batch Data Handling
===================

Data is batched using PyTorch Geometric's batching:

.. code-block:: python

   # Batched data attributes
   batch.n_atoms       # (batch_size,) - atoms per structure
   batch.batch         # (total_atoms,) - structure index per atom
   batch.species       # (total_atoms,) - atomic numbers
   batch.pos           # (total_atoms, 3) - atomic positions
   batch.cell          # (batch_size, 3, 3) - lattice vectors
   batch.ptr           # (batch_size + 1,) - index boundaries
   batch.property      # dict - additional properties

Configuring DataLoaders
------------------------

.. code-block:: yaml

   data:
     batch_size: 32
     num_workers: 4           # Parallel data loading
     pin_memory: true         # Faster GPU transfer
     persistent_workers: true # Keep workers alive between epochs

Dataset Statistics
==================

View dataset statistics:

.. code-block:: python

   from omg.datamodule import StructureDataset

   dataset = StructureDataset("data/mp_20/train.lmdb")

   print(f"Number of structures: {len(dataset)}")
   print(f"Elements: {dataset.get_elements()}")
   print(f"Max atoms: {max(len(d.species) for d in dataset)}")

Next Steps
==========

* :doc:`../user_guide/datamodule` - Deep dive into data handling
* :doc:`../user_guide/training` - Train models on your data
* :doc:`../api/datamodule` - DataModule API reference
