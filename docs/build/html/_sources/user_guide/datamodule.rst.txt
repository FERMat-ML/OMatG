==========
DataModule
==========

The DataModule handles loading, preprocessing, and batching crystal structure data for training and generation.

Overview
========

OMatG uses PyTorch Lightning's ``LightningDataModule`` pattern through ``OMGDataModule``, which manages:

* Training, validation, and prediction datasets
* Data loading with PyTorch DataLoaders
* Batching with PyTorch Geometric
* Optional preprocessing (Niggli reduction, lazy loading)

Data Formats
============

OMatG supports three input formats:

LMDB (Recommended)
------------------

Fast binary format for large datasets:

**Structure**: Key-value pairs where each value is a pickled dict:

.. code-block:: python

   {
       'pos': torch.Tensor,          # Shape: (N, 3) Cartesian coordinates
       'cell': torch.Tensor,         # Shape: (3, 3) lattice vectors
       'atomic_numbers': torch.Tensor,  # Shape: (N,) atomic numbers
   }

**Advantages**:

* Fast random access
* Memory-efficient with lazy loading
* Native OMatG format

CSV
---

Text format with CIF strings:

**Required column**: ``cif`` containing CIF-formatted structures

.. code-block:: csv

   cif
   "data_structure1\n_cell_length_a  5.0\n..."
   "data_structure2\n_cell_length_a  6.0\n..."

**Advantages**:

* Human-readable
* Easy to create from existing datasets
* Automatically converted to LMDB for lazy loading

Parquet
-------

Columnar format from Hugging Face datasets:

.. code-block:: python

   # Parquet files work directly
   file_path: "Alex-MP-20/train.parquet"

**Advantages**:

* Efficient for large datasets
* Standard format for ML datasets

Configuration
=============

Basic Configuration
-------------------

.. code-block:: yaml

   data:
     train_dataset:
       class_path: omg.datamodule.StructureDataset
       init_args:
         file_path: "data/mp_20/train.lmdb"
     val_dataset:
       class_path: omg.datamodule.StructureDataset
       init_args:
         file_path: "data/mp_20/val.lmdb"
     pred_dataset:
       class_path: omg.datamodule.StructureDataset
       init_args:
         file_path: "data/mp_20/test.lmdb"
     batch_size: 32
     num_workers: 4

With Preprocessing
------------------

.. code-block:: yaml

   data:
     train_dataset:
       class_path: omg.datamodule.StructureDataset
       init_args:
         file_path: "data/mp_20/train.lmdb"
         lazy_storage: true          # Memory-efficient loading
         niggli_reduce: true         # Standardize unit cells
     batch_size: 32
     num_workers: 4
     pin_memory: true                # Faster GPU transfer
     persistent_workers: true        # Keep workers alive

Dataset Options
===============

file_path
---------

Path to dataset file:

.. code-block:: yaml

   file_path: "data/mp_20/train.lmdb"          # Absolute or relative
   file_path: "/absolute/path/to/dataset.csv"  # Absolute path

Relative paths are resolved relative to:

1. Current working directory
2. ``omg/`` directory in OMatG repository

lazy_storage
------------

Enable lazy loading for large datasets:

.. code-block:: yaml

   lazy_storage: true   # Load structures on-demand
   lazy_storage: false  # Load all structures into memory (default)

**Use when**: Dataset doesn't fit in memory

niggli_reduce
-------------

Apply Niggli reduction to standardize cells:

.. code-block:: yaml

   niggli_reduce: true   # Standardize all cells
   niggli_reduce: false  # Use cells as-is (default)

**Effect**: Reduces cells to their unique conventional form

DataLoader Options
==================

batch_size
----------

Number of structures per batch:

.. code-block:: yaml

   batch_size: 32   # Default
   batch_size: 64   # Larger batches (more memory, faster)
   batch_size: 16   # Smaller batches (less memory, slower)

num_workers
-----------

Parallel data loading processes:

.. code-block:: yaml

   num_workers: 0   # Single process (slow, for debugging)
   num_workers: 4   # 4 parallel workers (recommended)
   num_workers: 8   # More workers (diminishing returns)

**Rule of thumb**: Start with ``num_workers = num_cpus // 2``

pin_memory
----------

Faster GPU transfer:

.. code-block:: yaml

   pin_memory: true   # Recommended for GPU training
   pin_memory: false  # For CPU training

persistent_workers
------------------

Keep workers alive between epochs:

.. code-block:: yaml

   persistent_workers: true   # Faster epoch transitions
   persistent_workers: false  # Lower memory usage

Batch Format
============

Batched Data Structure
----------------------

Data is batched using PyTorch Geometric's batching:

.. code-block:: python

   from torch_geometric.data import Data

   batch = Data(
       n_atoms=...,    # (batch_size,) - atoms per structure
       batch=...,      # (total_atoms,) - structure index per atom
       species=...,    # (total_atoms,) - atomic numbers
       pos=...,        # (total_atoms, 3) - Cartesian positions
       cell=...,       # (batch_size, 3, 3) - lattice vectors
       ptr=...,        # (batch_size + 1,) - cumulative atom counts
       property={}     # dict - optional properties
   )

Accessing Batched Data
-----------------------

.. code-block:: python

   # Get number of structures in batch
   batch_size = len(batch.n_atoms)

   # Get total number of atoms
   total_atoms = batch.n_atoms.sum()

   # Get atoms for structure i
   start_idx = batch.ptr[i]
   end_idx = batch.ptr[i + 1]
   atoms_i = batch.species[start_idx:end_idx]
   positions_i = batch.pos[start_idx:end_idx]

   # Get cell for structure i
   cell_i = batch.cell[i]

Creating Custom Datasets
=========================

From CIF Files
--------------

.. code-block:: python

   import pandas as pd
   from pymatgen.core import Structure

   # Read CIF files
   structures = [Structure.from_file(f) for f in cif_files]

   # Create CSV
   cifs = [s.to(fmt="cif") for s in structures]
   df = pd.DataFrame({'cif': cifs})
   df.to_csv('my_dataset.csv', index=False)

From Pymatgen Structures
-------------------------

.. code-block:: python

   import torch
   import lmdb
   import pickle
   from pymatgen.core import Structure

   # Create LMDB database
   env = lmdb.open('my_dataset.lmdb', map_size=1099511627776)

   with env.begin(write=True) as txn:
       for i, structure in enumerate(structures):
           data = {
               'pos': torch.tensor(structure.cart_coords),
               'cell': torch.tensor(structure.lattice.matrix),
               'atomic_numbers': torch.tensor([s.Z for s in structure.species]),
           }
           txn.put(str(i).encode(), pickle.dumps(data))

   env.close()

From ASE Atoms
--------------

.. code-block:: python

   import torch
   import lmdb
   import pickle
   from ase.io import read

   atoms_list = [read(f) for f in xyz_files]

   env = lmdb.open('my_dataset.lmdb', map_size=1099511627776)

   with env.begin(write=True) as txn:
       for i, atoms in enumerate(atoms_list):
           data = {
               'pos': torch.tensor(atoms.get_positions()),
               'cell': torch.tensor(atoms.get_cell().array),
               'atomic_numbers': torch.tensor(atoms.get_atomic_numbers()),
           }
           txn.put(str(i).encode(), pickle.dumps(data))

   env.close()

Data Splits
===========

Creating Splits
---------------

Split a dataset into train/val/test:

.. code-block:: python

   import lmdb
   import pickle

   def split_lmdb(input_path, train_ratio=0.8, val_ratio=0.1):
       # Read all data
       env = lmdb.open(input_path)
       with env.begin() as txn:
           cursor = txn.cursor()
           data = [(k, v) for k, v in cursor]

       # Split
       n_train = int(len(data) * train_ratio)
       n_val = int(len(data) * val_ratio)

       train_data = data[:n_train]
       val_data = data[n_train:n_train + n_val]
       test_data = data[n_train + n_val:]

       # Write splits
       for split_data, split_name in [
           (train_data, 'train'), (val_data, 'val'), (test_data, 'test')
       ]:
           split_env = lmdb.open(f'{split_name}.lmdb', map_size=1099511627776)
           with split_env.begin(write=True) as txn:
               for k, v in split_data:
                   txn.put(k, v)
           split_env.close()

Polymorph-Aware Splits
----------------------

Keep polymorphs together:

.. code-block:: python

   from collections import defaultdict
   from pymatgen.core import Composition

   # Group by composition
   composition_groups = defaultdict(list)
   for idx, structure in enumerate(structures):
       comp = structure.composition.reduced_formula
       composition_groups[comp].append(idx)

   # Split compositions (not structures)
   compositions = list(composition_groups.keys())
   # ... split compositions, then assign structures

Data Validation
===============

Check Dataset
-------------

.. code-block:: python

   from omg.datamodule import StructureDataset

   dataset = StructureDataset("data/mp_20/train.lmdb")

   print(f"Number of structures: {len(dataset)}")
   print(f"Elements present: {dataset.get_elements()}")

   # Check first structure
   data = dataset[0]
   print(f"Atoms: {len(data.species)}")
   print(f"Species: {data.species}")
   print(f"Cell shape: {data.cell.shape}")

Validate Structures
-------------------

.. code-block:: python

   from omg.analysis.valid_atoms import ValidAtoms

   validator = ValidAtoms()

   for data in dataset:
       is_valid = validator.validate_structure(
           data.species, data.pos, data.cell
       )
       if not is_valid:
           print(f"Invalid structure found")

Best Practices
==============

1. **Use LMDB for large datasets**: Faster and more memory-efficient

2. **Enable lazy loading**: For datasets that don't fit in memory:

   .. code-block:: yaml

      lazy_storage: true

3. **Set appropriate num_workers**: Start with 4, adjust based on CPU count

4. **Use pin_memory for GPU**: Faster data transfer:

   .. code-block:: yaml

      pin_memory: true

5. **Enable persistent_workers**: Reduces overhead between epochs:

   .. code-block:: yaml

      persistent_workers: true

6. **Validate data**: Check structures before training:

   .. code-block:: python

      python -c "from omg.datamodule import StructureDataset; \
                 d = StructureDataset('data.lmdb'); print(len(d))"

7. **Preprocess consistently**: Apply same preprocessing to all splits

8. **Monitor memory usage**: Adjust batch_size if OOM errors occur

Troubleshooting
===============

OOM (Out of Memory)
-------------------

**Solutions**:

1. Reduce batch size:

   .. code-block:: yaml

      batch_size: 16  # Instead of 32

2. Enable lazy loading:

   .. code-block:: yaml

      lazy_storage: true

3. Reduce num_workers:

   .. code-block:: yaml

      num_workers: 2

Slow Data Loading
-----------------

**Solutions**:

1. Increase num_workers:

   .. code-block:: yaml

      num_workers: 8

2. Enable persistent_workers:

   .. code-block:: yaml

      persistent_workers: true

3. Use LMDB instead of CSV

4. Use SSD instead of HDD

Next Steps
==========

* :doc:`model` - Configure model architecture
* :doc:`training` - Train your model
* :doc:`../getting_started/datasets` - Available datasets
* :doc:`../api/datamodule` - DataModule API reference
* :doc:`../development/contributing` - Contributing guidelines
