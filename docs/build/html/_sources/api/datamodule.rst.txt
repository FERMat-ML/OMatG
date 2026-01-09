==========
DataModule
==========

The ``omg.datamodule`` module handles crystal structure data loading and preprocessing.

Module Overview
===============

The DataModule provides:

* Dataset classes for LMDB, CSV, and Parquet formats
* PyTorch Lightning DataModule integration
* Lazy loading for large datasets
* Structure representation and batching
* Optional preprocessing (Niggli reduction)

Main Classes
============

.. currentmodule:: omg.datamodule

.. autosummary::
   :toctree: generated/
   :nosignatures:

   omg_datamodule.OMGDataModule
   structure_dataset.StructureDataset
   structure.Structure
   omg_data.OMGData

Detailed Documentation
======================

OMGDataModule
-------------

.. autoclass:: omg.datamodule.omg_datamodule.OMGDataModule
   :members:
   :undoc-members:
   :show-inheritance:

   PyTorch Lightning DataModule for OMatG.

   Manages train, validation, and prediction datasets along with their respective dataloaders.

   **Example**:

   .. code-block:: python

      from omg.datamodule import OMGDataModule, StructureDataset

      datamodule = OMGDataModule(
          train_dataset=StructureDataset("data/mp_20/train.lmdb"),
          val_dataset=StructureDataset("data/mp_20/val.lmdb"),
          batch_size=32,
          num_workers=4,
      )

StructureDataset
----------------

.. autoclass:: omg.datamodule.structure_dataset.StructureDataset
   :members:
   :undoc-members:
   :show-inheritance:

   Dataset for crystal structures in various formats.

   **Supported formats**:

   * LMDB: Fast binary format (recommended)
   * CSV: Text format with CIF strings
   * Parquet: Columnar format from Hugging Face

   **Example**:

   .. code-block:: python

      # LMDB format
      dataset = StructureDataset(
          file_path="data/mp_20/train.lmdb",
          lazy_storage=True,
          niggli_reduce=False,
      )

      # CSV format
      dataset = StructureDataset(
          file_path="data/structures.csv",
      )

      # Access data
      data = dataset[0]
      print(f"Atoms: {len(data.species)}")
      print(f"Cell: {data.cell}")

Structure
---------

.. autoclass:: omg.datamodule.structure.Structure
   :members:
   :undoc-members:
   :show-inheritance:

   Represents a single crystal structure.

OMGData
-------

.. autoclass:: omg.datamodule.omg_data.OMGData
   :members:
   :undoc-members:
   :show-inheritance:

   PyTorch Geometric Data object for batched structures.

   **Attributes**:

   * ``n_atoms``: (batch_size,) - number of atoms per structure
   * ``batch``: (total_atoms,) - structure index for each atom
   * ``species``: (total_atoms,) - atomic numbers
   * ``pos``: (total_atoms, 3) - Cartesian positions
   * ``cell``: (batch_size, 3, 3) - lattice vectors
   * ``ptr``: (batch_size + 1,) - cumulative atom counts
   * ``property``: dict - optional properties

Data Flow
=========

The typical data flow is:

1. **Load**: ``StructureDataset`` reads structures from disk
2. **Batch**: DataLoader batches structures using PyG
3. **Process**: Model receives batched ``OMGData`` objects
4. **Train/Generate**: Forward pass through model

File Formats
============

LMDB Format
-----------

Binary key-value store:

.. code-block:: python

   # Each value is a pickled dict:
   {
       'pos': torch.Tensor,          # Shape: (N, 3)
       'cell': torch.Tensor,         # Shape: (3, 3)
       'atomic_numbers': torch.Tensor,  # Shape: (N,)
   }

CSV Format
----------

Text file with CIF column:

.. code-block:: csv

   cif
   "data_structure1\n_cell_length_a  5.0\n..."
   "data_structure2\n_cell_length_a  6.0\n..."

Parquet Format
--------------

Columnar format from Hugging Face datasets, used directly by ``StructureDataset``.

Dataset Creation
================

From Pymatgen
-------------

.. code-block:: python

   import torch
   import lmdb
   import pickle
   from pymatgen.core import Structure

   structures = [...]  # List of Pymatgen structures

   env = lmdb.open('dataset.lmdb', map_size=1099511627776)
   with env.begin(write=True) as txn:
       for i, structure in enumerate(structures):
           data = {
               'pos': torch.tensor(structure.cart_coords),
               'cell': torch.tensor(structure.lattice.matrix),
               'atomic_numbers': torch.tensor([s.Z for s in structure.species]),
           }
           txn.put(str(i).encode(), pickle.dumps(data))
   env.close()

From ASE
--------

.. code-block:: python

   from ase.io import read

   atoms_list = [read(f) for f in xyz_files]

   env = lmdb.open('dataset.lmdb', map_size=1099511627776)
   with env.begin(write=True) as txn:
       for i, atoms in enumerate(atoms_list):
           data = {
               'pos': torch.tensor(atoms.get_positions()),
               'cell': torch.tensor(atoms.get_cell().array),
               'atomic_numbers': torch.tensor(atoms.get_atomic_numbers()),
           }
           txn.put(str(i).encode(), pickle.dumps(data))
   env.close()

See Also
========

* :doc:`../user_guide/datamodule` - User guide for data handling
* :doc:`../getting_started/datasets` - Available datasets
* :doc:`training` - Using data for training
