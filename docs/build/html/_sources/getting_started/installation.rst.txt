============
Installation
============

This guide will help you install OMatG and all its dependencies.

Requirements
============

* Python 3.11, 3.12, or 3.13
* PyTorch 2.8+
* PyTorch Lightning
* PyTorch Geometric (PyG) 2.7+
* PyTorch Scatter 2.1+

Basic Installation
==================

Install OMatG from the repository:

.. code-block:: bash

   git clone https://github.com/FerMat-ML/OMatG.git
   cd OMatG
   pip install .

For development (editable mode):

.. code-block:: bash

   pip install -e .

.. tip::
   When using editable mode, code changes are immediately available without reinstalling.

Recommended Installation
========================

To minimize installation errors, install PyTorch and related packages first:

.. code-block:: bash

   # Install PyTorch for your compute platform
   # See https://pytorch.org/get-started/locally/
   pip install torch==2.8.0

   # Install PyTorch Geometric
   # See https://pytorch-geometric.readthedocs.io/
   pip install torch-geometric==2.7.0

   # Install PyTorch Scatter
   pip install torch-scatter==2.1.0

   # Then install OMatG
   cd OMatG
   pip install .

Platform-Specific Installation
===============================

CUDA (GPU)
----------

For CUDA 11.8:

.. code-block:: bash

   pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu118
   pip install torch-geometric torch-scatter
   pip install .

For CUDA 12.1:

.. code-block:: bash

   pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121
   pip install torch-geometric torch-scatter
   pip install .

CPU Only
--------

.. code-block:: bash

   pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
   pip install torch-geometric torch-scatter
   pip install .

macOS (Apple Silicon)
---------------------

.. code-block:: bash

   pip install torch==2.8.0
   pip install torch-geometric torch-scatter
   pip install .

Verifying Installation
======================

Test your installation:

.. code-block:: python

   import omg
   from omg.model import Model
   from omg.si import StochasticInterpolants
   print(f"OMatG version: {omg.__version__}")

You can also check the available commands:

.. code-block:: bash

   omg --help
   omg_load --list

Dependencies
============

OMatG requires the following key dependencies (automatically installed):

Core Dependencies
-----------------

* torch >= 2.8.0
* pytorch-lightning >= 2.0.0
* torch-geometric >= 2.7.0
* torch-scatter >= 2.1.0

Scientific Computing
--------------------

* numpy
* scipy

Chemistry and Materials
-----------------------

* pymatgen >= 2023.0.0
* ase >= 3.22.0

Data and Utilities
------------------

* lmdb >= 1.0.0
* pandas
* pyyaml
* tqdm

Development Dependencies
========================

For contributors:

.. code-block:: bash

   pip install -e ".[dev]"

This installs additional tools for testing and development.

Troubleshooting
===============

Import Errors
-------------

If you see ``ModuleNotFoundError``, ensure you've installed OMatG:

.. code-block:: bash

   cd OMatG
   pip install .

CUDA Errors
-----------

If you get CUDA-related errors:

1. Check your CUDA version: ``nvidia-smi``
2. Install PyTorch with matching CUDA version
3. Verify PyTorch can see your GPU:

.. code-block:: python

   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.device_count())

PyTorch Geometric Errors
-------------------------

If PyTorch Geometric installation fails:

.. code-block:: bash

   # Try installing from wheels
   pip install torch-geometric torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu118.html

Updating OMatG
==============

To update to the latest version:

.. code-block:: bash

   cd OMatG
   git pull
   pip install .

Or for editable mode:

.. code-block:: bash

   cd OMatG
   git pull
   # Changes are automatically available

Next Steps
==========

Now that OMatG is installed:

* :doc:`quickstart` - Generate your first crystal structures
* :doc:`datasets` - Download and prepare datasets
* :doc:`../user_guide/index` - Learn about OMatG's architecture
