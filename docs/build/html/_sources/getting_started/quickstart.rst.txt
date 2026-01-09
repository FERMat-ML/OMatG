===========
Quick Start
===========

This guide will walk you through generating your first crystal structures with OMatG.

Using Pretrained Models
=======================

The fastest way to get started is using a pretrained model from Hugging Face.

Download a Pretrained Model
----------------------------

.. code-block:: bash

   # List available models
   omg_load --list

   # Download a pretrained checkpoint (example)
   # Download from https://huggingface.co/OMatG

Generate Structures
-------------------

Generate crystal structures using a pretrained checkpoint:

.. code-block:: bash

   omg predict \
       --config=path/to/config.yaml \
       --ckpt_path=path/to/checkpoint.ckpt \
       --model.generation_xyz_filename=generated_structures.xyz

This will:

1. Load the model from the checkpoint
2. Generate structures based on the prediction dataset
3. Save results to ``generated_structures.xyz``
4. Create ``generated_structures_init.xyz`` with initial random structures

Crystal Structure Prediction (CSP)
===================================

Predict structures for specific compositions:

Step 1: Create Composition File
--------------------------------

Create an LMDB file with target compositions:

.. code-block:: bash

   omg create_compositions \
       --config=config.yaml \
       --compositions='LiMn3O4' \
       --lmdb_file=target_compositions.lmdb

For multiple compositions:

.. code-block:: bash

   omg create_compositions \
       --config=config.yaml \
       --compositions='[LiMn3O4, Ga4Te4, NaCl]' \
       --lmdb_file=target_compositions.lmdb \
       --repeats=10

The ``--repeats`` argument generates multiple structures per composition.

Step 2: Generate Structures
----------------------------

.. code-block:: bash

   omg predict \
       --config=csp_config.yaml \
       --ckpt_path=csp_checkpoint.ckpt \
       --data.pred_dataset.init_args.file_path=target_compositions.lmdb \
       --model.generation_xyz_filename=predicted_structures.xyz

.. note::
   Use a CSP model checkpoint trained on datasets containing the target elements.

Step 3: Evaluate Results
-------------------------

Compute CSP metrics:

.. code-block:: bash

   omg csp_metrics \
       --config=csp_config.yaml \
       --xyz_file=predicted_structures.xyz

This generates:

* ``csp_metrics.json``: Match rates and RMSD statistics
* ``rmsds.pdf``: Histogram of root-mean-square displacements

*De Novo* Generation (DNG)
===========================

Generate novel crystal structures without fixed composition:

.. code-block:: bash

   omg predict \
       --config=dng_config.yaml \
       --ckpt_path=dng_checkpoint.ckpt \
       --model.generation_xyz_filename=novel_structures.xyz

Evaluate DNG Results
--------------------

.. code-block:: bash

   omg dng_metrics \
       --config=dng_config.yaml \
       --xyz_file=novel_structures.xyz \
       --dataset_name=mp_20

Metrics include validity, diversity, and coverage statistics saved to ``dng_metrics.json``.

Visualizing Results
===================

Compare generated structures to training data:

.. code-block:: bash

   omg visualize \
       --config=config.yaml \
       --xyz_file=generated_structures.xyz \
       --plot_name=distribution_comparison.pdf

This creates plots showing:

* Density distributions
* Volume distributions
* Number of atoms per structure
* Element distributions

Training Your Own Model
=======================

To train from scratch:

.. code-block:: bash

   omg fit --config=training_config.yaml

With Weights & Biases logging:

.. code-block:: bash

   omg fit \
       --config=training_config.yaml \
       --trainer.logger=WandbLogger \
       --trainer.logger.name=my_experiment

Resume from checkpoint:

.. code-block:: bash

   omg fit \
       --config=training_config.yaml \
       --ckpt_path=last.ckpt

Set random seed for reproducibility:

.. code-block:: bash

   omg fit \
       --config=training_config.yaml \
       --seed_everything=42

Configuration Files
===================

OMatG uses YAML configuration files. Here's a minimal example:

.. code-block:: yaml

   trainer:
     accelerator: "gpu"
     max_epochs: 2000
     precision: "32-true"

   data:
     train_dataset:
       class_path: omg.datamodule.StructureDataset
       init_args:
         file_path: "data/mp_20/train.lmdb"
     val_dataset:
       class_path: omg.datamodule.StructureDataset
       init_args:
         file_path: "data/mp_20/val.lmdb"
     batch_size: 32

   model:
     si:
       class_path: omg.si.stochastic_interpolants.StochasticInterpolants
       # ... stochastic interpolant configuration
     model:
       class_path: omg.model.model.Model
       # ... model architecture configuration

   optimizer:
     class_path: torch.optim.AdamW
     init_args:
       lr: 0.001

See ``omg/conf_examples/`` for complete configuration examples.

Common Options
==============

Batch Size
----------

Override batch size:

.. code-block:: bash

   omg predict --config=config.yaml --data.batch_size=64

Integration Steps
-----------------

Control generation quality (more steps = higher quality, slower):

.. code-block:: bash

   omg predict \
       --config=config.yaml \
       --model.si.init_args.integration_time_steps=2000

Random Seed
-----------

For reproducible generation:

.. code-block:: bash

   omg predict \
       --config=config.yaml \
       --seed_everything=42

Tutorials
=========

Interactive Jupyter notebooks:

* `OMatG Tutorial (Kaggle) <https://www.kaggle.com/code/philipphoellmer/generative-modeling-workshop-session-crystals>`__
* `Crystallography Primer (Kaggle) <https://www.kaggle.com/code/mayamartirossyan/crystal-representations-primer>`__
* `Generative Modeling Intro (Kaggle) <https://www.kaggle.com/code/philipphoellmer/generative-modeling-workshop-session-fashion>`__

Next Steps
==========

* :doc:`datasets` - Explore available datasets
* :doc:`../user_guide/stochastic_interpolants` - Understand the SI framework
* :doc:`../user_guide/training` - Deep dive into training
* :doc:`../api/index` - Browse the API reference
