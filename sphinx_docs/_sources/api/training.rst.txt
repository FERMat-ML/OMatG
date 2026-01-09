========
Training
========

The ``omg`` top-level module provides training infrastructure and CLI tools.

Module Overview
===============

Training components:

* Lightning module for model training
* Trainer with CSP and DNG metrics
* CLI interface for all operations
* Utilities and configuration helpers

Main Classes
============

.. currentmodule:: omg

.. autosummary::
   :toctree: generated/
   :nosignatures:

   omg_lightning.OMGLightning
   omg_trainer.OMGTrainer
   omg_cli.OMGCLI

CLI Functions
=============

.. autosummary::
   :toctree: generated/

   main.main

Utilities
=========

.. autosummary::
   :toctree: generated/

   utils

Detailed Documentation
======================

OMGLightning
------------

.. autoclass:: omg.omg_lightning.OMGLightning
   :members:
   :undoc-members:
   :show-inheritance:

   PyTorch Lightning module for OMatG.

   Handles training, validation, and prediction loops for stochastic interpolants.

   **Example**:

   .. code-block:: python

      from omg.omg_lightning import OMGLightning
      from omg.si import StochasticInterpolants
      from omg.sampler import IndependentSampler
      from omg.model import Model

      module = OMGLightning(
          si=StochasticInterpolants(...),
          sampler=IndependentSampler(...),
          model=Model(...),
          relative_si_costs={'pos_loss_b': 0.999, 'cell_loss_b': 0.001},
      )

      # Use with Lightning Trainer
      from pytorch_lightning import Trainer

      trainer = Trainer(max_epochs=2000)
      trainer.fit(module, datamodule=datamodule)

   **Key Methods**:

   * ``training_step(batch, batch_idx)``: Training step
   * ``validation_step(batch, batch_idx)``: Validation step
   * ``predict_step(batch, batch_idx)``: Generation step
   * ``configure_optimizers()``: Optimizer setup

OMGTrainer
----------

.. autoclass:: omg.omg_trainer.OMGTrainer
   :members:
   :undoc-members:
   :show-inheritance:

   Extended trainer with evaluation metrics.

   Adds CSP and DNG metrics computation to Lightning Trainer.

   **Example**:

   .. code-block:: python

      from omg.omg_trainer import OMGTrainer

      trainer = OMGTrainer(max_epochs=2000)
      trainer.fit(model, datamodule=datamodule)

      # Compute CSP metrics
      trainer.csp_metrics(
          config=config,
          xyz_file='generated.xyz',
      )

      # Compute DNG metrics
      trainer.dng_metrics(
          config=config,
          xyz_file='generated.xyz',
          dataset_name='mp_20',
      )

   **Key Methods**:

   * ``csp_metrics(...)``: Compute CSP evaluation metrics
   * ``dng_metrics(...)``: Compute DNG evaluation metrics
   * ``visualize(...)``: Create distribution plots

CLI Interface
=============

Main CLI
--------

.. autofunction:: omg.main.main

   Entry point for ``omg`` command.

   **Commands**:

   * ``fit``: Train a model
   * ``predict``: Generate structures
   * ``validate``: Validate model
   * ``test``: Test model

   **Example**:

   .. code-block:: bash

      omg fit --config=config.yaml
      omg predict --config=config.yaml --ckpt_path=model.ckpt
      omg test --config=config.yaml --ckpt_path=model.ckpt

OMatG CLI
---------

.. autofunction:: omg.omg_cli.cli

   Additional CLI commands.

   **Commands**:

   * ``csp_metrics``: Compute CSP metrics
   * ``dng_metrics``: Compute DNG metrics
   * ``visualize``: Create plots
   * ``create_compositions``: Create composition datasets

   **Example**:

   .. code-block:: bash

      omg csp_metrics --xyz_file=generated.xyz
      omg dng_metrics --xyz_file=generated.xyz --dataset_name=mp_20
      omg visualize --xyz_file=generated.xyz

Training Workflow
=================

Basic Training
--------------

.. code-block:: python

   from pytorch_lightning import Trainer
   from omg.omg_lightning import OMGLightningModule
   from omg.datamodule import OMGDataModule

   # Create module
   model = OMGLightningModule(
       si=si,
       sampler=sampler,
       model=model,
       relative_si_costs=costs,
   )

   # Create data
   datamodule = OMGDataModule(
       train_dataset=train_dataset,
       val_dataset=val_dataset,
       batch_size=32,
   )

   # Train
   trainer = Trainer(
       max_epochs=2000,
       accelerator='gpu',
       gradient_clip_val=0.5,
   )
   trainer.fit(model, datamodule=datamodule)

With Callbacks
--------------

.. code-block:: python

   from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

   trainer = Trainer(
       max_epochs=2000,
       callbacks=[
           ModelCheckpoint(
               monitor='val_loss_total',
               save_top_k=3,
               mode='min',
           ),
           EarlyStopping(
               monitor='val_loss_total',
               patience=100,
               mode='min',
           ),
       ],
   )

With Logging
------------

.. code-block:: python

   from pytorch_lightning.loggers import WandbLogger

   logger = WandbLogger(
       project='omatg',
       name='my_experiment',
   )

   trainer = Trainer(
       max_epochs=2000,
       logger=logger,
   )

Generation Workflow
===================

Basic Generation
----------------

.. code-block:: python

   from omg.omg_lightning import OMGLightningModule

   # Load checkpoint
   model = OMGLightningModule.load_from_checkpoint('best.ckpt')

   # Generate
   trainer = Trainer()
   predictions = trainer.predict(model, datamodule=datamodule)

   # Save to XYZ
   model.generation_xyz_filename = 'generated.xyz'
   trainer.predict(model, datamodule=datamodule)

Evaluation Workflow
===================

CSP Evaluation
--------------

.. code-block:: python

   from omg.omg_trainer import OMGTrainer

   trainer = OMGTrainer()

   # Compute metrics
   results = trainer.csp_metrics(
       config=config,
       xyz_file='generated.xyz',
       skip_validation=False,
       metre=False,
   )

   print(f"Match rate: {results['match_rate']:.2%}")
   print(f"Avg RMSD: {results['avg_rmsd']:.3f} Å")

DNG Evaluation
--------------

.. code-block:: python

   results = trainer.dng_metrics(
       config=config,
       xyz_file='generated.xyz',
       dataset_name='mp_20',
       skip_validation=False,
   )

   print(f"Validity: {results['validity_overall']:.2%}")
   print(f"Coverage: {results['coverage_recall']:.2%}")

Configuration
=============

YAML Configuration
------------------

OMatG uses YAML files for configuration:

.. code-block:: yaml

   # trainer
   trainer:
     accelerator: gpu
     max_epochs: 2000

   # data
   data:
     train_dataset: ...
     val_dataset: ...
     batch_size: 32

   # model
   model:
     si: ...
     sampler: ...
     model: ...

   # optimizer
   optimizer:
     class_path: torch.optim.AdamW
     init_args:
       lr: 0.001

Loading Configuration
---------------------

.. code-block:: python

   import yaml

   with open('config.yaml') as f:
       config = yaml.safe_load(f)

   # Use with Lightning CLI
   from pytorch_lightning.cli import LightningCLI

   cli = LightningCLI(
       OMGLightning,
       OMGDataModule,
       args=['--config', 'config.yaml'],
   )

See Also
========

* :doc:`../user_guide/training` - Training user guide
* :doc:`../user_guide/generation` - Generation user guide
* :doc:`../getting_started/quickstart` - Quick start guide
