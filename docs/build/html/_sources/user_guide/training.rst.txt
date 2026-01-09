========
Training
========

This guide covers training OMatG models from scratch.

Overview
========

Training an OMatG model involves:

1. Preparing configuration file
2. Setting up datasets
3. Configuring trainer parameters
4. Running training
5. Monitoring progress
6. Saving checkpoints

Basic Training
==============

Minimal Command
---------------

.. code-block:: bash

   omg fit --config=config.yaml

With Logging
------------

.. code-block:: bash

   omg fit \
       --config=config.yaml \
       --trainer.logger=WandbLogger \
       --trainer.logger.name=my_experiment

From Checkpoint
---------------

Resume interrupted training:

.. code-block:: bash

   omg fit \
       --config=config.yaml \
       --ckpt_path=last.ckpt

With Random Seed
----------------

For reproducibility:

.. code-block:: bash

   omg fit \
       --config=config.yaml \
       --seed_everything=42

Configuration
=============

Complete Training Config
-------------------------

.. code-block:: yaml

   # Trainer configuration
   trainer:
     accelerator: "gpu"
     devices: 1
     max_epochs: 2000
     precision: "32-true"
     gradient_clip_val: 0.5
     gradient_clip_algorithm: "value"
     num_sanity_val_steps: 0
     enable_progress_bar: true
     callbacks:
       - class_path: lightning.pytorch.callbacks.ModelCheckpoint
         init_args:
           filename: "best_val_loss"
           save_top_k: 3
           monitor: "val_loss_total"
           mode: "min"
           save_weights_only: false
       - class_path: lightning.pytorch.callbacks.EarlyStopping
         init_args:
           monitor: "val_loss_total"
           patience: 100
           mode: "min"

   # Data configuration
   data:
     train_dataset:
       class_path: omg.datamodule.StructureDataset
       init_args:
         file_path: "data/mp_20/train.lmdb"
         lazy_storage: true
     val_dataset:
       class_path: omg.datamodule.StructureDataset
       init_args:
         file_path: "data/mp_20/val.lmdb"
         lazy_storage: true
     batch_size: 32
     num_workers: 4
     pin_memory: true
     persistent_workers: true

   # Model configuration
   model:
     si: ...              # See stochastic_interpolants guide
     sampler: ...         # See sampler guide
     model: ...           # See model guide
     relative_si_costs:
       species_loss: 0.0
       pos_loss_b: 0.999
       cell_loss_b: 0.001

   # Optimizer
   optimizer:
     class_path: torch.optim.AdamW
     init_args:
       lr: 0.001
       weight_decay: 0.01

   # Learning rate scheduler
   lr_scheduler:
     class_path: torch.optim.lr_scheduler.CosineAnnealingLR
     init_args:
       T_max: 2000
       eta_min: 1e-7

Trainer Parameters
==================

Hardware
--------

GPU Training
^^^^^^^^^^^^

.. code-block:: yaml

   trainer:
     accelerator: "gpu"
     devices: 1              # Single GPU
     # devices: [0, 1]       # Multi-GPU
     # strategy: "ddp"       # Distributed data parallel

CPU Training
^^^^^^^^^^^^

.. code-block:: yaml

   trainer:
     accelerator: "cpu"
     devices: 1

Mixed Precision
^^^^^^^^^^^^^^^

Faster training with less memory:

.. code-block:: yaml

   trainer:
     precision: "16-mixed"   # Mixed precision (recommended)
     # precision: "32-true"  # Full precision

Training Duration
-----------------

.. code-block:: yaml

   trainer:
     max_epochs: 2000            # Maximum epochs
     max_steps: -1               # Maximum steps (-1 = unlimited)
     min_epochs: null            # Minimum epochs
     max_time: "12:00:00"        # Maximum training time

Validation
----------

.. code-block:: yaml

   trainer:
     val_check_interval: 1.0     # Validate every epoch
     # val_check_interval: 0.5   # Validate twice per epoch
     # val_check_interval: 100   # Validate every 100 batches
     num_sanity_val_steps: 0     # Sanity check steps (0 = skip)

Gradient Management
-------------------

.. code-block:: yaml

   trainer:
     gradient_clip_val: 0.5
     gradient_clip_algorithm: "value"  # or "norm"
     accumulate_grad_batches: 1        # Gradient accumulation

Callbacks
=========

Model Checkpointing
-------------------

Save best models:

.. code-block:: yaml

   trainer:
     callbacks:
       - class_path: lightning.pytorch.callbacks.ModelCheckpoint
         init_args:
           dirpath: "checkpoints"
           filename: "best-{epoch:02d}-{val_loss_total:.4f}"
           save_top_k: 3
           monitor: "val_loss_total"
           mode: "min"
           save_last: true
           save_weights_only: false
           every_n_epochs: 1

Early Stopping
--------------

Stop when validation plateaus:

.. code-block:: yaml

   trainer:
     callbacks:
       - class_path: lightning.pytorch.callbacks.EarlyStopping
         init_args:
           monitor: "val_loss_total"
           patience: 100
           mode: "min"
           min_delta: 1e-4

Learning Rate Monitor
---------------------

Log learning rate:

.. code-block:: yaml

   trainer:
     callbacks:
       - class_path: lightning.pytorch.callbacks.LearningRateMonitor
         init_args:
           logging_interval: "epoch"

Logging
=======

Weights & Biases
----------------

.. code-block:: bash

   omg fit \
       --config=config.yaml \
       --trainer.logger=WandbLogger \
       --trainer.logger.name=my_experiment \
       --trainer.logger.project=omatg

TensorBoard
-----------

.. code-block:: bash

   omg fit \
       --config=config.yaml \
       --trainer.logger=TensorBoardLogger \
       --trainer.logger.save_dir=logs \
       --trainer.logger.name=my_experiment

CSV Logger
----------

.. code-block:: bash

   omg fit \
       --config=config.yaml \
       --trainer.logger=CSVLogger \
       --trainer.logger.save_dir=logs

Monitoring Training
===================

Training Metrics
----------------

OMatG logs:

* ``train_loss_total``: Total training loss
* ``train_loss_species``: Species prediction loss
* ``train_loss_pos_b``: Position velocity loss
* ``train_loss_cell_b``: Cell velocity loss
* ``learning_rate``: Current learning rate

Validation Metrics
------------------

* ``val_loss_total``: Total validation loss
* ``val_loss_species``: Species validation loss
* ``val_loss_pos_b``: Position velocity validation loss
* ``val_loss_cell_b``: Cell velocity validation loss

Healthy Training
----------------

Signs of good training:

* Smooth loss decrease
* Validation loss tracking training loss
* No NaN/inf values
* Stable learning rate

Warning signs:

* Loss spikes
* Validation loss >> training loss (overfitting)
* NaN/inf losses
* No improvement after many epochs

Hyperparameter Tuning
======================

Learning Rate
-------------

Start with 1e-3, adjust if needed:

.. code-block:: yaml

   optimizer:
     init_args:
       lr: 0.001    # Default
       # lr: 0.0001 # Lower for stability
       # lr: 0.01   # Higher for faster training

Use learning rate finder:

.. code-block:: python

   # In PyTorch Lightning
   trainer.tuner.lr_find(model)

Batch Size
----------

Larger batch = more stable gradients:

.. code-block:: yaml

   data:
     batch_size: 16   # Small (less memory)
     # batch_size: 32   # Default
     # batch_size: 64   # Large (more memory)

Weight Decay
------------

Controls regularization:

.. code-block:: yaml

   optimizer:
     init_args:
       weight_decay: 0.01    # Default
       # weight_decay: 0.0   # No regularization
       # weight_decay: 0.1   # Strong regularization

Loss Weights
------------

Balance component losses:

.. code-block:: yaml

   model:
     relative_si_costs:
       species_loss: 0.0      # CSP (fixed composition)
       pos_loss_b: 0.999      # Positions dominate
       cell_loss_b: 0.001     # Small cell contribution

For DNG:

.. code-block:: yaml

   model:
     relative_si_costs:
       species_loss: 0.1      # Generate species
       pos_loss_b: 0.899
       cell_loss_b: 0.001

Integration Steps
-----------------

More steps = slower training, better quality:

.. code-block:: yaml

   model:
     si:
       init_args:
         integration_time_steps: 1000   # Default
         # integration_time_steps: 500  # Faster, lower quality
         # integration_time_steps: 2000 # Slower, higher quality

Multi-GPU Training
==================

Data Parallel
-------------

.. code-block:: bash

   omg fit \
       --config=config.yaml \
       --trainer.accelerator=gpu \
       --trainer.devices=4 \
       --trainer.strategy=ddp

Distributed Data Parallel (Recommended)
----------------------------------------

.. code-block:: yaml

   trainer:
     accelerator: "gpu"
     devices: 4
     strategy: "ddp"
     # strategy: "ddp_find_unused_parameters_false"  # Faster variant

Tips for Multi-GPU:

* Increase batch size: ``batch_size = base_batch_size * num_gpus``
* Scale learning rate: ``lr = base_lr * num_gpus``
* Use ``num_workers > 0``

Advanced Training
=================

Gradient Accumulation
---------------------

Simulate larger batches:

.. code-block:: yaml

   trainer:
     accumulate_grad_batches: 4  # Effective batch = batch_size * 4

Useful when GPU memory is limited.

Custom Training Loop
--------------------

Override training step:

.. code-block:: python

   from omg.omg_lightning import OMGLightningModule

   class CustomOMatG(OMGLightningModule):
       def training_step(self, batch, batch_idx):
           # Custom training logic
           loss = super().training_step(batch, batch_idx)
           # Additional processing
           return loss

Warm Restarts
-------------

Restart learning rate schedule:

.. code-block:: yaml

   lr_scheduler:
     class_path: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
     init_args:
       T_0: 100
       T_mult: 2
       eta_min: 1e-7

Transfer Learning
-----------------

Load pretrained encoder:

.. code-block:: python

   # Load checkpoint
   checkpoint = torch.load('pretrained.ckpt')

   # Initialize model
   model = OMGLightningModule(config)

   # Transfer encoder weights
   model.model.encoder.load_state_dict(
       checkpoint['state_dict']['encoder']
   )

   # Freeze encoder
   for param in model.model.encoder.parameters():
       param.requires_grad = False

Troubleshooting
===============

Out of Memory
-------------

**Solutions**:

1. Reduce batch size:

   .. code-block:: yaml

      batch_size: 16

2. Enable gradient checkpointing:

   .. code-block:: yaml

      trainer:
        gradient_checkpointing: true

3. Use mixed precision:

   .. code-block:: yaml

      trainer:
        precision: "16-mixed"

4. Reduce model size:

   .. code-block:: yaml

      encoder:
        init_args:
          hidden_dim: 128

NaN Losses
----------

**Solutions**:

1. Add gradient clipping:

   .. code-block:: yaml

      trainer:
        gradient_clip_val: 0.5

2. Reduce learning rate:

   .. code-block:: yaml

      optimizer:
        init_args:
          lr: 0.0001

3. Check data for NaN/inf values

4. Use mixed precision carefully

Slow Training
-------------

**Solutions**:

1. Increase batch size:

   .. code-block:: yaml

      batch_size: 64

2. Use mixed precision:

   .. code-block:: yaml

      trainer:
        precision: "16-mixed"

3. Increase num_workers:

   .. code-block:: yaml

      data:
        num_workers: 8

4. Enable persistent workers:

   .. code-block:: yaml

      data:
        persistent_workers: true

5. Reduce integration steps:

   .. code-block:: yaml

      integration_time_steps: 500

Overfitting
-----------

**Solutions**:

1. Increase weight decay:

   .. code-block:: yaml

      optimizer:
        init_args:
          weight_decay: 0.1

2. Add dropout:

   .. code-block:: yaml

      encoder:
        init_args:
          dropout: 0.1

3. Use early stopping:

   .. code-block:: yaml

      trainer:
        callbacks:
          - class_path: lightning.pytorch.callbacks.EarlyStopping

4. Get more training data

5. Reduce model size

Best Practices
==============

1. **Start simple**: Use default configuration first

2. **Monitor training**: Use Weights & Biases or TensorBoard

3. **Save checkpoints**: Keep top-k models

4. **Use validation set**: Monitor generalization

5. **Gradient clipping**: Always enable

6. **Learning rate schedule**: Cosine annealing works well

7. **Reproducibility**: Set random seed

8. **Experiment tracking**: Log all hyperparameters

9. **Validate early**: Check first few epochs

10. **Hardware utilization**: Use GPU and multiple workers

Example Training Script
=======================

.. code-block:: python

   #!/usr/bin/env python
   """Train OMatG model."""

   import sys
   from omg.main import main

   if __name__ == "__main__":
       sys.argv = [
           "omg",
           "fit",
           "--config=config.yaml",
           "--trainer.logger=WandbLogger",
           "--trainer.logger.name=my_experiment",
           "--seed_everything=42",
       ]
       main()

Next Steps
==========

* :doc:`generation` - Generate structures with trained model
* :doc:`analysis` - Evaluate generated structures
* :doc:`../api/training` - Training API reference
* :doc:`model` - Configure model architecture
* :doc:`../development/contributing` - Contributing guidelines
