=====
Model
=====

The Model component defines the neural network architectures that learn velocity and denoising fields for stochastic interpolants.

Overview
========

OMatG's model architecture consists of three main components:

1. **Encoder**: Processes crystal structures and time to produce embeddings
2. **Head**: Maps embeddings to output predictions (velocities, species logits)
3. **Time Embedder**: Encodes continuous time :math:`t \in [0,1]` for the model

.. code-block:: yaml

   model:
     model:
       class_path: omg.model.model.Model
       init_args:
         encoder: ...      # Structure encoder
         head: ...         # Output head
         time_embedder: ... # Time encoding

Architecture
============

Model Flow
----------

.. code-block:: text

   Input: (structures, time) → Time Embedder → time features
                             ↓
   Input: structures → Encoder → structural embeddings
                             ↓
   [time features + embeddings] → Head → predictions

The model learns to predict:

* **Velocity field** :math:`b(t, x)` for positions and lattice
* **Species logits** for discrete flow matching
* **Optional denoiser** :math:`z(t, x)` for SDE sampling

Encoders
========

CSPNet
------

The default encoder based on DiffCSP:

.. code-block:: yaml

   encoder:
     class_path: omg.model.encoders.cspnet_full.CSPNetFull
     init_args:
       hidden_dim: 256
       num_layers: 6
       num_message_passing_steps: 3
       max_neighbors: 20
       cutoff: 6.0
       use_pbc: true

**Architecture**:

* Graph neural network with periodic boundary conditions
* Message passing over atomic neighborhoods
* Captures local chemical environments

**Parameters**:

* ``hidden_dim``: Size of hidden representations
* ``num_layers``: Number of interaction layers
* ``num_message_passing_steps``: Message passing iterations
* ``max_neighbors``: Maximum neighbors per atom
* ``cutoff``: Interaction cutoff radius (Ångstroms)
* ``use_pbc``: Use periodic boundary conditions

Custom Encoders
---------------

Create custom encoders by subclassing:

.. code-block:: python

   from omg.model.encoders.encoder import Encoder
   import torch.nn as nn

   class MyEncoder(Encoder):
       def __init__(self, hidden_dim=256):
           super().__init__()
           self.layers = nn.ModuleList([...])

       def forward(self, data, time_emb):
           # data: PyG Data object
           # time_emb: (batch_size, time_dim)
           # Returns: embeddings
           pass

Heads
=====

PassThrough
-----------

Default head that outputs encoder features directly:

.. code-block:: yaml

   head:
     class_path: omg.model.heads.pass_through.PassThrough

The encoder output is used directly for predictions.

Custom Heads
------------

Add processing layers after the encoder:

.. code-block:: python

   from omg.model.heads.head import Head
   import torch.nn as nn

   class MyHead(Head):
       def __init__(self, input_dim, output_dim):
           super().__init__()
           self.layers = nn.Sequential(
               nn.Linear(input_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, output_dim),
           )

       def forward(self, embeddings):
           return self.layers(embeddings)

Time Embedders
==============

SinusoidalTimeEmbeddings
------------------------

Default time encoding using sinusoidal functions:

.. code-block:: yaml

   time_embedder:
     class_path: omg.model.model_utils.SinusoidalTimeEmbeddings
     init_args:
       dim: 256

**How it works**:

Maps time :math:`t \in [0, 1]` to high-dimensional features:

.. math::

   \text{emb}_i = \begin{cases}
   \sin(2\pi \cdot 10^{i/d} \cdot t) & i \text{ even} \\
   \cos(2\pi \cdot 10^{i/d} \cdot t) & i \text{ odd}
   \end{cases}

where :math:`d` is the embedding dimension.

Complete Configuration
======================

Basic Configuration
-------------------

.. code-block:: yaml

   model:
     model:
       class_path: omg.model.model.Model
       init_args:
         encoder:
           class_path: omg.model.encoders.cspnet_full.CSPNetFull
           init_args:
             hidden_dim: 256
             num_layers: 6
             num_message_passing_steps: 3
             max_neighbors: 20
             cutoff: 6.0
             use_pbc: true
         head:
           class_path: omg.model.heads.pass_through.PassThrough
         time_embedder:
           class_path: omg.model.model_utils.SinusoidalTimeEmbeddings
           init_args:
             dim: 256

Advanced Configuration
----------------------

With custom parameters:

.. code-block:: yaml

   model:
     model:
       class_path: omg.model.model.Model
       init_args:
         encoder:
           class_path: omg.model.encoders.cspnet_full.CSPNetFull
           init_args:
             hidden_dim: 512              # Larger model
             num_layers: 8                # Deeper network
             num_message_passing_steps: 5 # More interactions
             max_neighbors: 30            # Larger neighborhood
             cutoff: 8.0                  # Longer range
             use_pbc: true
             dropout: 0.1                 # Regularization
         head:
           class_path: omg.model.heads.pass_through.PassThrough
         time_embedder:
           class_path: omg.model.model_utils.SinusoidalTimeEmbeddings
           init_args:
             dim: 512

Model Outputs
=============

The model outputs predictions for each data field:

Position Predictions
--------------------

.. code-block:: python

   # Shape: (total_atoms, 3)
   pos_velocity = model_output['pos']

Lattice Predictions
-------------------

.. code-block:: python

   # Shape: (batch_size, 3, 3)
   cell_velocity = model_output['cell']

Species Predictions
-------------------

.. code-block:: python

   # Shape: (total_atoms, num_elements)
   species_logits = model_output['species']

Training Details
================

Loss Computation
----------------

The model is trained to minimize:

.. math::

   \mathcal{L} = \sum_{\text{field}} w_{\text{field}} \cdot \mathcal{L}_{\text{field}}

where weights are specified in:

.. code-block:: yaml

   model:
     relative_si_costs:
       species_loss: 0.0      # CSP
       pos_loss_b: 0.999      # Position velocity
       cell_loss_b: 0.001     # Cell velocity

Field-specific losses:

* **Velocity loss**: :math:`\| b^\theta(t, x_t) - \dot{x}_t \|^2`
* **Species loss**: Cross-entropy for discrete flow matching

Gradient Clipping
-----------------

Prevent exploding gradients:

.. code-block:: yaml

   trainer:
     gradient_clip_val: 0.5
     gradient_clip_algorithm: "value"

Optimization
============

Optimizer
---------

AdamW is recommended:

.. code-block:: yaml

   optimizer:
     class_path: torch.optim.AdamW
     init_args:
       lr: 0.001
       weight_decay: 0.01
       betas: [0.9, 0.999]

Learning Rate Scheduling
------------------------

Cosine annealing:

.. code-block:: yaml

   lr_scheduler:
     class_path: torch.optim.lr_scheduler.CosineAnnealingLR
     init_args:
       T_max: 2000           # Total epochs
       eta_min: 1e-7         # Minimum LR

Warm-up schedule:

.. code-block:: yaml

   lr_scheduler:
     class_path: torch.optim.lr_scheduler.LinearLR
     init_args:
       start_factor: 0.1
       total_iters: 100      # Warm-up epochs

Model Size Considerations
=========================

Small Model (Fast, Lower Quality)
----------------------------------

.. code-block:: yaml

   encoder:
     class_path: omg.model.encoders.cspnet_full.CSPNetFull
     init_args:
       hidden_dim: 128
       num_layers: 4
       num_message_passing_steps: 2

Medium Model (Balanced)
-----------------------

.. code-block:: yaml

   encoder:
     class_path: omg.model.encoders.cspnet_full.CSPNetFull
     init_args:
       hidden_dim: 256
       num_layers: 6
       num_message_passing_steps: 3

Large Model (High Quality, Slow)
---------------------------------

.. code-block:: yaml

   encoder:
     class_path: omg.model.encoders.cspnet_full.CSPNetFull
     init_args:
       hidden_dim: 512
       num_layers: 8
       num_message_passing_steps: 5

Inference
=========

Generation Process
------------------

During generation:

1. Sample initial structure :math:`x_0` from base distribution
2. For :math:`t = 0` to :math:`1` in small steps:

   a. Encode current structure and time
   b. Predict velocity/logits
   c. Update structure using ODE/SDE integrator

3. Return final structure :math:`x_1`

Generation Speed
----------------

Controlled by integration steps:

.. code-block:: yaml

   model:
     si:
       init_args:
         integration_time_steps: 1000  # Default
         # integration_time_steps: 2000  # Higher quality, 2x slower
         # integration_time_steps: 500   # Lower quality, 2x faster

Best Practices
==============

1. **Start with default architecture**: CSPNet with 256 hidden dim

2. **Tune gradually**: Adjust one hyperparameter at a time

3. **Match capacity to data**: Larger datasets need larger models

4. **Monitor validation loss**: Check for overfitting

5. **Use gradient clipping**: Essential for stable training:

   .. code-block:: yaml

      trainer:
        gradient_clip_val: 0.5

6. **Warm up learning rate**: Prevents early instability:

   .. code-block:: yaml

      lr_scheduler:
        class_path: torch.optim.lr_scheduler.LinearLR
        init_args:
          start_factor: 0.1
          total_iters: 100

7. **Save checkpoints**: Monitor and save best models:

   .. code-block:: yaml

      trainer:
        callbacks:
          - class_path: lightning.pytorch.callbacks.ModelCheckpoint
            init_args:
              monitor: "val_loss_total"
              save_top_k: 3

Troubleshooting
===============

NaN Losses
----------

**Causes**:

* Learning rate too high
* No gradient clipping
* Numerical instability

**Solutions**:

1. Add gradient clipping:

   .. code-block:: yaml

      trainer:
        gradient_clip_val: 0.5

2. Reduce learning rate:

   .. code-block:: yaml

      optimizer:
        init_args:
          lr: 0.0001  # Instead of 0.001

3. Check for inf/nan in data

Slow Training
-------------

**Solutions**:

1. Reduce model size:

   .. code-block:: yaml

      hidden_dim: 128  # Instead of 256

2. Reduce cutoff radius:

   .. code-block:: yaml

      cutoff: 5.0  # Instead of 6.0

3. Reduce max_neighbors:

   .. code-block:: yaml

      max_neighbors: 15  # Instead of 20

4. Use mixed precision:

   .. code-block:: yaml

      trainer:
        precision: "16-mixed"

Overfitting
-----------

**Solutions**:

1. Add dropout:

   .. code-block:: yaml

      encoder:
        init_args:
          dropout: 0.1

2. Increase weight decay:

   .. code-block:: yaml

      optimizer:
        init_args:
          weight_decay: 0.1  # Instead of 0.01

3. Use more data augmentation

4. Reduce model size

Next Steps
==========

* :doc:`training` - Train your model
* :doc:`generation` - Generate structures
* :doc:`../api/model` - Model API reference
* :doc:`stochastic_interpolants` - Configure SI framework
* :doc:`../development/contributing` - Contributing guidelines
