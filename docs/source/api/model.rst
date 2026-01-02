=====
Model
=====

The ``omg.model`` module contains neural network architectures for learning velocity and denoising fields.

Module Overview
===============

The model module provides:

* Main model class combining encoder, head, and time embedder
* Encoder architectures (CSPNet)
* Head modules for output processing
* Time embedding utilities
* Model utilities and helpers

Main Classes
============

.. currentmodule:: omg.model

.. autosummary::
   :toctree: generated/
   :nosignatures:

   model.Model

Encoders
========

.. currentmodule:: omg.model.encoders

.. autosummary::
   :toctree: generated/
   :nosignatures:

   encoder.Encoder
   cspnet_full.CSPNetFull

Heads
=====

.. currentmodule:: omg.model.heads

.. autosummary::
   :toctree: generated/
   :nosignatures:

   head.Head
   pass_through.PassThrough

Model Utilities
===============

.. currentmodule:: omg.model.model_utils

.. autosummary::
   :toctree: generated/

   SinusoidalTimeEmbeddings

Detailed Documentation
======================

Model
-----

.. autoclass:: omg.model.model.Model
   :members:
   :undoc-members:
   :show-inheritance:

   Main model class for OMatG.

   Combines an encoder (processes structures), a time embedder (encodes time), and a head (produces outputs).

   **Example**:

   .. code-block:: python

      from omg.model import Model
      from omg.model.encoders import CSPNetFull
      from omg.model.heads import PassThrough
      from omg.model.model_utils import SinusoidalTimeEmbeddings

      model = Model(
          encoder=CSPNetFull(hidden_dim=256, num_layers=6),
          head=PassThrough(),
          time_embedder=SinusoidalTimeEmbeddings(dim=256),
      )

      # Forward pass
      outputs = model(data, time)

Encoders
--------

CSPNetFull
^^^^^^^^^^

.. autoclass:: omg.model.encoders.cspnet_full.CSPNetFull
   :members:
   :undoc-members:
   :show-inheritance:

   Graph neural network encoder based on DiffCSP.

   **Features**:

   * Message passing over atomic neighborhoods
   * Periodic boundary conditions
   * Captures local chemical environments

   **Parameters**:

   * ``hidden_dim``: Size of hidden representations (default: 256)
   * ``num_layers``: Number of interaction layers (default: 6)
   * ``num_message_passing_steps``: Message passing iterations (default: 3)
   * ``max_neighbors``: Maximum neighbors per atom (default: 20)
   * ``cutoff``: Interaction cutoff radius in Ångstroms (default: 6.0)
   * ``use_pbc``: Use periodic boundary conditions (default: True)

   **Example**:

   .. code-block:: python

      encoder = CSPNetFull(
          hidden_dim=512,
          num_layers=8,
          num_message_passing_steps=5,
          max_neighbors=30,
          cutoff=8.0,
          use_pbc=True,
      )

Heads
-----

PassThrough
^^^^^^^^^^^

.. autoclass:: omg.model.heads.pass_through.PassThrough
   :members:
   :undoc-members:
   :show-inheritance:

   Default head that passes encoder output directly.

   No additional processing layers - encoder output is used directly for predictions.

   **Example**:

   .. code-block:: python

      head = PassThrough()
      output = head(encoder_output)

Time Embeddings
---------------

SinusoidalTimeEmbeddings
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: omg.model.model_utils.SinusoidalTimeEmbeddings
   :members:
   :undoc-members:
   :show-inheritance:

   Sinusoidal time embeddings for continuous time encoding.

   Maps time :math:`t \in [0, 1]` to high-dimensional features using sinusoidal functions:

   .. math::

      \text{emb}_i = \begin{cases}
      \sin(2\pi \cdot 10^{i/d} \cdot t) & i \text{ even} \\
      \cos(2\pi \cdot 10^{i/d} \cdot t) & i \text{ odd}
      \end{cases}

   **Parameters**:

   * ``dim``: Embedding dimension (default: 256)

   **Example**:

   .. code-block:: python

      time_embedder = SinusoidalTimeEmbeddings(dim=256)
      time_emb = time_embedder(t)  # t: (batch_size,)
      # time_emb: (batch_size, 256)

Model Architecture
==================

The model architecture is:

.. code-block:: text

   Input: (structures, time)
        ↓
   Time Embedder: time → time_emb
        ↓
   Encoder: (structures, time_emb) → embeddings
        ↓
   Head: embeddings → predictions
        ↓
   Output: {species, pos, cell} predictions

Predictions include:

* **species**: Logits for discrete species (shape: total_atoms × num_elements)
* **pos**: Velocity for positions (shape: total_atoms × 3)
* **cell**: Velocity for lattice (shape: batch_size × 3 × 3)

Creating Custom Components
===========================

Custom Encoder
--------------

.. code-block:: python

   from omg.model.encoders.encoder import Encoder
   import torch.nn as nn

   class MyEncoder(Encoder):
       def __init__(self, hidden_dim=256):
           super().__init__()
           self.layers = nn.ModuleList([
               nn.Linear(hidden_dim, hidden_dim)
               for _ in range(6)
           ])

       def forward(self, data, time_emb):
           # data: PyG Data object
           # time_emb: (batch_size, time_dim)
           # Return: embeddings
           pass

Custom Head
-----------

.. code-block:: python

   from omg.model.heads.head import Head
   import torch.nn as nn

   class MyHead(Head):
       def __init__(self, input_dim, output_dim):
           super().__init__()
           self.fc = nn.Linear(input_dim, output_dim)

       def forward(self, embeddings):
           return self.fc(embeddings)

See Also
========

* :doc:`../user_guide/model` - User guide for model architecture
* :doc:`training` - Training models
* :doc:`si` - Stochastic interpolants that use the model
