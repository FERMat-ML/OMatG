==========
User Guide
==========

The user guide provides in-depth explanations of OMatG's key concepts and components.

.. grid:: 2 2 3 3
    :gutter: 3

    .. grid-item-card:: Stochastic Interpolants
        :link: stochastic_interpolants
        :link-type: doc

        The theoretical framework underlying OMatG

    .. grid-item-card:: Sampler
        :link: sampler
        :link-type: doc

        Base distributions for crystal generation

    .. grid-item-card:: DataModule
        :link: datamodule
        :link-type: doc

        Loading and preprocessing crystal structures

    .. grid-item-card:: Model
        :link: model
        :link-type: doc

        Neural network architectures and components

    .. grid-item-card:: Analysis
        :link: analysis
        :link-type: doc

        Validation and evaluation metrics

    .. grid-item-card:: Training & Generation
        :link: training
        :link-type: doc

        Training models and generating structures

Core Concepts
=============

OMatG Architecture
------------------

OMatG's architecture consists of five main components:

1. **Stochastic Interpolants (SI)**: The mathematical framework for bridging base and data distributions
2. **Sampler**: Generates initial random structures from base distributions
3. **DataModule**: Handles crystal structure data loading and batching
4. **Model**: Neural networks that learn the velocity and denoising fields
5. **Analysis**: Tools for validating and evaluating generated structures

Generation Modes
----------------

OMatG supports two generation modes:

Crystal Structure Prediction (CSP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Input**: Atomic composition (e.g., LiMnO₄)
* **Output**: Stable crystal structure with those atoms
* **Use case**: Predict stable phases for known compositions
* **Configuration**: Use ``SingleStochasticInterpolantIdentity`` for species

*De Novo* Generation (DNG)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Input**: None (or constraints like number of atoms)
* **Output**: Novel crystal structures with both composition and geometry
* **Use case**: Discover new materials
* **Configuration**: Use ``DiscreteFlowMatchingMask`` for species

Workflow Overview
=================

Training Workflow
-----------------

1. **Prepare data**: Format crystal structures as LMDB/CSV files
2. **Configure model**: Set up YAML config with SI, sampler, and model settings
3. **Train**: Run ``omg fit --config=config.yaml``
4. **Monitor**: Track losses and validation metrics

Generation Workflow
-------------------

1. **Load checkpoint**: Trained model weights
2. **Prepare targets**: Composition file (CSP) or empty (DNG)
3. **Generate**: Run ``omg predict --config=config.yaml --ckpt_path=checkpoint.ckpt``
4. **Evaluate**: Compute metrics with ``omg csp_metrics`` or ``omg dng_metrics``

Mathematical Framework
======================

Stochastic interpolants bridge a base distribution :math:`p_0(x)` to a data distribution :math:`p_1(x)` via:

.. math::

   x_t = \alpha(t) x_0 + \beta(t) x_1 + \gamma(t) z

where:

* :math:`t \in [0, 1]` is time
* :math:`x_0 \sim p_0(x)` is a base sample
* :math:`x_1 \sim p_1(x)` is a data sample
* :math:`z \sim \mathcal{N}(0, I)` is Gaussian noise
* :math:`\alpha(t), \beta(t), \gamma(t)` are scheduling functions

The model learns a velocity field :math:`b^\theta(t, x)` enabling sampling via:

**ODE sampling** (deterministic):

.. math::

   \frac{dx_t}{dt} = b^\theta(t, x_t)

**SDE sampling** (stochastic):

.. math::

   dx_t = b^\theta(t, x_t) dt + \sigma(t) dW_t

For crystal structures, this is applied to:

* **Fractional coordinates** :math:`\mathbf{X}`: Periodic interpolant
* **Lattice vectors** :math:`\mathbf{L}`: Non-periodic interpolant
* **Atomic species** :math:`\mathbf{A}`: Discrete flow matching

Configuration System
====================

OMatG uses PyTorch Lightning's CLI for configuration. YAML files specify:

.. code-block:: yaml

   trainer:
     # Training parameters
     max_epochs: 2000
     accelerator: "gpu"

   data:
     # Data loading
     train_dataset: ...
     val_dataset: ...
     batch_size: 32

   model:
     # Model architecture
     si: ...              # Stochastic interpolants
     sampler: ...         # Base distributions
     model: ...           # Neural network

   optimizer:
     # Optimization
     class_path: torch.optim.AdamW
     init_args:
       lr: 0.001

See :doc:`stochastic_interpolants` for detailed configuration examples.

Guide Contents
==============

.. toctree::
   :maxdepth: 2

   stochastic_interpolants
   sampler
   datamodule
   model
   analysis
   training
   generation
