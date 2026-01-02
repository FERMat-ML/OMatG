=======================
Stochastic Interpolants
=======================

The stochastic interpolants (SI) framework is the mathematical foundation of OMatG. This guide explains how SI works and how to configure different interpolants for crystal generation.

Overview
========

Stochastic interpolants provide a unified framework for generative modeling that encompasses:

* Flow matching models
* Diffusion models
* Novel hybrid approaches

The flexibility comes from customizable interpolation schedules and the choice between deterministic (ODE) or stochastic (SDE) sampling.

Mathematical Framework
======================

Core Concept
------------

A stochastic interpolant bridges samples from a base distribution :math:`p_0(x)` to samples from a target data distribution :math:`p_1(x)`:

.. math::

   x_t = \alpha(t) x_0 + \beta(t) x_1 + \gamma(t) z

where:

* :math:`t \in [0, 1]` represents time
* :math:`x_0 \sim p_0(x)` is a sample from the base distribution
* :math:`x_1 \sim p_1(x)` is a sample from the data distribution
* :math:`z \sim \mathcal{N}(0, I)` is Gaussian noise
* :math:`\alpha(t), \beta(t), \gamma(t)` are scheduling functions satisfying:

  * :math:`x(t=0) = x_0` (starts at base)
  * :math:`x(t=1) = x_1` (ends at data)

Learned Fields
--------------

The model learns a velocity field :math:`b^\theta(t, x)` and optionally a denoiser :math:`z^\theta(t, x)` from training data by minimizing:

.. math::

   \mathcal{L} = \mathbb{E}_{t, x_0, x_1, z} \left[ \| b^\theta(t, x_t) - \dot{x}_t \|^2 \right]

Generation
----------

Generate new samples by integrating from :math:`t=0` to :math:`t=1`:

**ODE (deterministic)**:

.. math::

   \frac{dx_t}{dt} = b^\theta(t, x_t), \quad x_0 \sim p_0(x)

**SDE (stochastic)**:

.. math::

   dx_t = b^\theta(t, x_t) dt + \sigma(t) dW_t, \quad x_0 \sim p_0(x)

Interpolant Types
=================

Linear Interpolant
------------------

The simplest interpolant with linear scheduling:

.. math::

   x_t = (1-t) x_0 + t x_1 + \gamma(t) z

**Configuration**:

.. code-block:: yaml

   interpolant: omg.si.interpolants.LinearInterpolant

**Use for**: Lattice vectors (non-periodic)

Periodic Linear Interpolant
----------------------------

Handles periodic boundary conditions for fractional coordinates:

.. math::

   x_t = (1-t) x_0 + t x_1 + \gamma(t) z \pmod{1}

Ensures correct interpolation across periodic boundaries.

**Configuration**:

.. code-block:: yaml

   interpolant: omg.si.interpolants.PeriodicLinearInterpolant

**Use for**: Fractional coordinates

One-Sided (OS) Interpolants
----------------------------

Explicitly assume Gaussian base distribution (:math:`x_0 \sim \mathcal{N}(0, \sigma^2 I)`):

.. math::

   x_t = \beta(t) x_1 + \gamma(t) z

**Configuration**:

.. code-block:: yaml

   class_path: omg.si.single_stochastic_interpolant_os.SingleStochasticInterpolantOS

**Advantage**: Fewer hyperparameters, faster sampling

**Use for**: Both coordinates and lattice when using Gaussian base

Discrete Flow Matching
-----------------------

For discrete variables like atomic species:

* **Mask-based**: Base distribution is fully masked tokens
* **Uniform**: Base distribution is uniform over element types

.. code-block:: yaml

   # For DNG (de novo generation)
   class_path: omg.si.discrete_flow_matching_mask.DiscreteFlowMatchingMask

   # Alternative for DNG
   class_path: omg.si.discrete_flow_matching_uniform.DiscreteFlowMatchingUniform

   # For CSP (fixed composition)
   class_path: omg.si.single_stochastic_interpolant_identity.SingleStochasticInterpolantIdentity

Configuration
=============

Basic Structure
---------------

OMatG uses ``StochasticInterpolants`` to combine multiple SIs:

.. code-block:: yaml

   model:
     si:
       class_path: omg.si.stochastic_interpolants.StochasticInterpolants
       init_args:
         stochastic_interpolants:
           - # Species SI
           - # Position SI
           - # Cell SI
         data_fields:
           - "species"
           - "pos"
           - "cell"
         integration_time_steps: 1000

The order of ``stochastic_interpolants`` must match ``data_fields``.

CSP Configuration
-----------------

For crystal structure prediction (fixed composition):

.. code-block:: yaml

   model:
     si:
       class_path: omg.si.stochastic_interpolants.StochasticInterpolants
       init_args:
         stochastic_interpolants:
           # Keep species fixed
           - class_path: omg.si.single_stochastic_interpolant_identity.SingleStochasticInterpolantIdentity

           # Fractional coordinates (periodic)
           - class_path: omg.si.single_stochastic_interpolant.SingleStochasticInterpolant
             init_args:
               interpolant: omg.si.interpolants.PeriodicLinearInterpolant
               gamma: null
               epsilon: null
               differential_equation_type: "ODE"
               integrator_kwargs:
                 method: "euler"
               velocity_annealing_factor: 10.0
               correct_center_of_mass_motion: true

           # Lattice vectors (non-periodic)
           - class_path: omg.si.single_stochastic_interpolant.SingleStochasticInterpolant
             init_args:
               interpolant: omg.si.interpolants.LinearInterpolant
               gamma: null
               epsilon: null
               differential_equation_type: "ODE"
               integrator_kwargs:
                 method: "euler"
               velocity_annealing_factor: 2.0
               correct_center_of_mass_motion: false

         data_fields: ["species", "pos", "cell"]
         integration_time_steps: 1000

DNG Configuration
-----------------

For *de novo* generation (variable composition):

.. code-block:: yaml

   model:
     si:
       class_path: omg.si.stochastic_interpolants.StochasticInterpolants
       init_args:
         stochastic_interpolants:
           # Generate species
           - class_path: omg.si.discrete_flow_matching_mask.DiscreteFlowMatchingMask

           # Fractional coordinates
           - class_path: omg.si.single_stochastic_interpolant.SingleStochasticInterpolant
             init_args:
               interpolant: omg.si.interpolants.PeriodicLinearInterpolant
               # ... same as CSP

           # Lattice vectors
           - class_path: omg.si.single_stochastic_interpolant.SingleStochasticInterpolant
             init_args:
               interpolant: omg.si.interpolants.LinearInterpolant
               # ... same as CSP

         data_fields: ["species", "pos", "cell"]
         integration_time_steps: 1000

Hyperparameters
===============

Gamma (γ)
---------

Controls noise injection:

* ``gamma: null`` - No noise (pure flow matching)
* ``gamma: omg.si.gamma.GammaConstant`` - Constant noise
* ``gamma: omg.si.gamma.GammaSchedule`` - Time-varying noise

Epsilon (ε)
-----------

Prevents numerical issues at boundaries:

* ``epsilon: null`` - No smoothing
* ``epsilon: 1e-3`` - Small smoothing near :math:`t=0` and :math:`t=1`

Differential Equation Type
---------------------------

* ``"ODE"`` - Deterministic sampling (default, faster)
* ``"SDE"`` - Stochastic sampling (can improve diversity)

Integration Method
------------------

.. code-block:: yaml

   integrator_kwargs:
     method: "euler"        # Fast, first-order
     # method: "midpoint"   # Slower, second-order
     # method: "rk4"        # Slowest, fourth-order

Velocity Annealing
------------------

Scaling factor for learned velocity:

.. code-block:: yaml

   velocity_annealing_factor: 10.0  # For positions
   velocity_annealing_factor: 2.0   # For cell

Higher values = faster changes during generation. Tune based on dataset.

Center of Mass Correction
--------------------------

.. code-block:: yaml

   correct_center_of_mass_motion: true   # For positions
   correct_center_of_mass_motion: false  # For cell

Removes net translation during generation.

Integration Steps
-----------------

Number of timesteps for numerical integration:

.. code-block:: yaml

   integration_time_steps: 1000  # Default
   # integration_time_steps: 2000  # Higher quality, slower

More steps = better quality but slower generation.

Loss Weighting
==============

Balance losses from different components:

.. code-block:: yaml

   model:
     relative_si_costs:
       species_loss: 0.0      # CSP doesn't generate species
       pos_loss_b: 0.999      # Positions dominate
       cell_loss_b: 0.001     # Small cell contribution

For DNG:

.. code-block:: yaml

   model:
     relative_si_costs:
       species_loss: 0.1
       pos_loss_b: 0.899
       cell_loss_b: 0.001

Advanced Topics
===============

Custom Interpolants
-------------------

Create custom interpolants by subclassing:

.. code-block:: python

   from omg.si.interpolants import Interpolant

   class MyInterpolant(Interpolant):
       def alpha(self, t):
           # Define α(t)
           pass

       def beta(self, t):
           # Define β(t)
           pass

See :doc:`../api/si` for the full API.

Multiple Time Scales
---------------------

Use different integration steps per component:

.. code-block:: yaml

   # Configure per-component time scales
   # (Advanced feature, see paper for details)

Corrector Steps
---------------

Improve sample quality with corrector steps:

.. code-block:: yaml

   corrector:
     class_path: omg.si.corrector.LangevinCorrector
     init_args:
       n_steps: 5
       step_size: 0.001

Next Steps
==========

* :doc:`sampler` - Configure base distributions
* :doc:`model` - Set up neural network architectures
* :doc:`training` - Train your model
* :doc:`../api/si` - SI API reference
