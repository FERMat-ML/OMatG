======================
Stochastic Interpolants
======================

The ``omg.si`` module implements the stochastic interpolants framework.

Module Overview
===============

The SI module provides:

* Abstract base classes for stochastic interpolants
* Continuous interpolants (linear, periodic, one-sided)
* Discrete flow matching for atomic species
* Scheduling functions (alpha, beta, gamma, sigma, tau, epsilon)
* Corrector algorithms for improved sampling

Main Classes
============

.. currentmodule:: omg.si

.. autosummary::
   :toctree: generated/
   :nosignatures:

   stochastic_interpolants.StochasticInterpolants
   single_stochastic_interpolant.SingleStochasticInterpolant
   single_stochastic_interpolant_os.SingleStochasticInterpolantOS
   single_stochastic_interpolant_identity.SingleStochasticInterpolantIdentity
   discrete_flow_matching_mask.DiscreteFlowMatchingMask
   discrete_flow_matching_uniform.DiscreteFlowMatchingUniform

Abstract Classes
================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   abstracts.StochasticInterpolant
   abstracts.Integrator

Interpolants
============

.. currentmodule:: omg.si.interpolants

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Interpolant
   LinearInterpolant
   PeriodicLinearInterpolant
   TrigonometricInterpolant
   PeriodicTrigonometricInterpolant
   EncDecInterpolant
   PeriodicEncDecInterpolant

Scheduling Functions
====================

Alpha, Beta
-----------

.. currentmodule:: omg.si.interpolants

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Alpha
   Beta

Gamma (Noise Schedule)
-----------------------

.. currentmodule:: omg.si.gamma

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Gamma
   GammaConstant
   GammaLinear
   GammaCosine

Sigma (SDE Diffusion)
---------------------

.. currentmodule:: omg.si.sigma

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Sigma

Tau (Time Schedule)
-------------------

.. currentmodule:: omg.si.tau

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Tau

Epsilon (Boundary Smoothing)
-----------------------------

.. currentmodule:: omg.si.epsilon

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Epsilon

Corrector
=========

.. currentmodule:: omg.si.corrector

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Corrector

Detailed Documentation
======================

StochasticInterpolants
----------------------

.. autoclass:: omg.si.stochastic_interpolants.StochasticInterpolants
   :members:
   :undoc-members:
   :show-inheritance:

SingleStochasticInterpolant
----------------------------

.. autoclass:: omg.si.single_stochastic_interpolant.SingleStochasticInterpolant
   :members:
   :undoc-members:
   :show-inheritance:

SingleStochasticInterpolantOS
------------------------------

.. autoclass:: omg.si.single_stochastic_interpolant_os.SingleStochasticInterpolantOS
   :members:
   :undoc-members:
   :show-inheritance:

SingleStochasticInterpolantIdentity
------------------------------------

.. autoclass:: omg.si.single_stochastic_interpolant_identity.SingleStochasticInterpolantIdentity
   :members:
   :undoc-members:
   :show-inheritance:

DiscreteFlowMatchingMask
-------------------------

.. autoclass:: omg.si.discrete_flow_matching_mask.DiscreteFlowMatchingMask
   :members:
   :undoc-members:
   :show-inheritance:

DiscreteFlowMatchingUniform
----------------------------

.. autoclass:: omg.si.discrete_flow_matching_uniform.DiscreteFlowMatchingUniform
   :members:
   :undoc-members:
   :show-inheritance:

See Also
========

* :doc:`../user_guide/stochastic_interpolants` - User guide for SI
* :doc:`sampler` - Base distributions for SI
* :doc:`model` - Neural networks for learning SI fields
