=======
Sampler
=======

The ``omg.sampler`` module provides base distributions for crystal generation.

Module Overview
===============

Samplers generate initial random structures (:math:`x_0`) that are evolved to data samples through stochastic interpolants. The module includes:

* Independent sampler combining position, cell, and species distributions
* Position distributions (uniform, normal, mirror)
* Cell/lattice distributions (uniform, informed, normal, mirror)
* Species distributions (mask, uniform, mirror)
* Minimum permutation distance utilities

Main Classes
============

.. currentmodule:: omg.sampler

.. autosummary::
   :toctree: generated/
   :nosignatures:

   independent_sampler.IndependentSampler
   sampler.Sampler

Abstract Classes
================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   abstracts.PositionDistribution
   abstracts.CellDistribution
   abstracts.SpeciesDistribution

Position Distributions
======================

.. currentmodule:: omg.sampler.position_distributions

.. autosummary::
   :toctree: generated/
   :nosignatures:

   UniformPositionDistribution
   NormalPositionDistribution
   MirrorPosition

Cell Distributions
==================

.. currentmodule:: omg.sampler.cell_distributions

.. autosummary::
   :toctree: generated/
   :nosignatures:

   InformedLatticeDistribution
   NormalCellDistribution
   MirrorCell

Species Distributions
=====================

.. currentmodule:: omg.sampler.species_distributions

.. autosummary::
   :toctree: generated/
   :nosignatures:

   MaskSpeciesDistribution
   UniformSpeciesDistribution
   MirrorSpecies

Utilities
=========

.. currentmodule:: omg.sampler.minimum_permutation_distance

.. autosummary::
   :toctree: generated/

   compute_minimum_permutation_distance

Detailed Documentation
======================

IndependentSampler
------------------

.. autoclass:: omg.sampler.independent_sampler.IndependentSampler
   :members:
   :undoc-members:
   :show-inheritance:

Position Distributions
----------------------

.. autoclass:: omg.sampler.position_distributions.UniformPositionDistribution
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: omg.sampler.position_distributions.NormalPositionDistribution
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: omg.sampler.position_distributions.MirrorPosition
   :members:
   :undoc-members:
   :show-inheritance:

Cell Distributions
------------------

.. autoclass:: omg.sampler.cell_distributions.InformedLatticeDistribution
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: omg.sampler.cell_distributions.NormalCellDistribution
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: omg.sampler.cell_distributions.MirrorCell
   :members:
   :undoc-members:
   :show-inheritance:

Species Distributions
---------------------

.. autoclass:: omg.sampler.species_distributions.MaskSpeciesDistribution
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: omg.sampler.species_distributions.UniformSpeciesDistribution
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: omg.sampler.species_distributions.MirrorSpecies
   :members:
   :undoc-members:
   :show-inheritance:

See Also
========

* :doc:`../user_guide/sampler` - User guide for samplers
* :doc:`si` - Stochastic interpolants framework
* :doc:`datamodule` - Data loading for training distributions
