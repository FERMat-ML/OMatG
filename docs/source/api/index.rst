=============
API Reference
=============

Complete API documentation for all OMatG modules.

.. grid:: 2 2 3 3
    :gutter: 3

    .. grid-item-card:: Stochastic Interpolants
        :link: si
        :link-type: doc

        SI framework classes and interpolants

    .. grid-item-card:: Sampler
        :link: sampler
        :link-type: doc

        Base distribution samplers

    .. grid-item-card:: DataModule
        :link: datamodule
        :link-type: doc

        Data loading and preprocessing

    .. grid-item-card:: Model
        :link: model
        :link-type: doc

        Neural network architectures

    .. grid-item-card:: Analysis
        :link: analysis
        :link-type: doc

        Validation and metrics

    .. grid-item-card:: Training
        :link: training
        :link-type: doc

        Training and Lightning modules

Module Overview
===============

Core Modules
------------

:doc:`si`
    Stochastic interpolants framework, including:

    * Abstract base classes
    * Single and multi-field interpolants
    * Discrete flow matching
    * Corrector algorithms
    * Interpolant schedules (alpha, beta, gamma)

:doc:`sampler`
    Base distribution samplers:

    * Independent sampler
    * Position distributions
    * Cell distributions
    * Species distributions
    * Minimum permutation distance

:doc:`datamodule`
    Data handling:

    * Structure datasets
    * Data modules
    * Batching utilities
    * Structure representations

:doc:`model`
    Neural network components:

    * Model architecture
    * Encoders (CSPNet)
    * Heads
    * Time embeddings
    * Model utilities

:doc:`analysis`
    Evaluation tools:

    * Structure validation
    * CSP metrics
    * DNG metrics
    * Analysis utilities

:doc:`training`
    Training infrastructure:

    * Lightning modules
    * Trainers
    * CLI interface

Quick Links
===========

Frequently Used Classes
-----------------------

* :class:`omg.si.stochastic_interpolants.StochasticInterpolants` - Main SI container
* :class:`omg.sampler.independent_sampler.IndependentSampler` - Base sampler
* :class:`omg.datamodule.structure_dataset.StructureDataset` - Dataset class
* :class:`omg.model.model.Model` - Main model class
* :class:`omg.analysis.valid_atoms.ValidAtoms` - Structure validator
* :class:`omg.omg_lightning.OMGLightning` - Training module

Key Functions
-------------

* :func:`omg.main.main` - CLI entry point
* :func:`omg.omg_trainer.OMGTrainer.csp_metrics` - CSP evaluation
* :func:`omg.omg_trainer.OMGTrainer.dng_metrics` - DNG evaluation

.. toctree::
   :maxdepth: 2
   :hidden:

   si
   sampler
   datamodule
   model
   analysis
   training
