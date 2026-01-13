===============
Getting Started
===============

Welcome to OMatG! This section will help you get up and running with crystal structure generation using stochastic interpolants.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: Installation
        :link: installation
        :link-type: doc

        Install OMatG and its dependencies

    .. grid-item-card:: Quick Start
        :link: quickstart
        :link-type: doc

        Generate your first crystal structures

    .. grid-item-card:: Datasets
        :link: datasets
        :link-type: doc

        Learn about available datasets and formats

What is OMatG?
==============

OMatG (Open Materials Generation) is a state-of-the-art generative model for crystal structures that implements the
stochastic interpolants framework. It supports two main generation modes:

1. **Crystal Structure Prediction (CSP)**: Generate stable crystal structures for a given atomic composition
2. ***De Novo* Generation (DNG)**: Generate novel crystal structures with both composition and structure sampled

Key Features
============

* **Flexible framework** based on stochastic interpolants (unifying diffusion and flow matching)
* **Two generation modes**: CSP (fixed composition) and DNG (variable composition)
* **Multiple interpolants**: Linear, periodic, and one-sided stochastic interpolants
* **Discrete flow matching**: For atomic species generation
* **Pretrained models**: Available on Hugging Face
* **Comprehensive metrics**: For both CSP and DNG evaluation

Crystal Representation
======================

OMatG represents a crystalline material with *N* atoms using:

* **Lattice vectors** :math:`\mathbf{L} \in \mathbb{R}^{3×3}`: The unit cell vectors
* **Fractional coordinates** :math:`\mathbf{X} \in [0,1)^{3×N}`: Atomic positions with periodic boundary conditions
* **Atomic species** :math:`\mathbf{A} \in \mathbb{Z}^N_{>0}`: Atomic numbers of each atom

All three components are generated simultaneously using the stochastic interpolants framework for continuous variables
(:math:`\mathbf{X}, \mathbf{L}`) and discrete flow matching for atomic species (:math:`\mathbf{A}`).

Next Steps
==========

1. :doc:`installation` - Set up OMatG on your system
2. :doc:`quickstart` - Generate your first structures
3. :doc:`datasets` - Explore available training and evaluation datasets
4. :doc:`../user_guide/index` - Learn about the framework in depth

Advanced Topics
===============

* :doc:`../theoretical_background` - Deep dive into the stochastic interpolants framework
* :doc:`../configuration_files` - Comprehensive guide to YAML configuration files

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   quickstart
   datasets
