*******************
OMatG Documentation
*******************

**OMatG** is a state-of-the-art generative model for crystal structure prediction and *de novo* generation of inorganic crystals.

**Links**:
`GitHub Repository <https://github.com/FerMat-ML/OMatG>`__ |
`Huggingface Page <https://huggingface.co/OMatG>`__ |
`ICML 2025 Paper <https://openreview.net/forum?id=gHGrzxFujU>`__ |
`NeurIPS 2025 Paper <https://openreview.net/forum?id=ig9ujp50D4>`__

**Version**: 1.2.0 | **Date**: December 2025

This open-source framework accompanies the `ICML 2025 paper <https://openreview.net/forum?id=gHGrzxFujU>`__ about the
generative model itself (expanded version available on `arXiv <https://arxiv.org/abs/2502.02582>`__), and the
`NeurIPS 2025 paper <https://openreview.net/forum?id=ig9ujp50D4>`__ about newly introduced benchmark metrics and datasets
(expanded version available on `arXiv <https://arxiv.org/abs/2509.12178>`__).

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Getting started
        :link: getting_started/index
        :link-type: doc
        :text-align: center
        :class-card: sd-border-2

        New to OMatG? Check out the getting started guide for installation,
        basic concepts, and your first crystal generation.

    .. grid-item-card:: User guide
        :link: user_guide/index
        :link-type: doc
        :text-align: center
        :class-card: sd-border-2

        The user guide provides in-depth information on the key concepts of OMatG,
        including stochastic interpolants, samplers, data handling, and models.

    .. grid-item-card:: API reference
        :link: api/index
        :link-type: doc
        :text-align: center
        :class-card: sd-border-2

        The reference guide contains detailed descriptions of all classes and functions
        in OMatG. It assumes understanding of the key concepts.

    .. grid-item-card:: Development
        :link: development/index
        :link-type: doc
        :text-align: center
        :class-card: sd-border-2

        Want to contribute to OMatG? See the contributing guide for how to set up
        your development environment and contribute code.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   getting_started/index
   getting_started/installation
   getting_started/quickstart
   getting_started/datasets

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   user_guide/index
   user_guide/stochastic_interpolants
   user_guide/sampler
   user_guide/datamodule
   user_guide/model
   user_guide/analysis
   user_guide/training
   user_guide/generation

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api/index
   api/si
   api/sampler
   api/datamodule
   api/model
   api/analysis
   api/training

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Development

   development/index
   development/contributing
