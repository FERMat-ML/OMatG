========
Samplers
========

Samplers define base distributions for generating initial random structures. They must match the stochastic interpolants used in your model.

Overview
========

OMatG samplers generate random initial structures (:math:`x_0`) that are evolved to data samples (:math:`x_1`) through stochastic interpolants. The sampler must be compatible with the chosen SI:

* **SI with arbitrary base** → flexible sampler choice
* **One-sided SI** → must use Gaussian sampler
* **Identity SI** → must use mirror sampler
* **Discrete flow matching (mask)** → must use mask sampler
* **Discrete flow matching (uniform)** → must use uniform sampler

Sampler Architecture
====================

Independent Sampler
-------------------

The main sampler class combines three independent distributions:

.. code-block:: yaml

   model:
     sampler:
       class_path: omg.sampler.IndependentSampler
       init_args:
         species_distribution: ...  # For atomic species
         pos_distribution: ...      # For positions
         cell_distribution: ...     # For lattice vectors

Position Distributions
======================

UniformPositionDistribution
---------------------------

Samples fractional coordinates uniformly in :math:`[0,1)^3`:

.. code-block:: yaml

   pos_distribution:
     class_path: omg.sampler.position_distributions.UniformPositionDistribution

**Use with**: ``PeriodicLinearInterpolant``

NormalPositionDistribution
--------------------------

Samples from standard normal distribution:

.. code-block:: yaml

   pos_distribution:
     class_path: omg.sampler.position_distributions.NormalPositionDistribution

**Use with**: One-sided SI (``SingleStochasticInterpolantOS``)

MirrorPositions
---------------

Copies positions from data (for CSP):

.. code-block:: yaml

   pos_distribution:
     class_path: omg.sampler.position_distributions.MirrorPositions

**Use with**: ``SingleStochasticInterpolantIdentity``

Cell Distributions
==================

InformedLatticeDistribution
---------------------------

Samples from dataset statistics (recommended):

.. code-block:: yaml

   cell_distribution:
     class_path: omg.sampler.cell_distributions.InformedLatticeDistribution
     init_args:
       dataset_name: mp_20  # or: perov_5, carbon_24, mpts_52

This uses precomputed lattice statistics from the specified dataset, providing realistic initial cells.

Available datasets:

* ``mp_20``
* ``perov_5``
* ``carbon_24``
* ``mpts_52``

UniformLatticeDistribution
--------------------------

Samples lattice parameters uniformly:

.. code-block:: yaml

   cell_distribution:
     class_path: omg.sampler.cell_distributions.UniformLatticeDistribution
     init_args:
       min_length: 3.0
       max_length: 15.0
       min_angle: 60.0
       max_angle: 120.0

NormalLatticeDistribution
-------------------------

Samples from normal distribution:

.. code-block:: yaml

   cell_distribution:
     class_path: omg.sampler.cell_distributions.NormalLatticeDistribution

**Use with**: One-sided SI

MirrorCell
----------

Copies cell from data (for CSP):

.. code-block:: yaml

   cell_distribution:
     class_path: omg.sampler.cell_distributions.MirrorCell

**Use with**: ``SingleStochasticInterpolantIdentity``

Species Distributions
=====================

MaskSpecies
-----------

All atoms masked (for DNG):

.. code-block:: yaml

   species_distribution:
     class_path: omg.sampler.species_distributions.MaskSpecies

**Use with**: ``DiscreteFlowMatchingMask``

**Mask token**: Uses atomic number 0 to represent masked atoms

UniformSpeciesDistribution
--------------------------

Uniform over element types (for DNG):

.. code-block:: yaml

   species_distribution:
     class_path: omg.sampler.species_distributions.UniformSpeciesDistribution
     init_args:
       elements: [1, 6, 7, 8]  # H, C, N, O

**Use with**: ``DiscreteFlowMatchingUniform``

MirrorSpecies
-------------

Copies species from data (for CSP):

.. code-block:: yaml

   species_distribution:
     class_path: omg.sampler.species_distributions.MirrorSpecies

**Use with**: ``SingleStochasticInterpolantIdentity``

Configuration Examples
======================

CSP Configuration
-----------------

Crystal structure prediction with fixed composition:

.. code-block:: yaml

   model:
     sampler:
       class_path: omg.sampler.IndependentSampler
       init_args:
         species_distribution:
           class_path: omg.sampler.species_distributions.MirrorSpecies
         pos_distribution:
           class_path: omg.sampler.position_distributions.UniformPositionDistribution
         cell_distribution:
           class_path: omg.sampler.cell_distributions.InformedLatticeDistribution
           init_args:
             dataset_name: mp_20

DNG Configuration
-----------------

*De novo* generation with masked species:

.. code-block:: yaml

   model:
     sampler:
       class_path: omg.sampler.IndependentSampler
       init_args:
         species_distribution:
           class_path: omg.sampler.species_distributions.MaskSpecies
         pos_distribution:
           class_path: omg.sampler.position_distributions.UniformPositionDistribution
         cell_distribution:
           class_path: omg.sampler.cell_distributions.InformedLatticeDistribution
           init_args:
             dataset_name: mp_20

One-Sided SI Configuration
---------------------------

Using one-sided stochastic interpolants:

.. code-block:: yaml

   model:
     sampler:
       class_path: omg.sampler.IndependentSampler
       init_args:
         species_distribution:
           class_path: omg.sampler.species_distributions.MaskSpecies
         pos_distribution:
           class_path: omg.sampler.position_distributions.NormalPositionDistribution
         cell_distribution:
           class_path: omg.sampler.cell_distributions.NormalLatticeDistribution

Custom Distributions
====================

Create custom distributions by subclassing base classes:

Position Distribution
---------------------

.. code-block:: python

   from omg.sampler.abstracts import PositionDistribution
   import torch

   class MyPositionDistribution(PositionDistribution):
       def sample(self, data):
           n_atoms = data.n_atoms.sum()
           # Return (n_atoms, 3) tensor
           return torch.rand(n_atoms, 3)

Cell Distribution
-----------------

.. code-block:: python

   from omg.sampler.abstracts import CellDistribution
   import torch

   class MyCellDistribution(CellDistribution):
       def sample(self, data):
           batch_size = len(data.n_atoms)
           # Return (batch_size, 3, 3) tensor
           return torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)

Species Distribution
--------------------

.. code-block:: python

   from omg.sampler.abstracts import SpeciesDistribution
   import torch

   class MySpeciesDistribution(SpeciesDistribution):
       def sample(self, data):
           n_atoms = data.n_atoms.sum()
           # Return (n_atoms,) tensor of atomic numbers
           return torch.randint(1, 119, (n_atoms,))

Advanced Features
=================

Minimum Permutation Distance
-----------------------------

Ensure sampled positions are well-separated:

.. code-block:: python

   from omg.sampler.minimum_permutation_distance import (
       compute_minimum_permutation_distance
   )

   # Compute MPD between structures
   mpd = compute_minimum_permutation_distance(pos1, cell1, pos2, cell2)

This is useful for:

* Validating generated structures
* Computing structural similarity
* Filtering duplicate structures

Conditional Sampling
--------------------

Condition generation on properties:

.. code-block:: python

   # Future feature - conditioning on formation energy, band gap, etc.
   # Currently, use composition constraints for CSP

Best Practices
==============

1. **Match sampler to SI**: Ensure compatibility between base distribution and interpolant

2. **Use informed distributions**: ``InformedLatticeDistribution`` gives better initial guesses

3. **Dataset-specific sampling**: Use statistics from your training dataset:

   .. code-block:: yaml

      cell_distribution:
        class_path: omg.sampler.cell_distributions.InformedLatticeDistribution
        init_args:
          dataset_name: my_dataset  # Must match training data

4. **CSP best practices**:

   * Always use ``MirrorSpecies`` to keep composition fixed
   * Use ``UniformPositionDistribution`` for coordinates
   * Use ``InformedLatticeDistribution`` for realistic cells

5. **DNG best practices**:

   * Use ``MaskSpecies`` or ``UniformSpeciesDistribution``
   * Start with uniform position distribution
   * Use informed lattice distribution for physical relevance

Troubleshooting
===============

Incompatible Sampler and SI
----------------------------

**Error**: Generated structures have incorrect format

**Solution**: Check SI and sampler compatibility:

.. code-block:: yaml

   # This is WRONG:
   stochastic_interpolants:
     - SingleStochasticInterpolantIdentity  # Mirror required
   sampler:
     species_distribution:
       MaskSpecies  # Incompatible!

   # This is CORRECT:
   stochastic_interpolants:
     - SingleStochasticInterpolantIdentity
   sampler:
     species_distribution:
       MirrorSpecies  # Compatible

Poor Initial Structures
-----------------------

**Problem**: Slow convergence or low quality

**Solution**: Use informed distributions:

.. code-block:: yaml

   cell_distribution:
     class_path: omg.sampler.cell_distributions.InformedLatticeDistribution
     init_args:
       dataset_name: mp_20  # Use dataset statistics

Next Steps
==========

* :doc:`stochastic_interpolants` - Configure SI to match sampler
* :doc:`datamodule` - Prepare training data
* :doc:`training` - Train your model
* :doc:`../api/sampler` - Sampler API reference
* :doc:`../development/contributing` - Contributing guidelines
