========
Analysis
========

The ``omg.analysis`` module provides tools for validating and evaluating generated crystal structures.

Module Overview
===============

The analysis module includes:

* Structure validation (volume, structure, composition checks)
* CSP metrics (match rate, RMSD, corrected RMSE)
* DNG metrics (validity, coverage, diversity)
* Analysis utilities

Main Classes
============

.. currentmodule:: omg.analysis

.. autosummary::
   :toctree: generated/
   :nosignatures:

   valid_atoms.ValidAtoms

Detailed Documentation
======================

ValidAtoms
----------

.. autoclass:: omg.analysis.valid_atoms.ValidAtoms
   :members:
   :undoc-members:
   :show-inheritance:

   Comprehensive structure validation.

   Performs multiple checks to determine if a crystal structure is physically valid:

   * **Volume check**: Atoms have reasonable volumes (not too dense/sparse)
   * **Structure check**: No overlapping atoms, valid bond lengths
   * **Composition check**: Known elements, charge neutrality

   **Example**:

   .. code-block:: python

      from omg.analysis.valid_atoms import ValidAtoms
      import torch

      validator = ValidAtoms()

      # Simple validation
      is_valid = validator.validate_structure(
          species=torch.tensor([8, 8, 1, 1]),  # O, O, H, H
          positions=torch.tensor([[0, 0, 0], [3, 0, 0], [1, 0, 0], [2, 0, 0]]),
          cell=torch.eye(3) * 10,
      )
      print(f"Valid: {is_valid}")

      # Detailed validation
      result = validator.validate_with_details(species, positions, cell)
      print(f"Volume check: {result['volume_check']}")
      print(f"Structure check: {result['structure_check']}")
      print(f"Composition check: {result['composition_check']}")

   **Methods**:

   * ``validate_structure(species, positions, cell)``: Quick validation
   * ``validate_with_details(species, positions, cell)``: Detailed results
   * ``check_volume(species, cell)``: Volume check only
   * ``check_structure(species, positions, cell)``: Structure check only
   * ``check_composition(species)``: Composition check only

Validation Checks
=================

Volume Check
------------

Ensures atoms have physically reasonable volumes:

.. code-block:: python

   validator = ValidAtoms()
   valid = validator.check_volume(
       species=torch.tensor([6, 6, 6]),  # Carbon atoms
       cell=torch.eye(3) * 5.0,
   )

**Criteria**:

* Minimum volume per atom: ~5 ų/atom
* Maximum volume per atom: ~100 ų/atom
* Varies by element type

Structure Check
---------------

Validates atomic positions and bonding:

.. code-block:: python

   valid = validator.check_structure(
       species=torch.tensor([8, 1, 1]),  # H2O
       positions=torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
       cell=torch.eye(3) * 10,
   )

**Criteria**:

* No overlapping atoms (distance > ~0.5 Å)
* Bond lengths within reasonable ranges
* Valid coordination numbers

Composition Check
-----------------

Verifies chemical composition:

.. code-block:: python

   valid = validator.check_composition(
       species=torch.tensor([8, 1, 1])  # H2O
   )

**Criteria**:

* Known element types (1 ≤ Z ≤ 118)
* Reasonable stoichiometry
* Charge neutrality (for ionic compounds)

Metrics
=======

CSP Metrics
-----------

Computed by :meth:`omg.omg_trainer.OMGTrainer.csp_metrics`:

.. code-block:: bash

   omg csp_metrics --config=config.yaml --xyz_file=generated.xyz

**Outputs**:

* ``match_rate``: Percentage of structures matching targets
* ``match_rate_valid``: Match rate among valid structures
* ``validity_rate``: Percentage of valid structures
* ``avg_rmsd``: Average RMSD for matched structures
* ``avg_crmse``: Corrected RMSE with penalty for non-matches

DNG Metrics
-----------

Computed by :meth:`omg.omg_trainer.OMGTrainer.dng_metrics`:

.. code-block:: bash

   omg dng_metrics --config=config.yaml --xyz_file=generated.xyz --dataset_name=mp_20

**Outputs**:

* ``validity_structural``: Structural validity rate
* ``validity_compositional``: Compositional validity rate
* ``validity_overall``: Overall validity rate
* ``coverage_recall``: Coverage of test compositions
* ``coverage_precision``: Precision of generated compositions
* ``wasserstein_*``: Wasserstein distances for various properties

Batch Validation
================

Validate Multiple Structures
-----------------------------

.. code-block:: python

   from ase.io import read
   from omg.analysis.valid_atoms import ValidAtoms

   validator = ValidAtoms()
   atoms_list = read('generated.xyz', index=':')

   valid_count = 0
   for atoms in atoms_list:
       species = torch.tensor(atoms.get_atomic_numbers())
       positions = torch.tensor(atoms.get_positions())
       cell = torch.tensor(atoms.get_cell().array)

       if validator.validate_structure(species, positions, cell):
           valid_count += 1

   validity_rate = valid_count / len(atoms_list)
   print(f"Validity: {validity_rate:.2%}")

Parallel Validation
-------------------

The CLI commands support parallel processing:

.. code-block:: bash

   omg csp_metrics \
       --xyz_file=generated.xyz \
       --number_cpus=16  # Use 16 cores

Performance
===========

Typical validation times (16 CPUs):

* 1,000 structures: ~30 seconds
* 10,000 structures: ~5 minutes
* 100,000 structures: ~50 minutes

Memory usage:

* Validation: ~2 GB
* CSP metrics: ~4 GB
* DNG metrics: ~3 GB

Custom Analysis
===============

Computing Properties
--------------------

.. code-block:: python

   from pymatgen.io.ase import AseAtomsAdaptor
   from pymatgen.analysis.local_env import CrystalNN

   adaptor = AseAtomsAdaptor()
   structure = adaptor.get_structure(atoms)

   # Density
   density = structure.density
   print(f"Density: {density:.2f} g/cm³")

   # Coordination numbers
   nn = CrystalNN()
   cns = [nn.get_cn(structure, i) for i in range(len(structure))]
   avg_cn = sum(cns) / len(cns)
   print(f"Average CN: {avg_cn:.2f}")

   # Space group
   spg = structure.get_space_group_info()
   print(f"Space group: {spg}")

Structural Similarity
---------------------

.. code-block:: python

   from omg.sampler.minimum_permutation_distance import (
       compute_minimum_permutation_distance
   )

   # Compare two structures
   mpd = compute_minimum_permutation_distance(
       pos1=positions1,
       cell1=cell1,
       pos2=positions2,
       cell2=cell2,
   )
   print(f"MPD: {mpd:.3f}")

Distribution Analysis
---------------------

.. code-block:: python

   import pandas as pd
   from ase.io import read

   atoms_list = read('generated.xyz', index=':')

   data = []
   for atoms in atoms_list:
       data.append({
           'n_atoms': len(atoms),
           'volume': atoms.get_volume(),
           'density': len(atoms) / atoms.get_volume(),
       })

   df = pd.DataFrame(data)
   print(df.describe())

   # Plot distributions
   import matplotlib.pyplot as plt
   df.hist(bins=50, figsize=(12, 4))
   plt.tight_layout()
   plt.savefig('distributions.pdf')

See Also
========

* :doc:`../user_guide/analysis` - User guide for analysis
* :doc:`../getting_started/quickstart` - Quick start with metrics
* :doc:`training` - Training and evaluation workflow
