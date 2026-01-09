========
Analysis
========

Analysis tools for validating and evaluating generated crystal structures.

Overview
========

OMatG provides comprehensive metrics for both generation modes:

* **CSP Metrics**: Match rates, RMSD, corrected RMSD
* **DNG Metrics**: Validity, diversity, coverage, Wasserstein distances
* **Validation**: Structural and compositional checks

Crystal Structure Prediction Metrics
=====================================

Match Rate
----------

Percentage of generated structures matching target structures using PyMatgen's ``StructureMatcher``:

.. code-block:: bash

   omg csp_metrics --config=config.yaml --xyz_file=generated.xyz

**Output** (``csp_metrics.json``):

.. code-block:: json

   {
       "match_rate": 0.85,
       "match_rate_valid": 0.92,
       "validity_rate": 0.93,
       "avg_rmsd": 0.12,
       "avg_crmse": 0.15
   }

Match-Everyone-to-Reference (METRe)
-----------------------------------

Compute best match for each reference structure:

.. code-block:: bash

   omg csp_metrics \
       --config=config.yaml \
       --xyz_file=generated.xyz \
       --metre=True

Finds best match from all generated structures for each target.

Root Mean Square Displacement (RMSD)
-------------------------------------

Average RMSD between matched structures:

.. math::

   \text{RMSD} = \sqrt{\frac{1}{N} \sum_{i=1}^N \| \mathbf{x}_i - \mathbf{x}'_i \|^2}

**Interpretation**:

* < 0.1 Å: Excellent match
* 0.1-0.3 Å: Good match
* > 0.3 Å: Poor match

Corrected RMSE (cRMSE)
----------------------

RMSD with penalty for non-matches:

.. code-block:: python

   cRMSE = RMSD if matched else stol

where ``stol`` is the site tolerance parameter (default: 0.3 Å).

**Purpose**: Avoids ignoring failed matches in averages.

Validation
----------

Structures are validated before matching:

* **Volume check**: Reasonable atomic volumes
* **Structure check**: Valid atomic positions
* **Composition check**: Expected elements
* **Fingerprint check**: Structural integrity

Skip validation (faster):

.. code-block:: bash

   omg csp_metrics \
       --xyz_file=generated.xyz \
       --skip_validation=True

Configuration
-------------

Customize ``StructureMatcher`` parameters:

.. code-block:: bash

   omg csp_metrics \
       --xyz_file=generated.xyz \
       --ltol=0.2 \           # Lattice tolerance
       --stol=0.3 \           # Site tolerance
       --angle_tol=5.0        # Angle tolerance (degrees)

*De Novo* Generation Metrics
=============================

Validity
--------

Percentage of structurally and compositionally valid structures:

.. code-block:: bash

   omg dng_metrics \
       --config=config.yaml \
       --xyz_file=generated.xyz \
       --dataset_name=mp_20

**Checks**:

* Valid atomic positions (no overlaps)
* Physically reasonable bond lengths
* Charge neutrality
* Positive lattice parameters
* Sensible density

Output:

.. code-block:: json

   {
       "validity_structural": 0.95,
       "validity_compositional": 0.88,
       "validity_overall": 0.85
   }

Coverage
--------

Percentage of test set compositions covered by generated structures:

.. math::

   \text{Coverage} = \frac{|\text{Generated Compositions} \cap \text{Test Compositions}|}{|\text{Test Compositions}|}

Recall and Precision:

* **Recall**: Coverage of test compositions
* **Precision**: Fraction of generated structures with test compositions

.. code-block:: json

   {
       "coverage_recall": 0.75,
       "coverage_precision": 0.68
   }

Diversity
---------

Measured via Wasserstein distances between distributions:

.. code-block:: json

   {
       "wasserstein_density": 0.15,
       "wasserstein_volume_fraction": 0.12,
       "wasserstein_num_atoms": 0.08,
       "wasserstein_num_elements": 0.05,
       "wasserstein_coordination": 0.18
   }

**Interpretation**: Lower is better (closer to training distribution)

Property Distributions
----------------------

Generated vs. training distributions:

* **Density**: Mass density (g/cm³)
* **Volume fraction**: Atomic packing fraction
* **Number of atoms**: Atoms per unit cell
* **Number of elements**: Unique element types
* **Coordination number**: Average atomic coordination

Validation Tools
================

ValidAtoms Class
----------------

Comprehensive structure validation:

.. code-block:: python

   from omg.analysis.valid_atoms import ValidAtoms

   validator = ValidAtoms()

   # Validate single structure
   is_valid = validator.validate_structure(
       species=atomic_numbers,
       positions=cartesian_coords,
       cell=lattice_matrix
   )

   # Get detailed results
   result = validator.validate_with_details(species, positions, cell)
   print(result['volume_check'])      # Pass/fail
   print(result['structure_check'])   # Pass/fail
   print(result['composition_check']) # Pass/fail

Validation Checks
-----------------

**Volume Check**:

.. code-block:: python

   valid = validator.check_volume(species, cell)

Ensures atoms have reasonable volumes (not too dense/sparse).

**Structure Check**:

.. code-block:: python

   valid = validator.check_structure(species, positions, cell)

Checks for:

* No overlapping atoms
* Reasonable bond lengths
* Valid coordination

**Composition Check**:

.. code-block:: python

   valid = validator.check_composition(species)

Verifies:

* Known element types (1 ≤ Z ≤ 118)
* Charge neutrality (for ionic compounds)
* Sensible stoichiometry

Visualization
=============

Distribution Plots
------------------

Compare generated to training data:

.. code-block:: bash

   omg visualize \
       --config=config.yaml \
       --xyz_file=generated.xyz \
       --plot_name=distributions.pdf

Creates plots for:

* Density distributions
* Volume distributions
* Atom count distributions
* Element distributions
* Lattice parameter distributions

RMSD Histogram
--------------

Automatically generated with CSP metrics:

.. code-block:: bash

   omg csp_metrics \
       --config=config.yaml \
       --xyz_file=generated.xyz
   # Creates rmsds.pdf

Shows distribution of RMSD values for matched structures.

Custom Analysis
===============

Analyzing Generated Structures
-------------------------------

.. code-block:: python

   from ase.io import read
   from pymatgen.core import Structure

   # Read generated structures
   atoms_list = read('generated.xyz', index=':')

   for i, atoms in enumerate(atoms_list):
       # Convert to pymatgen
       structure = Structure(
           lattice=atoms.get_cell(),
           species=atoms.get_chemical_symbols(),
           coords=atoms.get_scaled_positions()
       )

       # Analyze
       print(f"Structure {i}:")
       print(f"  Formula: {structure.composition.reduced_formula}")
       print(f"  Density: {structure.density:.2f} g/cm³")
       print(f"  Volume: {structure.volume:.2f} Ų")
       print(f"  Space group: {structure.get_space_group_info()}")

Computing Custom Properties
----------------------------

.. code-block:: python

   from pymatgen.analysis.local_env import CrystalNN

   # Compute coordination numbers
   nn = CrystalNN()
   for i, site in enumerate(structure):
       cn = nn.get_cn(structure, i)
       print(f"Site {i}: coordination = {cn}")

Structural Similarity
---------------------

.. code-block:: python

   from omg.sampler.minimum_permutation_distance import (
       compute_minimum_permutation_distance
   )

   # Compute similarity between structures
   mpd = compute_minimum_permutation_distance(
       pos1, cell1, pos2, cell2
   )
   print(f"Minimum Permutation Distance: {mpd:.3f}")

Batch Analysis
--------------

.. code-block:: python

   import pandas as pd
   from tqdm import tqdm

   results = []
   for atoms in tqdm(atoms_list):
       structure = ...  # Convert to pymatgen
       results.append({
           'formula': structure.composition.reduced_formula,
           'density': structure.density,
           'volume': structure.volume,
           'n_atoms': len(structure),
           'n_elements': len(set(structure.atomic_numbers)),
       })

   df = pd.DataFrame(results)
   print(df.describe())
   df.to_csv('analysis.csv', index=False)

Performance Metrics
===================

Parallelization
---------------

Analysis commands support parallel processing:

.. code-block:: bash

   omg csp_metrics \
       --xyz_file=generated.xyz \
       --number_cpus=16  # Use 16 CPU cores

Default: Uses all available CPUs (``os.cpu_count()``)

Processing Time
---------------

Typical times for 10,000 structures:

* **Validation**: ~5 minutes (16 CPUs)
* **CSP matching**: ~10 minutes (16 CPUs)
* **DNG metrics**: ~3 minutes (16 CPUs)
* **Visualization**: ~1 minute

Memory Usage
------------

* **Validation**: ~2 GB
* **CSP metrics**: ~4 GB
* **DNG metrics**: ~3 GB

Best Practices
==============

1. **Validate before metrics**: Check structural validity first:

   .. code-block:: bash

      omg csp_metrics --xyz_file=generated.xyz  # Includes validation

2. **Use parallelization**: Speed up analysis with multiple CPUs:

   .. code-block:: bash

      --number_cpus=16

3. **Save intermediate results**: Keep JSON outputs for later analysis

4. **Compare multiple runs**: Track metrics across experiments:

   .. code-block:: python

      import json
      results = []
      for exp in experiments:
          with open(f'{exp}/csp_metrics.json') as f:
              results.append(json.load(f))

5. **Visualize distributions**: Always generate distribution plots:

   .. code-block:: bash

      omg visualize --xyz_file=generated.xyz

6. **Check initial structures**: Compare ``generated.xyz`` and ``generated_init.xyz``

7. **Use appropriate metrics**: CSP for fixed composition, DNG for variable composition

Troubleshooting
===============

Low Match Rate
--------------

**Possible causes**:

* Insufficient integration steps
* Poor model training
* Mismatched sampler/SI configuration
* Wrong tolerance parameters

**Solutions**:

1. Increase integration steps:

   .. code-block:: yaml

      integration_time_steps: 2000

2. Train longer or with more data

3. Verify sampler matches SI type

4. Adjust tolerance:

   .. code-block:: bash

      omg csp_metrics --stol=0.5  # More lenient

Low Validity
------------

**Causes**:

* Model not converged
* Poor base distribution
* Too few integration steps

**Solutions**:

1. Train longer:

   .. code-block:: yaml

      trainer:
        max_epochs: 3000

2. Use informed base distributions:

   .. code-block:: yaml

      cell_distribution:
        class_path: omg.sampler.cell_distributions.InformedLatticeDistribution

3. Increase integration steps

Next Steps
==========

* :doc:`training` - Improve model performance
* :doc:`generation` - Generate structures
* :doc:`../api/analysis` - Analysis API reference
* :doc:`stochastic_interpolants` - Tune SI parameters
* :doc:`../development/contributing` - Contributing guidelines
