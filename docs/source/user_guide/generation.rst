==========
Generation
==========

This guide covers generating crystal structures with trained OMatG models.

Overview
========

OMatG can generate structures in two modes:

1. **Crystal Structure Prediction (CSP)**: Generate structures for given compositions
2. ***De Novo* Generation (DNG)**: Generate novel structures without composition constraints

Basic Generation
================

From Checkpoint
---------------

.. code-block:: bash

   omg predict \
       --config=config.yaml \
       --ckpt_path=best_model.ckpt \
       --model.generation_xyz_filename=generated.xyz

This generates:

* ``generated.xyz``: Final generated structures
* ``generated_init.xyz``: Initial random structures (before integration)

Override Config
---------------

.. code-block:: bash

   omg predict \
       --config=config.yaml \
       --ckpt_path=best_model.ckpt \
       --model.generation_xyz_filename=generated.xyz \
       --data.batch_size=64 \
       --model.si.init_args.integration_time_steps=2000

Crystal Structure Prediction
=============================

Predict structures for specific compositions:

Step 1: Create Composition File
--------------------------------

Single composition:

.. code-block:: bash

   omg create_compositions \
       --config=config.yaml \
       --compositions='LiMn2O4' \
       --lmdb_file=targets.lmdb

Multiple compositions:

.. code-block:: bash

   omg create_compositions \
       --config=config.yaml \
       --compositions='[LiMn2O4, GaTe, NaCl, CuO]' \
       --lmdb_file=targets.lmdb

Multiple structures per composition:

.. code-block:: bash

   omg create_compositions \
       --config=config.yaml \
       --compositions='LiMn2O4' \
       --lmdb_file=targets.lmdb \
       --repeats=10  # Generate 10 structures for this composition

Step 2: Generate Structures
----------------------------

.. code-block:: bash

   omg predict \
       --config=csp_config.yaml \
       --ckpt_path=csp_model.ckpt \
       --data.pred_dataset.init_args.file_path=targets.lmdb \
       --model.generation_xyz_filename=predicted.xyz

Requirements:

* Use CSP model (species fixed with ``SingleStochasticInterpolantIdentity``)
* Model trained on datasets containing the target elements
* Compatible sampler (``MirrorSpecies`` for species)

*De Novo* Generation
====================

Generate novel structures:

From Test Set
-------------

Generate one structure per test sample:

.. code-block:: bash

   omg predict \
       --config=dng_config.yaml \
       --ckpt_path=dng_model.ckpt \
       --model.generation_xyz_filename=novel_structures.xyz

The prediction dataset in ``dng_config.yaml`` determines how many structures are generated.

Custom Number of Structures
----------------------------

Create dummy dataset:

.. code-block:: bash

   omg create_compositions \
       --config=dng_config.yaml \
       --compositions='[H1]' \  # Dummy composition (will be replaced)
       --lmdb_file=dummy.lmdb \
       --repeats=1000  # Generate 1000 structures

Then generate:

.. code-block:: bash

   omg predict \
       --config=dng_config.yaml \
       --ckpt_path=dng_model.ckpt \
       --data.pred_dataset.init_args.file_path=dummy.lmdb \
       --model.generation_xyz_filename=novel_structures.xyz

Generation Parameters
=====================

Integration Steps
-----------------

Control generation quality:

.. code-block:: bash

   omg predict \
       --config=config.yaml \
       --ckpt_path=model.ckpt \
       --model.generation_xyz_filename=output.xyz \
       --model.si.init_args.integration_time_steps=2000

**Trade-off**:

* More steps = higher quality, slower
* Fewer steps = lower quality, faster

**Recommendations**:

* Fast preview: 500 steps
* Default: 1000 steps
* High quality: 2000-5000 steps

Batch Size
----------

Generate multiple structures simultaneously:

.. code-block:: bash

   omg predict \
       --config=config.yaml \
       --data.batch_size=128

**Considerations**:

* Larger batch = faster total time
* Limited by GPU memory
* No effect on quality

Random Seed
-----------

For reproducible generation:

.. code-block:: bash

   omg predict \
       --config=config.yaml \
       --ckpt_path=model.ckpt \
       --model.generation_xyz_filename=output.xyz \
       --seed_everything=42

Differential Equation
---------------------

Choose between ODE (deterministic) and SDE (stochastic):

.. code-block:: yaml

   # In config.yaml
   model:
     si:
       init_args:
         stochastic_interpolants:
           - init_args:
               differential_equation_type: "ODE"  # Deterministic (default)
               # differential_equation_type: "SDE"  # Stochastic

**SDE vs ODE**:

* ODE: Same seed → same output
* SDE: More diverse outputs
* ODE: Generally faster

Output Formats
==============

XYZ Format
----------

Default output format:

.. code-block:: bash

   # Structure with comment line
   32
   Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3
   Li 0.0 0.0 0.0
   ...

CIF Format
----------

Convert to CIF:

.. code-block:: python

   from ase.io import read, write

   atoms = read('generated.xyz', index=':')
   for i, a in enumerate(atoms):
       write(f'structure_{i}.cif', a)

Pymatgen Structures
-------------------

Convert to Pymatgen:

.. code-block:: python

   from ase.io import read
   from pymatgen.io.ase import AseAtomsAdaptor

   atoms_list = read('generated.xyz', index=':')
   adaptor = AseAtomsAdaptor()

   structures = [adaptor.get_structure(a) for a in atoms_list]

   # Save as CIF
   for i, s in enumerate(structures):
       s.to(filename=f'structure_{i}.cif')

Batch Generation
================

Generate Large Datasets
-----------------------

Script for generating many structures:

.. code-block:: python

   #!/usr/bin/env python
   """Generate large batches of structures."""

   import subprocess
   import sys

   # Configuration
   config = "dng_config.yaml"
   checkpoint = "dng_model.ckpt"
   batch_size = 1000
   num_batches = 10

   for i in range(num_batches):
       output_file = f"batch_{i:03d}.xyz"

       cmd = [
           "omg", "predict",
           f"--config={config}",
           f"--ckpt_path={checkpoint}",
           f"--model.generation_xyz_filename={output_file}",
           f"--data.batch_size={batch_size}",
           f"--seed_everything={i}",  # Different seed per batch
       ]

       print(f"Generating batch {i+1}/{num_batches}...")
       subprocess.run(cmd, check=True)

   print(f"Generated {num_batches * batch_size} structures total")

Parallel Generation
-------------------

Generate on multiple GPUs:

.. code-block:: bash

   # GPU 0
   CUDA_VISIBLE_DEVICES=0 omg predict \
       --config=config.yaml \
       --ckpt_path=model.ckpt \
       --model.generation_xyz_filename=batch_0.xyz \
       --seed_everything=0 &

   # GPU 1
   CUDA_VISIBLE_DEVICES=1 omg predict \
       --config=config.yaml \
       --ckpt_path=model.ckpt \
       --model.generation_xyz_filename=batch_1.xyz \
       --seed_everything=1 &

   wait

Post-Processing
===============

Filtering Structures
--------------------

Remove invalid structures:

.. code-block:: python

   from ase.io import read, write
   from omg.analysis.valid_atoms import ValidAtoms

   validator = ValidAtoms()
   atoms_list = read('generated.xyz', index=':')

   valid_atoms = []
   for atoms in atoms_list:
       species = atoms.get_atomic_numbers()
       positions = atoms.get_positions()
       cell = atoms.get_cell().array

       if validator.validate_structure(species, positions, cell):
           valid_atoms.append(atoms)

   write('valid_structures.xyz', valid_atoms)
   print(f"Kept {len(valid_atoms)}/{len(atoms_list)} structures")

Deduplication
-------------

Remove duplicate structures:

.. code-block:: python

   from pymatgen.analysis.structure_matcher import StructureMatcher
   from pymatgen.io.ase import AseAtomsAdaptor

   atoms_list = read('generated.xyz', index=':')
   adaptor = AseAtomsAdaptor()
   structures = [adaptor.get_structure(a) for a in atoms_list]

   matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)

   unique_structures = []
   for s in structures:
       is_duplicate = False
       for u in unique_structures:
           if matcher.fit(s, u):
               is_duplicate = True
               break

       if not is_duplicate:
           unique_structures.append(s)

   print(f"Unique: {len(unique_structures)}/{len(structures)}")

Relaxation
----------

Relax generated structures with ASE:

.. code-block:: python

   from ase.io import read, write
   from ase.optimize import BFGS
   from ase.calculators.emt import EMT  # Replace with real calculator

   atoms_list = read('generated.xyz', index=':')
   relaxed = []

   for i, atoms in enumerate(atoms_list):
       atoms.calc = EMT()  # Use VASP, GPAW, etc. in practice
       opt = BFGS(atoms, trajectory=f'opt_{i}.traj')
       opt.run(fmax=0.05)
       relaxed.append(atoms)

   write('relaxed_structures.xyz', relaxed)

Analysis
========

Immediate Evaluation
--------------------

Evaluate right after generation:

.. code-block:: bash

   # Generate
   omg predict \
       --config=config.yaml \
       --ckpt_path=model.ckpt \
       --model.generation_xyz_filename=generated.xyz

   # Evaluate CSP
   omg csp_metrics \
       --config=config.yaml \
       --xyz_file=generated.xyz

   # Or evaluate DNG
   omg dng_metrics \
       --config=config.yaml \
       --xyz_file=generated.xyz \
       --dataset_name=mp_20

Visualization
-------------

Compare distributions:

.. code-block:: bash

   omg visualize \
       --config=config.yaml \
       --xyz_file=generated.xyz \
       --plot_name=distributions.pdf

Tips & Best Practices
======================

1. **Start with defaults**: Use 1000 integration steps

2. **Validate first**: Check a small batch before generating many

3. **Save initial structures**: Keep ``*_init.xyz`` for debugging

4. **Use appropriate batch size**: Balance speed and memory

5. **Set random seeds**: For reproducibility

6. **Monitor GPU memory**: Adjust batch size if needed

7. **Filter invalid structures**: Post-process with validation

8. **Deduplicate**: Remove repeated structures

9. **Relax with DFT**: For property predictions

10. **Generate extras**: Plan for invalid/duplicate structures

Troubleshooting
===============

Low Quality Structures
----------------------

**Solutions**:

1. Increase integration steps:

   .. code-block:: bash

      --model.si.init_args.integration_time_steps=2000

2. Check model training convergence

3. Verify sampler matches SI type

4. Use correct checkpoint for task (CSP vs DNG)

Out of Memory
-------------

**Solutions**:

1. Reduce batch size:

   .. code-block:: bash

      --data.batch_size=16

2. Use mixed precision:

   .. code-block:: yaml

      trainer:
        precision: "16-mixed"

3. Generate in smaller batches

Wrong Compositions (CSP)
------------------------

**Problem**: Generated structures have different composition than target

**Solutions**:

1. Verify CSP config:

   .. code-block:: yaml

      stochastic_interpolants:
        - SingleStochasticInterpolantIdentity  # For species

2. Check sampler:

   .. code-block:: yaml

      species_distribution:
        MirrorSpecies  # Not MaskSpecies

3. Use CSP checkpoint, not DNG checkpoint

Slow Generation
---------------

**Solutions**:

1. Reduce integration steps:

   .. code-block:: bash

      --model.si.init_args.integration_time_steps=500

2. Increase batch size:

   .. code-block:: bash

      --data.batch_size=128

3. Use ODE instead of SDE:

   .. code-block:: yaml

      differential_equation_type: "ODE"

Advanced Topics
===============

Conditional Generation
----------------------

Future feature - condition on properties:

.. code-block:: python

   # Planned feature
   omg predict \
       --config=config.yaml \
       --ckpt_path=model.ckpt \
       --model.generation_xyz_filename=output.xyz \
       --condition_on_property=band_gap \
       --property_value=2.0

Trajectory Saving
-----------------

Save intermediate structures during generation:

.. code-block:: python

   # Modify generation code to save trajectory
   # See omg_lightning.py for implementation details

Next Steps
==========

* :doc:`analysis` - Evaluate generated structures
* :doc:`../getting_started/quickstart` - Quick start guide
* :doc:`stochastic_interpolants` - Tune SI parameters
* :doc:`../api/training` - API reference
