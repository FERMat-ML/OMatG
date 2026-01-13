*************************************
How to Create Configuration Files
*************************************

Machine-learning models implemented with PyTorch Lightning rely on three essential parts:

1. **Trainer**: The training engine.
2. **LightningDataModule**: Handles data loading and preprocessing.
3. **LightningModule**: Defines the model and training logic.

Configuration files of OMatG thus generally contain specifications for these three parts.

Trainer
=======

OMatG uses the standard `PyTorch Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`__. Its
parameters are specified in the ``trainer`` section of the configuration file, for example:

.. code-block:: yaml

    trainer:
      callbacks:  # List of callbacks to be used during training.
        - class_path: lightning.pytorch.callbacks.ModelCheckpoint
          init_args:
            filename: "best_val_loss_total"
            save_top_k: 1
            monitor: "val_loss_total"
            save_weights_only: true
      accelerator: "gpu"
      gradient_clip_val: 0.5
      gradient_clip_algorithm: "value"
      num_sanity_val_steps: 0
      precision: "32-true"
      max_epochs: 2000
      enable_progress_bar: true

Note that it is possible to initialize specialized classes in the configuration file by specifying the ``class_path`` and
``init_args``. The ``init_args`` dictionary contains the arguments that are passed to the constructor of the class.

In addition to the trainer, one should specify the optimizer and (optionally) the learning rate scheduler in their own
sections:

.. code-block:: yaml

    optimizer:
      class_path: torch.optim.AdamW
      init_args:
        lr: 0.001
        weight_decay: 0.01
    lr_scheduler:
      class_path: torch.optim.lr_scheduler.CosineAnnealingLR
      init_args:
        T_max: 2000
        eta_min: 1e-07

LightningDataModule
===================

The ``data`` section of the configuration constructs the ``OMGDataModule`` (see
`omg/datamodule/omg_datamodule.py <https://github.com/FERMat-ML/OMatG/blob/main/omg/datamodule/omg_datamodule.py>`__). It mainly expects the
``train_dataset``, ``val_dataset``, and ``pred_dataset`` sections. Each of these sections should construct an
``StructureDataset`` (see `omg/datamodule/structure_dataset.py <https://github.com/FERMat-ML/OMatG/blob/main/omg/datamodule/structure_dataset.py>`__). This can be
done based on `LMDB <https://lmdb.readthedocs.io/en/release/>`__ or CSV files:

.. code-block:: yaml

    data:
      train_dataset:
        class_path: omg.datamodule.StructureDataset
        init_args:
          file_path: "data/mp_20/train.lmdb"
          lazy_storage: True
          niggli_reduce: False  # Apply ASE's Niggli reduction to all structures.
      val_dataset:
        class_path: omg.datamodule.StructureDataset
        init_args:
          file_path: "data/mp_20/val.lmdb"
          lazy_storage: True
          niggli_reduce: False  # Apply ASE's Niggli reduction to all structures.
      pred_dataset:
        class_path: omg.datamodule.StructureDataset
        init_args:
          file_path: "data/mp_20/test.lmdb"
          lazy_storage: True
          niggli_reduce: False  # Apply ASE's Niggli reduction to all structures.
      batch_size: 32
      num_workers: 4
      pin_memory: True
      persistent_workers: True

Dataset File Format
-------------------

Every record in the LMDB files should contain a crystal structure. The key of each record is assumed to be an
(arbitrary) encoded string, while the value is assumed to be a pickled dictionary with, at least, the following keys:

- ``pos``: A ``torch.Tensor`` of shape ``(N, 3)`` containing the Cartesian coordinates of the atoms in the crystal structure.
- ``cell``: A ``torch.Tensor`` of shape ``(3, 3)`` containing the lattice vectors of the crystal structure.
- ``atomic_numbers``: A ``torch.Tensor`` of shape ``(N,)`` containing the atomic numbers of the atoms in the crystal structure.

CSV files should contain a ``cif`` column with the CIF representation of the structures. This will be used to infer the
cell, atomic numbers, and positions of the structures.

For large datasets, it is possible to read the structures lazily from disk by setting ``lazy_storage: True`` in the
``init_args`` of the respective dataset section. In this case, the structures are only read from disk when they are
accessed. CSV files are converted to lmdb files in this case for faster access.

The ``data`` section can also contain additional parameters for the data loading (such as ``batch_size``, ``num_workers``,
``pin_memory``, and ``persistent_workers`` in the above example). These parameters are passed to the underlying
`PyTorch DataLoader <https://docs.pytorch.org/docs/stable/data.html>`__ instances.

Data Format in OMatG
--------------------

Within OMatG, the data is passed around as ``torch_geometric.data.Data`` instances. For a batch size of ``batch_size``,
these instances contain the following attributes:

- ``n_atoms``: ``torch.Tensor`` of shape ``(batch_size, )`` containing the number of atoms in each configuration.
- ``batch``: ``torch.Tensor`` of shape ``(sum(n_atoms),)`` containing the index of the configuration to which each atom belongs.
- ``species``: ``torch.Tensor`` of shape ``(sum(n_atoms),)`` containing the atomic numbers of the atoms in the configurations.
- ``pos``: ``torch.Tensor`` of shape ``(sum(n_atoms), 3)`` containing the atomic positions of the atoms in the configurations.
- ``cell``: ``torch.Tensor`` of shape ``(batch_size, 3, 3)`` containing the cell vectors of the configurations.
- ``ptr``: ``torch.Tensor`` of shape ``(batch_size + 1,)`` containing the indices of the first atom of each configuration in the ``species`` and ``pos`` tensors.
- ``property``: dict containing the properties of the configurations.

LightningModule
===============

The ``model`` section of the configuration file constructs the ``OMGLightningModule`` (see
`omg/omg_lightning.py <https://github.com/FERMat-ML/OMatG/blob/main/omg/omg_lightning.py>`__). Its arguments are documented in the class docstring.
An example ``model`` section looks as follows:

.. code-block:: yaml

    model:
      si:  # Collection of stochastic interpolants.
        class_path: omg.si.stochastic_interpolants.StochasticInterpolants
        init_args:
          stochastic_interpolants:
            # Chemical species.
            # The SingleStochasticInterpolantIdentity keeps the species unchanged during interpolation (CSP task).
            # For DNG, use, e.g., omg.si.discrete_flow_matching_mask.DiscreteFlowMatchingMask.
            - class_path: omg.si.single_stochastic_interpolant_identity.SingleStochasticInterpolantIdentity
            # Fractional coordinates.
            - class_path: omg.si.single_stochastic_interpolant.SingleStochasticInterpolant
              init_args:
                # Use a periodic interpolant for fractional coordinates.
                interpolant: omg.si.interpolants.PeriodicLinearInterpolant
                gamma: null
                epsilon: null
                differential_equation_type: "ODE"
                integrator_kwargs:
                  method: "euler"
                velocity_annealing_factor: 10.182659004291072
                correct_center_of_mass_motion: true
            # Lattice vectors.
            - class_path: omg.si.single_stochastic_interpolant.SingleStochasticInterpolant
              init_args:
                # Use a non-periodic interpolant for lattice vectors.
                interpolant: omg.si.interpolants.LinearInterpolant
                gamma: null
                epsilon: null
                differential_equation_type: "ODE"
                integrator_kwargs:
                  method: "euler"
                velocity_annealing_factor: 1.824475401606087
                correct_center_of_mass_motion: false
          data_fields:
            # If the order of the data_fields changes,
            # the order of the above StochasticInterpolant inputs must also change.
            - "species"
            - "pos"
            - "cell"
          integration_time_steps: 1000
      relative_si_costs:
        species_loss: 0.0
        pos_loss_b: 0.999
        cell_loss_b: 0.001
      sampler:
        class_path: omg.sampler.IndependentSampler
        init_args:
          pos_distribution:
            class_path: omg.sampler.position_distributions.UniformPositionDistribution
          cell_distribution:
            class_path: omg.sampler.cell_distributions.InformedLatticeDistribution
            init_args:
              dataset_name: mp_20
          species_distribution:
            class_path: omg.sampler.species_distributions.MirrorSpecies
      model:
        class_path: omg.model.model.Model
        init_args:
          encoder:
            class_path: omg.model.encoders.cspnet_full.CSPNetFull
          head:
            class_path: omg.model.heads.pass_through.PassThrough
          time_embedder:
            class_path: omg.model.model_utils.SinusoidalTimeEmbeddings
            init_args:
              dim: 256

Stochastic Interpolants Section
--------------------------------

The ``si`` section combines the stochastic interpolants for the ``species``, ``pos``, and ``cell`` data fields of the crystal
structures in the `StochasticInterpolants <https://github.com/FERMat-ML/OMatG/blob/main/omg/si/stochastic_interpolants.py>`__ class.
This class is documented in its docstring but, in a nutshell, it is a container for multiple
`StochasticInterpolant <https://github.com/FERMat-ML/OMatG/blob/main/omg/si/abstracts.py>`__ instances. The typically used implementations of this abstract class are:

- `SingleStochasticInterpolant <https://github.com/FERMat-ML/OMatG/blob/main/omg/si/single_stochastic_interpolant.py>`__: For continuous data fields such as fractional coordinates and lattice vectors with arbitrary base distributions. The specific interpolant and its parameters are specified on initialization of this class. Every interpolant has a periodic (for fractional coordinates) and a non-periodic (for lattice vectors) version.
- `SingleStochasticInterpolantOS <https://github.com/FERMat-ML/OMatG/blob/main/omg/si/single_stochastic_interpolant_os.py>`__: For continuous data fields such as fractional coordinates and lattice vectors, but explicitly assuming a Gaussian base distribution as it implements one-sided stochastic interpolants.
- `SingleStochasticInterpolantIdentity <https://github.com/FERMat-ML/OMatG/blob/main/omg/si/single_stochastic_interpolant_identity.py>`__: For keeping the corresponding data field unchanged during interpolation and generation.
- `DiscreteFlowMatchingMask <https://github.com/FERMat-ML/OMatG/blob/main/omg/si/discrete_flow_matching_mask.py>`__: For discrete data fields such as atomic species with a completely masked base distribution.
- `DiscreteFlowMatchingUniform <https://github.com/FERMat-ML/OMatG/blob/main/omg/si/discrete_flow_matching_uniform.py>`__: For discrete data fields such as atomic species with a uniform base distribution.

Loss Weighting
--------------

Every ``StochasticInterpolant`` in the ``StochasticInterpolants`` class computes losses and returns them in a
dictionary (see the ``loss_keys`` method in the respective class). The ``StochasticInterpolants`` class prefixes these keys
with the name of the corresponding data field so that the losses can be identified. The ``relative_si_costs`` section
specifies the relative weights of these losses when they are added up during training.

Sampler Section
---------------

The ``sampler`` section specifies the base distributions for the positions, lattice vectors, and atomic species. Depending
on the choice of the stochastic interpolant, one should choose the matching base distribution:

- `SingleStochasticInterpolant <https://github.com/FERMat-ML/OMatG/blob/main/omg/si/single_stochastic_interpolant.py>`__: The choice of the base distribution is arbitrary. As in the example above, we typically use a uniform distribution for the fractional coordinates and an informed base distribution for the lattice vectors.
- `SingleStochasticInterpolantOS <https://github.com/FERMat-ML/OMatG/blob/main/omg/si/single_stochastic_interpolant_os.py>`__: Explicitly assumes a normal base distribution.
- `SingleStochasticInterpolantIdentity <https://github.com/FERMat-ML/OMatG/blob/main/omg/si/single_stochastic_interpolant_identity.py>`__: Explicitly assumes that the training data is just taken over in the "random" sample as implemented by the mirror distributions.
- `DiscreteFlowMatchingMask <https://github.com/FERMat-ML/OMatG/blob/main/omg/si/discrete_flow_matching_mask.py>`__: Explicitly assumes fully masked samples as the base distribution as implemented in the `MaskSpeciesDistribution <https://github.com/FERMat-ML/OMatG/blob/main/omg/sampler/species_distributions.py>`__.
- `DiscreteFlowMatchingUniform <https://github.com/FERMat-ML/OMatG/blob/main/omg/si/discrete_flow_matching_uniform.py>`__: Explicitly assumes uniformly distributed atomic species as the base distribution as implemented in the `UniformSpeciesDistribution <https://github.com/FERMat-ML/OMatG/blob/main/omg/sampler/species_distributions.py>`__.

Model Architecture
------------------

The ``model`` section specifies the model architecture. In the above example, we use DiffCSPNet.
