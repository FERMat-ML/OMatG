# OMatG: Open Materials Generation

A generative model for crystal structure prediction and de novo generation of inorganic crystals. 

This open-source framework accompanies the paper available at [OpenReview](https://openreview.net/forum?id=ka2jxtCrkK#discussion),
which should be cited when using it.

**Note:** This repository is currently work in progress. 

## Dependencies

You can use any Python version between 3.10 and 3.12.

Install the dependencies and the `omg` package itself via pip. 

To install `omg` in editable mode (recommended for developers), use `pip install -e .` within the base directory of this 
repository once. Any changes in code are directly available in the installed `omg` package.

To install `omg` as a package, use `pip install .` instead. If the code in this repository changes, this command has to 
executed again to also change the code of the installed package.

## Training

Run the following command in any directory to train from scratch based on the configuration file `config.yaml`:

```bash
omg fit --config config.yaml --trainer.accelerator=gpu
```

This command will create checkpoints, log files, and cache files in the working directory.

If you want to include a Wandb logger with a name, add the `--trainer.logger=WandbLogger --trainer.logger.name=<name>` 
argument. Other loggers can be found [here](https://lightning.ai/docs/pytorch/stable/extensions/logging.html).

In order to restart training from a checkpoint, add the `--ckpt_path=<checkpoint_file.ckpt>` argument. 

In order to seed the random number generators before training, use `--seed_everything=<seed>`.

Exemplary configuration files can be found in the `omg/conf_examples` directory.

The training command can be executed in any directory. The configuration files contain paths to lmbd data files that are 
used, e.g., for training. The path to these data files can either be relative to the working directory, or relative to 
the `omg` directory (that is, use `"data/mp_20/val.lmdb"` for `lmdb_paths` in order to use the `mp_20` data set as in 
exemplary configuration files).

## Sampling

For generating new structures in an xyz file, run the following command:

```bash
omg predict --config {config_file} --ckpt_path=<checkpoint_file.ckpt> --model.generation_xyz_filename=<xyz_file> --data.batch_size=1024 --seed_everything=42 --trainer.max_epochs=1
```

For an xyz filename `filename.xyz`, this command will also create a file `filename_init.xyz` that contains the initial
structures that were integrated to yield the structures in `filename.xyz`. This file is required for the visualization
below.

## Visualize

Run the following command to compare distributions over the generated structures in an xyz file to distributions over 
training dataset:

```bash
omg visualize --config {config_file} --xyz_file {xyz_file} --plot_name {plot_name}
```

## Match Rate (CSP)

Run the following command to compute the match rate between the generated structures in an xyz file and the structures 
in the prediction dataset, as well as the rate of unique structures in the generated structures:

```bash
omg match --config {config_file} --xyz_file {xyz_file}
```

Computing the match rate as in DiffCSP or FlowMM requires to validate every structure which is quite slow. Also, 
computing the unique rate is slow. One can use the `--skip_validation=true` and `--skip_unique=true` arguments to skip
these computations. 

The validations, and the computations of the match rate and unique rate are parallelized. The number of processes is 
determined by `os.cpu_count()`. This can be changed by setting the `--number_cpus` argument (which is probably most 
useful in cluster environments).

## Alex-MP20 dataset

The Alex-MP20 dataset is too large to be stored in this repository. It will be released later.

## Curriculum Learning

Run the following command to use a given configuration file with stochastic interpolants for all datafields 'pos', 
'species', and 'cell' to generate a new configuration file where only some of the stochastic interpolants are used, 
while the others are replaced by identity interpolants:

```bash
omg curriculum --config {config_file} --lessons {lessons}
```

Here, lessons should be a list of strings so, e.g., [pos] or [pos,species] (spaces can be included when the brackets are
surrounded by quotes).

## OMatG Data Format

For a batch size of batch_size, the `torch_geometric.data.Data` instances contain the following attributes:
- `n_atoms`: `torch.Tensor` of shape `(batch_size, )` containing the number of atoms in each configuration.
- `batch`: `torch.Tensor` of shape `(sum(n_atoms),)` containing the index of the configuration to which each atom 
belongs.
- `species`: `torch.Tensor` of shape `(sum(n_atoms),)` containing the atomic numbers of the atoms in the configurations.
- `pos`: `torch.Tensor` of shape `(sum(n_atoms), 3)` containing the atomic positions of the atoms in the configurations.
- `cell`: `torch.Tensor` of shape `(batch_size, 3, 3)` containing the cell vectors of the configurations.
- `ptr`: `torch.Tensor` of shape `(batch_size + 1,)` containing the indices of the first atom of each configuration in 
the `species` and `pos` tensors.
- `property`: dict containing the properties of the configurations.
