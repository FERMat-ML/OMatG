# OMFG

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
omg fit --config config.yaml --trainer.accelerator=cpu
```

This command will create checkpoints, log files, and cache files in the working directory.

If you want to include a Wandb logger with a name, add the `--trainer.logger=WandbLogger --trainer.logger.name=<name>` 
argument. Other loggers can be found [here](https://lightning.ai/docs/pytorch/stable/extensions/logging.html).

In order to restart training from a checkpoint, add the `--model.load_checkpoint=<checkpoint_file.ckpt>` argument. 

In order to seed the random number generators before training, use `--seed_everything=<seed>`.

Exemplary configuration files can be found in the `omg/conf_examples` directory.

The training command can be executed in any directory. The configuration files contain paths to lmbd data files that are 
used, e.g., for training. The path to these data files can either be relative to the working directory, or relative to 
the `omg` directory (that is, use `"data/mp_20/val.lmdb"` for `lmdb_paths` in order to use the `mp_20` data set as in 
exemplary configuration files).

## Sampling

For generating new structures in an xyz file, run the following command:

```bash
omg predict --config {config_file} --model.load_checkpoint=<checkpoint_file.ckpt> --model.generation_xyz_filename=<xyz_file>
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
