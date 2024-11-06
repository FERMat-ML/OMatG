# OMFG

## Dependencies

You can use any Python version between 3.10 and 3.12.

The following packages are required (see pyproject.toml for an up-to-date list; diffcsp should also be installed for 
CSPNet):

- torch ~= 2.4
- lightning ~= 2.4
- torch_geometric ~= 2.6

## Training

Run the following command to train from scratch:

```bash
python scripts/main.py fit --config omg/conf_examples/test_config_ode.yaml --trainer.accelerator=cpu
```

If you want to include a Wandb logger with a name, add the `--trainer.logger=WandbLogger --trainer.logger.name=<name>` 
argument. Other loggers can be found [here](https://lightning.ai/docs/pytorch/stable/extensions/logging.html).

In order to restart training from a checkpoint, add the `--model.load_checkpoint=<checkpoint_file.ckpt>` argument. 

In order to seed the random number generators before training, use `--seed_everything=<seed>`.

## Sampling

For generating new structures run the following command:

```bash
python scripts/main.py predict --config {config_file} --model.load_checkpoint=<checkpoint_file.ckpt>
```
