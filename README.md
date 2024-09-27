# OMFG

## Dependencies

You can use any Python version between 3.9 and 3.12.

The following packages are required (see pyproject.toml for an up-to-date list; diffcsp should also be installed for 
CSPNet):

- torch ~= 2.4
- lightning ~= 2.4
- torch_geometric ~= 2.6

## Training

Run the following command to train from scratch:

```bash
python scripts/main.py fit --config omg/conf_examples/test_config.yaml --trainer.accelerator=cpu
```

Run the following command to restart training from a checkpoint:

```bash
python scripts/main.py fit --config omg/conf_examples/test_config.yaml --trainer.accelerator=cpu --model.load_checkpoint=<checkpoint_file.ckpt> 
```

## TODO

- Add diffcsp as a git submodule and update the dependencies.