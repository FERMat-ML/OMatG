# Lightning Quick Start

Get started with RL-based residual learning in 5 minutes!

## Installation

```bash
# RL module uses Lightning (already installed with base OMatG)
pip install -r omg/rl/requirements.txt
```

## Minimal Example

```python
import lightning
from omg.omg_lightning import OMGLightning
from omg.datamodule import OMGDataModule
from omg.rl import OMGRLLightning, ResidualModel, RLConfig, VolumeReward

# 1. Load base model
base = OMGLightning.load_from_checkpoint("path/to/base_model.ckpt")

# 2. Create residual model (same architecture as base)
from copy import deepcopy
residual = ResidualModel(
    base_architecture=deepcopy(base.model),
    noise_scale=0.1,
    regularization_weight=0.01,
)

# 3. Create RL module
rl_module = OMGRLLightning(
    base_model=base.model,
    residual_model=residual,
    stochastic_interpolants=base.si,
    sampler=base.sampler,
    reward_function=VolumeReward(),  # Or your custom reward
    config=RLConfig(algorithm='grpo', batch_size=32),
)

# 4. Setup data
datamodule = OMGDataModule(
    train_csv="data/train.csv",
    val_csv="data/val.csv",
    batch_size=32,
)

# 5. Train!
trainer = lightning.Trainer(max_epochs=100, accelerator='auto')
trainer.fit(rl_module, datamodule)
```

Done! Lightning handles everything else (logging, checkpointing, GPU, etc.).

## Custom Reward

```python
from omg.rl.reward_functions import RewardFunction
import torch

class MyReward(RewardFunction):
    def compute(self, structures):
        rewards = []
        for atoms in structures:
            # Your reward logic
            reward = your_calculation(atoms)
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)

    def is_differentiable(self):
        return False  # or True

# Use it
rl_module = OMGRLLightning(
    ...,
    reward_function=MyReward(),
    ...
)
```

## With Callbacks

```python
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

trainer = lightning.Trainer(
    max_epochs=100,
    callbacks=[
        ModelCheckpoint(monitor='val_reward_mean', mode='max'),
        EarlyStopping(monitor='val_reward_mean', patience=20),
    ],
)

trainer.fit(rl_module, datamodule)
```

## Multi-GPU

```python
trainer = lightning.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=4,  # Use 4 GPUs
    strategy='ddp',
)
```

## Load and Use

```python
# Load best checkpoint
best = OMGRLLightning.load_from_checkpoint(
    'checkpoints/best.ckpt',
    base_model=base.model,
    residual_model=residual,
    stochastic_interpolants=base.si,
    sampler=base.sampler,
    reward_function=reward_fn,
    config=config,
)

best.eval()
# Use for generation
```

## Next Steps

- 📖 Read `README_LIGHTNING.md` for full documentation
- 💡 Check `example_rl_lightning.py` for complete examples
- ⚡ See `LIGHTNING_SUMMARY.md` for implementation details

Happy training! ⚡
