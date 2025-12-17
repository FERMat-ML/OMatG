# RL Module - Lightning Integration

This guide covers the **PyTorch Lightning** interface for residual policy learning, which integrates seamlessly with your existing OMatG workflow.

## Why Lightning?

Using PyTorch Lightning for RL training gives you:

- ✅ **Consistent interface** with your base model training (OMGLightning)
- ✅ **Multi-GPU training** with minimal code changes
- ✅ **Automatic checkpointing** and resuming
- ✅ **TensorBoard logging** built-in
- ✅ **Early stopping**, learning rate scheduling, and other callbacks
- ✅ **Mixed precision training** for faster computation
- ✅ **No need for TorchRL** - simpler dependencies

## Quick Start

### Basic Training

```python
import lightning
from omg.omg_lightning import OMGLightning
from omg.rl import OMGRLLightning, ResidualModel, VolumeReward, RLConfig

# 1. Load base model
base_lightning = OMGLightning.load_from_checkpoint("base_model.ckpt")

# 2. Create residual model
residual_model = ResidualModel(
    base_architecture=your_architecture,
    noise_scale=0.1,
    regularization_weight=0.01,
)

# 3. Define reward
reward_fn = VolumeReward(scale=1.0)

# 4. Create RL module
rl_module = OMGRLLightning(
    base_model=base_lightning.model,
    residual_model=residual_model,
    stochastic_interpolants=base_lightning.si,
    sampler=base_lightning.sampler,
    reward_function=reward_fn,
    config=RLConfig(algorithm='grpo', batch_size=32),
)

# 5. Train with Lightning
trainer = lightning.Trainer(max_epochs=100, accelerator='auto')
trainer.fit(rl_module, datamodule)
```

That's it! Lightning handles everything else.

## Complete Example

```python
from pathlib import Path
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from omg.omg_lightning import OMGLightning
from omg.datamodule import OMGDataModule
from omg.rl import OMGRLLightning, ResidualModel, RLConfig
from omg.rl.reward_functions import VolumeReward

# Load base model
base_lightning = OMGLightning.load_from_checkpoint("checkpoints/base_final.ckpt")

# Create residual model
from copy import deepcopy
residual_arch = deepcopy(base_lightning.model)  # Or create smaller version

residual_model = ResidualModel(
    base_architecture=residual_arch,
    noise_scale=0.1,
    regularization_weight=0.01,
)

# Define reward
reward_fn = VolumeReward(scale=1.0)

# Configure RL
config = RLConfig(
    algorithm='grpo',
    batch_size=64,
    learning_rate=1e-4,
    noise_anneal=True,
    grpo_group_size=8,
)

# Create RL Lightning module
rl_module = OMGRLLightning(
    base_model=base_lightning.model,
    residual_model=residual_model,
    stochastic_interpolants=base_lightning.si,
    sampler=base_lightning.sampler,
    reward_function=reward_fn,
    config=config,
    save_structures=True,
    structure_save_dir=Path("./best_structures"),
)

# Setup data
datamodule = OMGDataModule(
    train_csv="data/mp_20/train.csv",
    val_csv="data/mp_20/val.csv",
    batch_size=64,
)

# Setup callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath='./rl_checkpoints',
    filename='residual-{epoch:02d}-{val_reward_mean:.4f}',
    monitor='val_reward_mean',
    mode='max',
    save_top_k=3,
)

early_stop = EarlyStopping(
    monitor='val_reward_mean',
    patience=30,
    mode='max',
)

logger = TensorBoardLogger(
    save_dir='./rl_logs',
    name='residual_training',
)

# Create trainer
trainer = lightning.Trainer(
    max_epochs=200,
    accelerator='auto',
    devices=1,
    callbacks=[checkpoint_callback, early_stop],
    logger=logger,
    gradient_clip_val=1.0,
    log_every_n_steps=10,
)

# Train!
trainer.fit(rl_module, datamodule)

# Load best model
best_model = OMGRLLightning.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    # ... (provide same arguments as above)
)
```

## Lightning Features You Get for Free

### 1. Multi-GPU Training

Just change one line:

```python
trainer = lightning.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=4,  # Use 4 GPUs
    strategy='ddp',  # Distributed Data Parallel
)
```

Lightning handles all the distributed training complexity!

### 2. Mixed Precision (Faster Training)

```python
trainer = lightning.Trainer(
    max_epochs=100,
    precision='16-mixed',  # Use mixed precision
)
```

### 3. Automatic Checkpointing

```python
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    monitor='val_reward_mean',
    mode='max',
    save_top_k=5,  # Keep best 5
    save_last=True,  # Also save last
)

trainer = lightning.Trainer(callbacks=[checkpoint])
```

### 4. Early Stopping

```python
from lightning.pytorch.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_reward_mean',
    patience=20,
    mode='max',
)

trainer = lightning.Trainer(callbacks=[early_stop])
```

### 5. TensorBoard Logging

```python
from lightning.pytorch.loggers import TensorBoardLogger

logger = TensorBoardLogger(
    save_dir='./logs',
    name='my_experiment',
)

trainer = lightning.Trainer(logger=logger)
```

View with: `tensorboard --logdir=./logs`

### 6. Learning Rate Scheduling

```python
# In OMGRLLightning, override configure_optimizers:

def configure_optimizers(self):
    optimizer = Adam(self.residual_model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_reward_mean",
        }
    }
```

### 7. Progress Bars

Automatic with rich information:
```
Epoch 5: 100%|███████| 125/125 [01:23<00:00,  1.50it/s, v_num=0, train_loss=0.245, train_reward_mean=123.4]
```

## Logged Metrics

The Lightning module automatically logs:

**Training:**
- `train_loss` - Total loss
- `train_reward_mean` - Mean reward
- `train_reward_std` - Reward standard deviation
- `train_policy_loss` - Policy gradient loss
- `train_reg_loss` - Regularization loss
- `noise_scale` - Current noise scale (if annealing)

**Validation:**
- `val_loss` - Validation loss
- `val_reward_mean` - Mean validation reward
- `val_reward_std` - Validation reward std
- `val_policy_loss` - Validation policy loss
- `val_reg_loss` - Validation regularization loss

All accessible in TensorBoard!

## Comparing with Base OMGLightning

Your base model training and RL training now have the same interface:

```python
# Base model training (supervised)
base_module = OMGLightning(si, sampler, model, ...)
trainer.fit(base_module, datamodule)

# RL training (reinforcement learning)
rl_module = OMGRLLightning(base_model, residual_model, si, sampler, reward_fn, ...)
trainer.fit(rl_module, datamodule)
```

Same callbacks, same logging, same workflow!

## Advanced Usage

### Custom Callbacks

```python
class RewardMonitor(lightning.Callback):
    """Custom callback to monitor rewards."""

    def on_validation_epoch_end(self, trainer, pl_module):
        reward = trainer.callback_metrics.get('val_reward_mean')
        if reward is not None:
            print(f"Current reward: {reward:.4f}")
            # Could save structures, adjust hyperparameters, etc.

trainer = lightning.Trainer(callbacks=[RewardMonitor()])
```

### Resume Training

```python
# Training was interrupted, resume from checkpoint
trainer = lightning.Trainer()
trainer.fit(rl_module, datamodule, ckpt_path='last.ckpt')
```

### Hyperparameter Tuning with Lightning

```python
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

def train_function(config):
    rl_module = OMGRLLightning(
        ...,
        config=RLConfig(
            learning_rate=config['lr'],
            batch_size=config['batch_size'],
        )
    )

    trainer = lightning.Trainer(
        max_epochs=10,
        callbacks=[TuneReportCallback(metrics='val_reward_mean')]
    )

    trainer.fit(rl_module, datamodule)

# Run hyperparameter search
analysis = tune.run(
    train_function,
    config={
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([32, 64, 128]),
    }
)
```

## Integration with Your Existing Code

If you have custom OMGLightning callbacks or extensions, they work with OMGRLLightning too!

```python
# Your existing custom callback
class MyCustomCallback(lightning.Callback):
    def on_epoch_end(self, trainer, pl_module):
        # Works with both OMGLightning and OMGRLLightning
        pass

trainer = lightning.Trainer(callbacks=[MyCustomCallback()])
```

## Migration from Standalone Trainer

If you started with the standalone `RLTrainer`, migrating to Lightning is easy:

**Before (Standalone):**
```python
from omg.rl import RLTrainer

trainer = RLTrainer(base_model, residual_model, si, sampler, reward_fn, config)
trainer.train(dataloader)
```

**After (Lightning):**
```python
from omg.rl import OMGRLLightning
import lightning

rl_module = OMGRLLightning(base_model, residual_model, si, sampler, reward_fn, config)
trainer = lightning.Trainer(max_epochs=100)
trainer.fit(rl_module, datamodule)
```

You get all Lightning features with minimal code changes!

## Troubleshooting

**Q: My base model is on GPU but RL training fails**
- Make sure you don't manually move base_model to GPU before passing to OMGRLLightning
- Lightning handles device placement automatically

**Q: How do I use multiple GPUs?**
```python
trainer = lightning.Trainer(accelerator='gpu', devices=4, strategy='ddp')
```

**Q: Training is slow**
- Enable mixed precision: `precision='16-mixed'`
- Increase batch size if you have memory
- Use multiple GPUs

**Q: How do I save only the residual model (not base)?**
The checkpoint saves everything, but to extract just residual:
```python
checkpoint = torch.load('rl_checkpoint.ckpt')
residual_state = checkpoint['state_dict']
# Filter for residual_model keys
residual_only = {k: v for k, v in residual_state.items() if k.startswith('residual_model')}
torch.save(residual_only, 'residual_only.pt')
```

**Q: Can I use this with Lightning Fabric instead of Trainer?**
Yes! OMGRLLightning is a LightningModule, so it works with both Trainer and Fabric.

## Best Practices

1. **Start simple**: Use default Lightning Trainer first
2. **Monitor rewards**: Check TensorBoard regularly
3. **Checkpoint frequently**: Use ModelCheckpoint callback
4. **Validate often**: Set `check_val_every_n_epoch` appropriately
5. **Use early stopping**: Prevent overfitting
6. **Log everything**: Lightning does this automatically
7. **Test on small data first**: Verify workflow before full training

## Next Steps

- See `example_rl_lightning.py` for complete examples
- Check Lightning docs: https://lightning.ai/docs/pytorch/stable/
- Original README.md for RL concepts and reward functions

## Summary

**PyTorch Lightning provides:**
- Same interface as your base model training
- Multi-GPU, checkpointing, logging out of the box
- Rich ecosystem of callbacks and integrations
- Production-ready training infrastructure

**Use OMGRLLightning for:**
- All your RL-based residual training
- Consistent workflow with base OMatG
- Easy experimentation and scaling

Enjoy training! ⚡
