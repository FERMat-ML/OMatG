# Residual Policy Learning for OMatG

This module implements a flexible reinforcement learning framework for training residual models that learn small velocity corrections on top of a pre-trained flow matching model.

## Overview

### The Approach

The residual policy learning framework allows you to:

1. **Keep your base flow model frozen**: Preserve the diversity and quality of your pre-trained model
2. **Train task-specific corrections**: Learn small velocity nudges optimized for specific objectives
3. **Use flexible rewards**: Support any reward function (differentiable or non-differentiable)
4. **Switch between tasks**: Train different residual models for different objectives

### How It Works

During training:
- Base model provides the main velocity field (frozen)
- Residual model learns correction velocities (trainable)
- Stochastic noise is added to residual velocities to enable policy gradients
- Rewards guide the residual model to improve generated structures

During inference:
- Residual model outputs deterministic corrections (no noise)
- Total velocity = base velocity + residual velocity
- Integrate as usual to generate structures

## Components

### 1. `ResidualModel`

Wraps your model architecture and adds stochastic policy capabilities:

```python
from omg.rl import ResidualModel

residual_model = ResidualModel(
    base_architecture=your_model_architecture,
    noise_scale=0.1,              # Noise for exploration
    regularization_weight=0.01,    # Penalty for large residuals
)
```

**Key features:**
- Automatic noise injection during training
- Deterministic during inference
- Built-in regularization to keep residuals small

### 2. `RewardFunction`

Abstract base class for defining objectives:

```python
from omg.rl.reward_functions import RewardFunction

class MyReward(RewardFunction):
    def compute(self, structures: List[Atoms]) -> torch.Tensor:
        # Compute rewards for each structure
        rewards = ...
        return rewards

    def is_differentiable(self) -> bool:
        return False  # or True if differentiable
```

**Included reward functions:**
- `VolumeReward`: Maximize unit cell volume (toy example)
- `DensityReward`: Match target density
- `StabilityReward`: Optimize structural stability (interfaces with energy calculators)
- `CompositeReward`: Combine multiple rewards with weights

### 3. `RLConfig`

Configuration for training:

```python
from omg.rl import RLConfig

config = RLConfig(
    algorithm='grpo',              # 'reinforce', 'ppo', or 'grpo'
    batch_size=32,
    learning_rate=1e-4,
    num_iterations=1000,
    noise_scale=0.1,
    noise_anneal=True,             # Gradually reduce noise
    regularization_weight=0.01,
    grpo_group_size=8,             # For GRPO algorithm
)
```

### 4. `RLTrainer`

Main training loop:

```python
from omg.rl import RLTrainer

trainer = RLTrainer(
    base_model=base_model,                    # Frozen
    residual_model=residual_model,            # Trainable
    stochastic_interpolants=si,
    sampler=sampler,
    reward_function=reward_function,
    config=config,
    checkpoint_dir=Path("./checkpoints"),
)

trainer.train(dataloader)
```

## Quick Start

### Basic Usage

```python
import torch
from pathlib import Path
from omg.rl import ResidualModel, RLTrainer, RLConfig, VolumeReward

# 1. Load your base model
base_lightning = OMGLightning.load_from_checkpoint("base_model.ckpt")

# 2. Create residual model (same or smaller architecture)
residual_model = ResidualModel(
    base_architecture=create_smaller_model(),  # Your model architecture
    noise_scale=0.1,
    regularization_weight=0.01,
)

# 3. Define reward
reward_fn = VolumeReward(scale=1.0)

# 4. Configure training
config = RLConfig(
    algorithm='grpo',
    batch_size=32,
    learning_rate=1e-4,
    num_iterations=1000,
)

# 5. Train
trainer = RLTrainer(
    base_model=base_lightning.model,
    residual_model=residual_model,
    stochastic_interpolants=base_lightning.si,
    sampler=base_lightning.sampler,
    reward_function=reward_fn,
    config=config,
)

trainer.train(your_dataloader)

# 6. Save
torch.save(residual_model.state_dict(), "residual_model.pt")
```

### Inference

```python
# Load residual model
residual_model.load_state_dict(torch.load("residual_model.pt"))
residual_model.eval()  # No noise during inference

# Generate structures
# Combine base + residual velocities during integration
# (See example_rl_training.py for details)
```

## Algorithms

### REINFORCE
- Simple policy gradient method
- Good baseline, easy to understand
- Can have high variance

```python
config = RLConfig(algorithm='reinforce', ...)
```

### GRPO (Group Relative Policy Optimization)
- Normalizes rewards within groups
- Lower variance than REINFORCE
- No value network needed
- **Recommended for most cases**

```python
config = RLConfig(
    algorithm='grpo',
    grpo_group_size=8,  # Size of groups for normalization
    ...
)
```

### PPO (Proximal Policy Optimization)
- More stable training
- Requires multiple epochs per batch
- More complex, higher computational cost

```python
config = RLConfig(
    algorithm='ppo',
    ppo_epochs=4,
    clip_epsilon=0.2,
    ...
)
```

## Advanced Usage

### Custom Reward Functions

```python
from omg.rl.reward_functions import RewardFunction
import torch

class BandGapReward(RewardFunction):
    """Reward based on target band gap."""

    def __init__(self, target_gap: float = 2.0):
        self.target_gap = target_gap
        # Setup your band gap calculator here

    def compute(self, structures: List[Atoms]) -> torch.Tensor:
        rewards = []
        for atoms in structures:
            # Calculate band gap (placeholder)
            gap = calculate_band_gap(atoms)
            # Reward proximity to target
            reward = -abs(gap - self.target_gap)
            rewards.append(reward)
        return torch.tensor(rewards)

    def is_differentiable(self) -> bool:
        return False  # Band gap calculation not differentiable
```

### Multi-Objective Optimization

```python
from omg.rl.reward_functions import CompositeReward

# Combine multiple objectives
composite = CompositeReward(
    reward_functions=[
        StabilityReward(energy_calculator=my_calc),
        DensityReward(target_density=3.5),
        BandGapReward(target_gap=2.0),
    ],
    weights=[0.5, 0.3, 0.2],  # Must sum to 1
)
```

### Noise Annealing

Start with high exploration, gradually reduce:

```python
config = RLConfig(
    noise_scale=0.2,              # Initial noise
    noise_anneal=True,
    noise_anneal_factor=0.999,    # Multiply by this each iteration
    ...
)
```

### Smaller Residual Architecture

You can use a smaller model for the residual:

```python
# Create a smaller version of your base architecture
def create_smaller_residual_model():
    return Model(
        encoder=create_encoder(hidden_dim=128),  # vs 256 in base
        head=create_head(num_layers=2),          # vs 4 in base
        ...
    )

residual_model = ResidualModel(
    base_architecture=create_smaller_residual_model(),
    ...
)
```

## Tips and Best Practices

1. **Start simple**: Use GRPO with a simple reward function first
2. **Monitor rewards**: Track mean and std of rewards during training
3. **Regularization is important**: Prevents residuals from becoming too large
4. **Noise annealing**: Helps convergence - start exploring, end exploiting
5. **Smaller is better**: Residual model can be smaller than base model
6. **Batch size matters**: Larger batches = lower variance in GRPO
7. **Group size**: For GRPO, group_size=8 is a good default
8. **Checkpoint often**: Save checkpoints to recover from divergence

## Troubleshooting

**Q: Rewards not improving**
- Check reward function is correct (higher = better)
- Increase learning rate or batch size
- Check if residuals are too small (reduce regularization_weight)

**Q: Training diverges**
- Reduce learning rate
- Increase regularization_weight
- Reduce noise_scale
- Use gradient clipping (max_grad_norm in config)

**Q: High variance in rewards**
- Increase batch_size
- Use GRPO instead of REINFORCE
- Enable noise annealing

**Q: Residuals too large**
- Increase regularization_weight
- Check reward scaling

## Technical Details

### How Policy Gradients Work Here

Your flow is stochastic because the residual velocities are sampled:

```
v_residual ~ N(μ_θ(x,t), σ²I)  # μ_θ from residual network

v_total = v_base(x,t) + v_residual

dx/dt = v_total
```

The log probability of the sampled residual:
```
log π(v_residual | x, t, θ) = -||v_residual - μ_θ(x,t)||² / (2σ²)
```

Policy gradient:
```
∇_θ E[reward] = E[reward · ∇_θ log π]
```

This gradient tells us how to adjust μ_θ to get higher rewards.

### Relationship to Flow-GRPO

This approach differs from papers like "Flow-GRPO" which modify the base flow itself:
- **Flow-GRPO**: Converts deterministic ODE to SDE, trains entire flow model
- **This framework**: Adds stochastic residual on top, keeps base frozen

Benefits of residual approach:
- Simpler (no need for SDE formulation)
- Modular (can swap reward functions easily)
- Preserves base model quality

## Citation

If you use this framework, please cite the OMatG paper and mention the residual policy learning extension.

## See Also

- `example_rl_training.py` - Complete usage examples
- Main OMatG documentation
- TorchRL documentation: https://pytorch.org/rl/
