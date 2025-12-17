# Setup Guide for RL Module

## Installation

### 1. Install Base OMatG
First, ensure you have the base OMatG package installed:
```bash
cd /path/to/OMatG-Private
pip install -e .
```

### 2. Install RL Dependencies
Install additional dependencies for the RL module:
```bash
pip install -r omg/rl/requirements.txt
```

Or install individually:
```bash
pip install torchrl tensordict
```

### 3. Verify Installation
Run the tests to verify everything is installed correctly:
```bash
pytest omg/rl/tests/test_residual_model.py -v
```

## Quick Start

### 1. Prepare Your Base Model

You need a trained base flow matching model. If you don't have one:
```bash
# Train a base model first (using standard OMatG)
omg fit --config your_config.yaml
```

### 2. Create a Training Script

Copy and modify the example:
```bash
cp omg/rl/example_rl_training.py my_rl_training.py
```

Edit `my_rl_training.py`:
- Update checkpoint paths
- Define your reward function
- Adjust hyperparameters

### 3. Define Your Reward Function

Create a custom reward in a new file `my_rewards.py`:

```python
from omg.rl.reward_functions import RewardFunction
import torch

class MyReward(RewardFunction):
    def compute(self, structures):
        # Your reward logic here
        rewards = []
        for atoms in structures:
            # Example: reward based on some property
            reward = compute_my_property(atoms)
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)

    def is_differentiable(self):
        return False  # Change if differentiable
```

### 4. Train

Run your training script:
```python
python my_rl_training.py
```

### 5. Use Trained Model

```python
from omg.rl import ResidualModel
from omg.rl.integration_guide import ResidualIntegrator

# Load
residual_model = ResidualModel(...)
residual_model.load_state_dict(torch.load("residual_model.pt"))
residual_model.eval()

# Generate
integrator = ResidualIntegrator(base_model, residual_model, si)
structures = integrator.integrate(x_0)
```

## Directory Structure

After setup, you should have:
```
omg/
├── rl/
│   ├── __init__.py                  # Module exports
│   ├── residual_model.py            # Core residual model
│   ├── reward_functions.py          # Reward function classes
│   ├── rl_config.py                 # Configuration
│   ├── rl_trainer.py                # Training loop
│   ├── integration_guide.py         # Inference integration
│   ├── example_rl_training.py       # Usage examples
│   ├── README.md                    # Documentation
│   ├── SETUP.md                     # This file
│   ├── IMPLEMENTATION_SUMMARY.md    # Technical summary
│   ├── requirements.txt             # Dependencies
│   └── tests/
│       ├── __init__.py
│       └── test_residual_model.py   # Unit tests
```

## Configuration

### Minimal Configuration

```python
from omg.rl import RLConfig

config = RLConfig(
    algorithm='grpo',
    batch_size=32,
    learning_rate=1e-4,
    num_iterations=1000,
)
```

### Recommended Configuration

For most cases:
```python
config = RLConfig(
    algorithm='grpo',
    batch_size=64,
    learning_rate=1e-4,
    num_iterations=5000,
    noise_scale=0.1,
    noise_anneal=True,
    noise_anneal_factor=0.999,
    regularization_weight=0.01,
    grpo_group_size=8,
    max_grad_norm=1.0,
    log_interval=50,
    save_interval=500,
)
```

### For Quick Experiments

```python
config = RLConfig(
    algorithm='grpo',
    batch_size=16,
    learning_rate=5e-4,
    num_iterations=100,
    log_interval=10,
)
```

## Common Issues

### Import Error: `No module named 'torchrl'`
```bash
pip install torchrl tensordict
```

### Import Error: `No module named 'omg'`
Make sure you're in the OMatG directory and have installed it:
```bash
pip install -e .
```

### CUDA Out of Memory
- Reduce `batch_size` in config
- Use smaller residual model architecture
- Move some computation to CPU

### Training Not Improving
- Check reward function (higher = better)
- Increase batch size
- Reduce regularization_weight
- Try different learning rate
- Use GRPO instead of REINFORCE

### Rewards Too Noisy
- Increase batch_size
- Use GRPO (group normalization)
- Check reward scaling

## Hardware Requirements

### Minimum
- CPU: Any modern CPU
- RAM: 16 GB
- GPU: Optional but recommended

### Recommended
- CPU: Multi-core (8+ cores)
- RAM: 32 GB
- GPU: NVIDIA GPU with 8+ GB VRAM (for base model + residual)

### For Large-Scale Training
- GPU: NVIDIA GPU with 16+ GB VRAM
- Multiple GPUs for distributed training (future feature)

## Development Setup

If you want to contribute or modify the RL module:

### 1. Install in Development Mode
```bash
pip install -e .[dev]
```

### 2. Install Pre-commit Hooks (Optional)
```bash
pip install pre-commit
pre-commit install
```

### 3. Run Tests
```bash
pytest omg/rl/tests/ -v --cov=omg.rl
```

### 4. Format Code
```bash
black omg/rl/
isort omg/rl/
```

## Next Steps

1. ✅ Complete installation
2. ✅ Run tests
3. ✅ Review `README.md`
4. ✅ Study `example_rl_training.py`
5. ✅ Define your reward function
6. ✅ Run small experiment (100 iterations)
7. ✅ Analyze results
8. ✅ Scale up training
9. ✅ Evaluate on test set
10. ✅ Use for generation

## Getting Help

- **Documentation**: See `README.md` for detailed docs
- **Examples**: Check `example_rl_training.py`
- **Tests**: Look at `tests/test_residual_model.py` for usage patterns
- **Integration**: See `integration_guide.py` for inference

## Troubleshooting Checklist

Before asking for help, verify:
- [ ] Base OMatG is installed and working
- [ ] All dependencies are installed (`pip install -r omg/rl/requirements.txt`)
- [ ] Tests pass (`pytest omg/rl/tests/`)
- [ ] You have a trained base model checkpoint
- [ ] Your reward function returns correct shapes
- [ ] Config validation passes

## Tips

1. **Start small**: Test with 100 iterations before scaling up
2. **Monitor rewards**: Should generally increase over training
3. **Save checkpoints**: Training can be unstable, save frequently
4. **Compare baselines**: Generate with and without residual
5. **Visualize**: Plot reward curves to diagnose issues
6. **Regularization**: Start with 0.01, adjust if residuals too large/small

## Resources

- TorchRL docs: https://pytorch.org/rl/
- ASE docs: https://wiki.fysik.dtu.dk/ase/
- PyTorch Lightning: https://lightning.ai/docs/pytorch/

## Status

✅ Framework complete and ready for testing
🔄 Awaiting user feedback and iteration
📈 Future: PPO implementation, distributed training, more reward functions

Happy training!
