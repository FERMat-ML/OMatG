"""
Example script demonstrating how to use the RL framework for residual policy learning.

This script shows how to:
1. Load a pre-trained base model
2. Create a residual model with the same architecture (but potentially smaller)
3. Define a reward function
4. Train the residual model using RL
5. Use the trained residual model for inference
"""

import torch
from pathlib import Path

# OMatG imports (adjust as needed based on your setup)
from omg.rl import ResidualModel, RLTrainer, RLConfig
from omg.rl.reward_functions import VolumeReward, DensityReward, StabilityReward, CompositeReward
from omg.omg_lightning import OMGLightning
from omg.datamodule import OMGDataModule

# Example: Train a residual model to maximize volume


def example_volume_maximization():
    """
    Example: Train residual model to maximize crystal volume.
    """
    print("=" * 80)
    print("Example 1: Volume Maximization")
    print("=" * 80)

    # 1. Load pre-trained base model
    # You would typically load this from a checkpoint
    checkpoint_path = Path("path/to/your/base_model_checkpoint.ckpt")
    base_lightning = OMGLightning.load_from_checkpoint(checkpoint_path)
    base_model = base_lightning.model
    si = base_lightning.si
    sampler = base_lightning.sampler

    # 2. Create residual model
    # Use the same architecture as base model, but you could make it smaller
    # For example, reduce hidden dimensions, number of layers, etc.
    from copy import deepcopy
    residual_architecture = deepcopy(base_model)  # Or create a smaller version

    residual_model = ResidualModel(
        base_architecture=residual_architecture,
        noise_scale=0.1,
        regularization_weight=0.01,
    )

    # 3. Define reward function
    reward_function = VolumeReward(scale=1.0)

    # 4. Configure RL training
    config = RLConfig(
        algorithm='grpo',  # Options: 'reinforce', 'ppo', 'grpo'
        batch_size=32,
        learning_rate=1e-4,
        num_iterations=1000,
        noise_scale=0.1,
        noise_anneal=True,
        regularization_weight=0.01,
        grpo_group_size=8,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # 5. Create trainer
    trainer = RLTrainer(
        base_model=base_model,
        residual_model=residual_model,
        stochastic_interpolants=si,
        sampler=sampler,
        reward_function=reward_function,
        config=config,
        checkpoint_dir=Path("./rl_checkpoints/volume_max"),
    )

    # 6. Setup data
    # You'll need a dataloader that provides structures/compositions
    datamodule = OMGDataModule.load_from_checkpoint(checkpoint_path)
    datamodule.setup('fit')
    train_dataloader = datamodule.train_dataloader()

    # 7. Train
    print("\nStarting RL training...")
    trainer.train(train_dataloader)

    print("\nTraining complete!")

    # 8. Save final model
    torch.save(residual_model.state_dict(), "residual_model_volume.pt")

    return residual_model


def example_stability_optimization():
    """
    Example: Train residual model to optimize for structural stability.
    """
    print("=" * 80)
    print("Example 2: Stability Optimization")
    print("=" * 80)

    # Setup similar to above...
    # (skipping redundant setup code)

    # Define a stability reward with an energy calculator
    # This is a placeholder - you'd provide your actual energy calculator
    def dummy_energy_calculator(atoms):
        """Placeholder energy calculator."""
        # In practice, this would call DFT, ML potential, etc.
        return -2.5 * len(atoms)  # Dummy energy

    reward_function = StabilityReward(
        energy_calculator=dummy_energy_calculator,
        energy_scale=1.0,
    )

    # Rest of training is the same...
    print("This example uses a stability-based reward function")


def example_composite_reward():
    """
    Example: Train residual model with multiple objectives.
    """
    print("=" * 80)
    print("Example 3: Multi-Objective Optimization")
    print("=" * 80)

    # Combine multiple reward functions
    volume_reward = VolumeReward(scale=1.0)
    density_reward = DensityReward(target_density=3.5, tolerance=0.5)

    # 60% weight on density, 40% on volume
    composite_reward = CompositeReward(
        reward_functions=[density_reward, volume_reward],
        weights=[0.6, 0.4],
    )

    # Use composite_reward in trainer...
    print("This example optimizes for both target density and volume")


def example_inference():
    """
    Example: Use trained residual model for inference.
    """
    print("=" * 80)
    print("Example 4: Inference with Trained Residual Model")
    print("=" * 80)

    # Load base model
    checkpoint_path = Path("path/to/your/base_model_checkpoint.ckpt")
    base_lightning = OMGLightning.load_from_checkpoint(checkpoint_path)

    # Load trained residual model
    # First create the architecture
    from copy import deepcopy
    residual_architecture = deepcopy(base_lightning.model)

    residual_model = ResidualModel(
        base_architecture=residual_architecture,
        noise_scale=0.1,  # Will be unused during inference
        regularization_weight=0.01,
    )

    # Load trained weights
    residual_model.load_state_dict(torch.load("residual_model_volume.pt"))
    residual_model.eval()  # Set to eval mode (no noise)

    # Generate structures
    # During inference, you would integrate using base + residual
    # The residual_model.forward() will return deterministic outputs when not in training mode

    print("During inference:")
    print("- residual_model is in eval mode (no noise)")
    print("- Combine base_model velocities with residual_model velocities")
    print("- Integrate as usual")


def example_algorithm_comparison():
    """
    Example: Compare different RL algorithms.
    """
    print("=" * 80)
    print("Example 5: Comparing RL Algorithms")
    print("=" * 80)

    algorithms = ['reinforce', 'grpo']  # PPO requires more setup

    for algo in algorithms:
        print(f"\nTraining with {algo.upper()}...")

        config = RLConfig(
            algorithm=algo,
            batch_size=32,
            learning_rate=1e-4,
            num_iterations=100,  # Fewer iterations for comparison
            grpo_group_size=8,
        )

        # Setup and train...
        print(f"  {algo} configured")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  OMatG Residual Policy Learning Examples                      ║
    ║                                                                ║
    ║  This script demonstrates various use cases for RL-based       ║
    ║  residual learning in crystal structure generation.           ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    # Run examples (comment out as needed)
    # Note: These are demonstrations of the API, not complete runnable examples
    # You'll need to provide actual model checkpoints and data

    print("\n📖 These are API usage examples.")
    print("To run them, you need to:")
    print("  1. Provide a trained base model checkpoint")
    print("  2. Setup your data module")
    print("  3. Define your reward function")
    print("\n")

    # Uncomment to see example outputs:
    # example_volume_maximization()
    # example_stability_optimization()
    # example_composite_reward()
    # example_inference()
    # example_algorithm_comparison()

    print("Example demonstrations complete!")
