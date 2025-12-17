"""
Example script using Lightning-based RL training.

This demonstrates the Lightning workflow for residual policy learning,
consistent with your existing OMatG training pipeline.
"""

import torch
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

from omg.omg_lightning import OMGLightning
from omg.datamodule import OMGDataModule
from omg.rl.omg_rl_lightning import OMGRLLightning
from omg.rl.residual_model import ResidualModel
from omg.rl.reward_functions import VolumeReward, DensityReward, CompositeReward
from omg.rl.rl_config import RLConfig


def example_basic_training():
    """
    Basic example: Train residual model using Lightning.

    This is the recommended approach - uses Lightning for everything!
    """
    print("=" * 80)
    print("Example 1: Basic Lightning-based RL Training")
    print("=" * 80)

    # 1. Load your pre-trained base model
    base_checkpoint = "path/to/your/base_model.ckpt"
    base_lightning = OMGLightning.load_from_checkpoint(base_checkpoint)

    # 2. Create residual model
    # You can use same architecture or smaller
    from copy import deepcopy
    residual_architecture = deepcopy(base_lightning.model)  # Or create smaller version

    residual_model = ResidualModel(
        base_architecture=residual_architecture,
        noise_scale=0.1,
        regularization_weight=0.01,
    )

    # 3. Define reward function
    reward_function = VolumeReward(scale=1.0)

    # 4. Configure RL
    rl_config = RLConfig(
        algorithm='grpo',
        batch_size=32,
        learning_rate=1e-4,
        num_iterations=1000,  # This is handled by Lightning epochs now
        noise_scale=0.1,
        noise_anneal=True,
        regularization_weight=0.01,
        grpo_group_size=8,
    )

    # 5. Create RL Lightning module
    rl_module = OMGRLLightning(
        base_model=base_lightning.model,
        residual_model=residual_model,
        stochastic_interpolants=base_lightning.si,
        sampler=base_lightning.sampler,
        reward_function=reward_function,
        config=rl_config,
        save_structures=True,
        structure_save_dir=Path("./generated_structures"),
    )

    # 6. Setup data
    # Use same datamodule as base training, or create new one
    datamodule = OMGDataModule.load_from_checkpoint(base_checkpoint)
    # Or create fresh: datamodule = OMGDataModule(...)

    # 7. Setup Lightning trainer with callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./rl_checkpoints',
        filename='residual-{epoch:02d}-{val_reward_mean:.4f}',
        monitor='val_reward_mean',
        mode='max',  # Maximize reward
        save_top_k=3,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    early_stop_callback = EarlyStopping(
        monitor='val_reward_mean',
        patience=50,
        mode='max',
        verbose=True,
    )

    logger = TensorBoardLogger(
        save_dir='./rl_logs',
        name='residual_training',
    )

    trainer = lightning.Trainer(
        max_epochs=100,
        accelerator='auto',  # Automatically use GPU if available
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # Gradient clipping
        # precision='16-mixed',  # Uncomment for mixed precision
    )

    # 8. Train!
    print("\nStarting RL training with Lightning...")
    trainer.fit(rl_module, datamodule)

    print("\nTraining complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    # 9. Load best model for inference
    best_model = OMGRLLightning.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        base_model=base_lightning.model,
        residual_model=residual_model,
        stochastic_interpolants=base_lightning.si,
        sampler=base_lightning.sampler,
        reward_function=reward_function,
        config=rl_config,
    )
    best_model.eval()

    print("Best model loaded and ready for inference!")

    return best_model


def example_multi_gpu_training():
    """
    Example: Multi-GPU distributed training.

    Lightning makes this trivial!
    """
    print("=" * 80)
    print("Example 2: Multi-GPU Training")
    print("=" * 80)

    # Setup (same as basic example)
    # ... (omitted for brevity)

    # The only difference: configure trainer for multiple GPUs
    trainer = lightning.Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=4,  # Use 4 GPUs
        strategy='ddp',  # Distributed Data Parallel
        # Everything else same as before
    )

    print("Lightning will automatically handle multi-GPU training!")


def example_with_validation():
    """
    Example: Training with validation and early stopping.
    """
    print("=" * 80)
    print("Example 3: Training with Validation")
    print("=" * 80)

    # Setup components (same as before)
    base_lightning = OMGLightning.load_from_checkpoint("base.ckpt")
    residual_model = ResidualModel(...)  # Your setup
    reward_function = VolumeReward()
    rl_config = RLConfig(algorithm='grpo', batch_size=32)

    rl_module = OMGRLLightning(
        base_model=base_lightning.model,
        residual_model=residual_model,
        stochastic_interpolants=base_lightning.si,
        sampler=base_lightning.sampler,
        reward_function=reward_function,
        config=rl_config,
    )

    # Data with train/val split
    datamodule = OMGDataModule(
        train_csv="train.csv",
        val_csv="val.csv",
        batch_size=32,
        # ... other params
    )

    # Early stopping based on validation reward
    early_stop = EarlyStopping(
        monitor='val_reward_mean',
        patience=20,
        mode='max',
    )

    trainer = lightning.Trainer(
        max_epochs=100,
        callbacks=[early_stop],
        check_val_every_n_epoch=5,  # Validate every 5 epochs
    )

    trainer.fit(rl_module, datamodule)


def example_composite_reward():
    """
    Example: Multi-objective optimization with composite reward.
    """
    print("=" * 80)
    print("Example 4: Multi-Objective Optimization")
    print("=" * 80)

    # Setup base
    base_lightning = OMGLightning.load_from_checkpoint("base.ckpt")
    residual_model = ResidualModel(...)

    # Combine multiple rewards
    volume_reward = VolumeReward(scale=1.0)
    density_reward = DensityReward(target_density=3.5, tolerance=0.5)

    composite_reward = CompositeReward(
        reward_functions=[density_reward, volume_reward],
        weights=[0.7, 0.3],  # 70% density, 30% volume
    )

    rl_config = RLConfig(algorithm='grpo', batch_size=32)

    rl_module = OMGRLLightning(
        base_model=base_lightning.model,
        residual_model=residual_model,
        stochastic_interpolants=base_lightning.si,
        sampler=base_lightning.sampler,
        reward_function=composite_reward,  # Use composite
        config=rl_config,
    )

    # Train as usual
    trainer = lightning.Trainer(max_epochs=100)
    trainer.fit(rl_module, datamodule)


def example_custom_reward():
    """
    Example: Define and use a custom reward function.
    """
    print("=" * 80)
    print("Example 5: Custom Reward Function")
    print("=" * 80)

    from omg.rl.reward_functions import RewardFunction

    class BandGapReward(RewardFunction):
        """Custom reward based on band gap."""

        def __init__(self, target_gap: float = 2.0):
            self.target_gap = target_gap

        def compute(self, structures):
            import torch
            rewards = []
            for atoms in structures:
                # Your band gap calculation here
                # This is a placeholder
                gap = self.calculate_band_gap(atoms)
                reward = -abs(gap - self.target_gap)
                rewards.append(reward)
            return torch.tensor(rewards, dtype=torch.float32)

        def calculate_band_gap(self, atoms):
            """Placeholder for actual band gap calculation."""
            # In practice, call your ML model or DFT here
            return 1.5  # Dummy value

        def is_differentiable(self):
            return False

    # Use custom reward
    reward_function = BandGapReward(target_gap=2.0)

    # Setup and train as usual
    rl_module = OMGRLLightning(
        base_model=...,
        residual_model=...,
        stochastic_interpolants=...,
        sampler=...,
        reward_function=reward_function,  # Your custom reward
        config=RLConfig(algorithm='grpo'),
    )

    trainer = lightning.Trainer(max_epochs=100)
    trainer.fit(rl_module, datamodule)


def example_inference():
    """
    Example: Use trained model for inference.
    """
    print("=" * 80)
    print("Example 6: Inference with Trained Residual Model")
    print("=" * 80)

    # Load trained RL module
    checkpoint_path = "rl_checkpoints/best.ckpt"

    # You need to provide the components
    base_lightning = OMGLightning.load_from_checkpoint("base.ckpt")
    residual_model = ResidualModel(...)  # Same architecture as training
    reward_function = VolumeReward()  # Same as training

    rl_module = OMGRLLightning.load_from_checkpoint(
        checkpoint_path,
        base_model=base_lightning.model,
        residual_model=residual_model,
        stochastic_interpolants=base_lightning.si,
        sampler=base_lightning.sampler,
        reward_function=reward_function,
        config=RLConfig(),  # Will be loaded from checkpoint
    )

    rl_module.eval()

    # Generate structures
    # Use Lightning's predict functionality
    trainer = lightning.Trainer()
    predictions = trainer.predict(rl_module, dataloaders=test_dataloader)

    print(f"Generated {len(predictions)} batches of structures")


def example_algorithm_comparison():
    """
    Example: Compare REINFORCE vs GRPO.
    """
    print("=" * 80)
    print("Example 7: Algorithm Comparison")
    print("=" * 80)

    base_lightning = OMGLightning.load_from_checkpoint("base.ckpt")

    for algorithm in ['reinforce', 'grpo']:
        print(f"\nTraining with {algorithm.upper()}...")

        # Create fresh residual model for each
        residual_model = ResidualModel(...)

        rl_config = RLConfig(
            algorithm=algorithm,
            batch_size=32,
            grpo_group_size=8,  # Only used for GRPO
        )

        rl_module = OMGRLLightning(
            base_model=base_lightning.model,
            residual_model=residual_model,
            stochastic_interpolants=base_lightning.si,
            sampler=base_lightning.sampler,
            reward_function=VolumeReward(),
            config=rl_config,
        )

        # Separate logger for each algorithm
        logger = TensorBoardLogger(
            save_dir='./rl_logs',
            name=f'algorithm_{algorithm}',
        )

        trainer = lightning.Trainer(
            max_epochs=100,
            logger=logger,
        )

        trainer.fit(rl_module, datamodule)

        print(f"{algorithm.upper()} training complete!")

    print("\nCompare results in TensorBoard:")
    print("tensorboard --logdir=./rl_logs")


def example_full_workflow():
    """
    Complete workflow from scratch.
    """
    print("=" * 80)
    print("Example 8: Complete Workflow")
    print("=" * 80)

    # This is the full pipeline you'd actually use

    # 1. Load base model
    print("1. Loading base model...")
    base_lightning = OMGLightning.load_from_checkpoint(
        "checkpoints/base_model_final.ckpt"
    )

    # 2. Create smaller residual architecture (optional)
    print("2. Creating residual model...")
    # Option A: Same size
    from copy import deepcopy
    residual_arch = deepcopy(base_lightning.model)

    # Option B: Smaller (example - adjust to your architecture)
    # residual_arch = Model(
    #     encoder=create_encoder(hidden_dim=128),  # vs 256 in base
    #     head=create_head(...),
    #     ...
    # )

    residual_model = ResidualModel(
        base_architecture=residual_arch,
        noise_scale=0.1,
        regularization_weight=0.01,
    )

    # 3. Define reward
    print("3. Setting up reward function...")
    reward_function = VolumeReward(scale=1.0)
    # Or your custom reward

    # 4. Configure RL
    print("4. Configuring RL training...")
    rl_config = RLConfig(
        algorithm='grpo',
        batch_size=64,
        learning_rate=1e-4,
        noise_scale=0.1,
        noise_anneal=True,
        noise_anneal_factor=0.999,
        regularization_weight=0.01,
        grpo_group_size=8,
    )

    # 5. Create RL module
    print("5. Creating RL Lightning module...")
    rl_module = OMGRLLightning(
        base_model=base_lightning.model,
        residual_model=residual_model,
        stochastic_interpolants=base_lightning.si,
        sampler=base_lightning.sampler,
        reward_function=reward_function,
        config=rl_config,
        save_structures=True,
        structure_save_dir=Path("./best_structures"),
    )

    # 6. Setup data
    print("6. Setting up data...")
    datamodule = OMGDataModule(
        train_csv="data/mp_20/train.csv",
        val_csv="data/mp_20/val.csv",
        batch_size=64,
        # ... other params
    )

    # 7. Setup callbacks
    print("7. Configuring callbacks...")
    callbacks = [
        ModelCheckpoint(
            dirpath='./rl_checkpoints',
            filename='residual-{epoch:02d}-{val_reward_mean:.4f}',
            monitor='val_reward_mean',
            mode='max',
            save_top_k=5,
            save_last=True,
        ),
        EarlyStopping(
            monitor='val_reward_mean',
            patience=30,
            mode='max',
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    logger = TensorBoardLogger(
        save_dir='./rl_logs',
        name='residual_volume_optimization',
    )

    # 8. Create trainer
    print("8. Creating Lightning trainer...")
    trainer = lightning.Trainer(
        max_epochs=200,
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=5,
    )

    # 9. Train
    print("9. Starting training...")
    trainer.fit(rl_module, datamodule)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best model: {callbacks[0].best_model_path}")
    print(f"View logs: tensorboard --logdir=./rl_logs")
    print("=" * 80)


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  OMatG Residual RL Training - Lightning Examples              ║
    ║                                                                ║
    ║  These examples show how to use PyTorch Lightning for         ║
    ║  RL-based residual learning - consistent with your existing   ║
    ║  OMatG training workflow!                                      ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    print("\n📖 Example workflows:")
    print("  1. Basic training")
    print("  2. Multi-GPU training")
    print("  3. Training with validation")
    print("  4. Multi-objective optimization")
    print("  5. Custom reward function")
    print("  6. Inference")
    print("  7. Algorithm comparison")
    print("  8. Complete workflow\n")

    # Uncomment to run examples:
    # example_basic_training()
    # example_full_workflow()

    print("To run an example, uncomment it in the script!")
