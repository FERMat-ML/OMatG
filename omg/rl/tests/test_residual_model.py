"""
Unit tests for the residual policy learning framework.

Run with: pytest omg/rl/tests/test_residual_model.py
"""

import pytest
import torch
import torch.nn as nn
from ase import Atoms

from omg.rl.residual_model import ResidualModel
from omg.rl.reward_functions import (
    VolumeReward,
    DensityReward,
    CompositeReward,
    RewardFunction,
)
from omg.rl.rl_config import RLConfig


# ============================================================================
# Mock components for testing
# ============================================================================


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, output_dim=10):
        super().__init__()
        self.linear = nn.Linear(5, output_dim)

    def forward(self, x, t):
        """Mock forward pass."""
        # Return dict with velocity fields
        batch_size = len(t)
        return {
            'pos_b': torch.randn(batch_size, 3),
            'cell_b': torch.randn(batch_size, 3, 3),
        }


def create_mock_structure(n_atoms=10):
    """Create a mock ASE structure for testing."""
    return Atoms(
        numbers=[6] * n_atoms,
        positions=torch.randn(n_atoms, 3).numpy(),
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )


# ============================================================================
# Tests for ResidualModel
# ============================================================================


class TestResidualModel:
    """Tests for ResidualModel class."""

    def test_initialization(self):
        """Test ResidualModel initialization."""
        base_arch = MockModel()
        residual = ResidualModel(
            base_architecture=base_arch,
            noise_scale=0.1,
            regularization_weight=0.01,
        )

        assert residual.noise_scale == 0.1
        assert residual.regularization_weight == 0.01

    def test_forward_training(self):
        """Test forward pass in training mode."""
        base_arch = MockModel()
        residual = ResidualModel(base_architecture=base_arch)
        residual.train()

        # Mock input
        x = {'pos': torch.randn(5, 3)}
        t = torch.randn(5)

        # Forward pass
        output, log_prob, mean = residual(x, t, return_log_prob=True)

        # Check outputs
        assert output is not None
        assert log_prob is not None
        assert mean is not None
        assert log_prob.shape == (5,)

    def test_forward_eval(self):
        """Test forward pass in eval mode (no noise)."""
        base_arch = MockModel()
        residual = ResidualModel(base_architecture=base_arch)
        residual.eval()

        # Mock input
        x = {'pos': torch.randn(5, 3)}
        t = torch.randn(5)

        # Forward pass
        output, log_prob, mean = residual(x, t, return_log_prob=False)

        # Check outputs
        assert output is not None
        assert log_prob is None  # No log prob in eval mode
        assert mean is not None

        # Output should equal mean in eval mode
        for key in output.keys():
            assert torch.allclose(output[key], mean[key])

    def test_noise_scale_update(self):
        """Test noise scale can be updated."""
        base_arch = MockModel()
        residual = ResidualModel(base_architecture=base_arch, noise_scale=0.1)

        residual.set_noise_scale(0.05)
        assert residual.noise_scale == 0.05

    def test_regularization_loss(self):
        """Test regularization loss computation."""
        base_arch = MockModel()
        residual = ResidualModel(
            base_architecture=base_arch,
            regularization_weight=0.1,
        )

        # Mock mean residuals
        mean_residuals = {
            'pos_b': torch.randn(5, 3),
            'cell_b': torch.randn(5, 3, 3),
        }

        reg_loss = residual.compute_regularization_loss(mean_residuals)

        assert reg_loss.item() >= 0
        assert reg_loss.requires_grad


# ============================================================================
# Tests for Reward Functions
# ============================================================================


class TestRewardFunctions:
    """Tests for reward function classes."""

    def test_volume_reward(self):
        """Test VolumeReward."""
        reward_fn = VolumeReward(scale=1.0)

        structures = [create_mock_structure() for _ in range(5)]
        rewards = reward_fn.compute(structures)

        assert rewards.shape == (5,)
        assert rewards.dtype == torch.float32
        assert reward_fn.is_differentiable()

    def test_density_reward(self):
        """Test DensityReward."""
        reward_fn = DensityReward(target_density=3.0, tolerance=0.5)

        structures = [create_mock_structure() for _ in range(5)]
        rewards = reward_fn.compute(structures)

        assert rewards.shape == (5,)
        assert (rewards >= 0).all()  # Rewards should be non-negative
        assert (rewards <= 1).all()  # Should be normalized to [0, 1]

    def test_composite_reward(self):
        """Test CompositeReward."""
        volume_reward = VolumeReward(scale=1.0)
        density_reward = DensityReward(target_density=3.0)

        composite = CompositeReward(
            reward_functions=[volume_reward, density_reward],
            weights=[0.6, 0.4],
        )

        structures = [create_mock_structure() for _ in range(5)]
        rewards = composite.compute(structures)

        assert rewards.shape == (5,)

    def test_composite_reward_validation(self):
        """Test CompositeReward weight validation."""
        volume_reward = VolumeReward()
        density_reward = DensityReward()

        # Weights don't sum to 1
        with pytest.raises(ValueError, match="must sum to 1"):
            CompositeReward(
                reward_functions=[volume_reward, density_reward],
                weights=[0.5, 0.6],
            )

        # Mismatched lengths
        with pytest.raises(ValueError, match="must match"):
            CompositeReward(
                reward_functions=[volume_reward],
                weights=[0.5, 0.5],
            )


# ============================================================================
# Tests for RLConfig
# ============================================================================


class TestRLConfig:
    """Tests for RLConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = RLConfig()

        assert config.algorithm == 'grpo'
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4

    def test_custom_config(self):
        """Test custom configuration."""
        config = RLConfig(
            algorithm='reinforce',
            batch_size=64,
            learning_rate=1e-3,
        )

        assert config.algorithm == 'reinforce'
        assert config.batch_size == 64
        assert config.learning_rate == 1e-3

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid algorithm
        with pytest.raises(ValueError, match="Unknown algorithm"):
            RLConfig(algorithm='invalid')

        # Invalid batch size
        with pytest.raises(ValueError, match="must be positive"):
            RLConfig(batch_size=0)

        # GRPO group size larger than batch size
        with pytest.raises(ValueError, match="must be <= batch_size"):
            RLConfig(batch_size=32, grpo_group_size=64)

        # Batch size not divisible by group size
        with pytest.raises(ValueError, match="must be divisible"):
            RLConfig(batch_size=32, grpo_group_size=7)

    def test_config_algorithms(self):
        """Test all supported algorithms."""
        for algo in ['reinforce', 'ppo', 'grpo']:
            config = RLConfig(algorithm=algo)
            assert config.algorithm == algo


# ============================================================================
# Integration tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_residual_model_with_reward(self):
        """Test using residual model with reward function."""
        # Create models
        base_arch = MockModel()
        residual = ResidualModel(base_architecture=base_arch)

        # Create reward
        reward_fn = VolumeReward()

        # Generate mock structures
        structures = [create_mock_structure() for _ in range(5)]

        # Compute rewards
        rewards = reward_fn.compute(structures)

        assert rewards.shape == (5,)

    def test_end_to_end_workflow(self):
        """Test a simple end-to-end workflow."""
        # 1. Create residual model
        base_arch = MockModel()
        residual = ResidualModel(base_architecture=base_arch, noise_scale=0.1)

        # 2. Create config
        config = RLConfig(
            algorithm='grpo',
            batch_size=16,
            grpo_group_size=4,
        )

        # 3. Create reward
        reward_fn = VolumeReward()

        # 4. Mock generation
        residual.train()
        x = {'pos': torch.randn(16, 3)}
        t = torch.randn(16)

        output, log_prob, mean = residual(x, t, return_log_prob=True)

        # 5. Mock reward computation
        structures = [create_mock_structure() for _ in range(16)]
        rewards = reward_fn.compute(structures)

        # 6. Mock loss computation (simplified)
        normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        policy_loss = -(normalized_rewards * log_prob).mean()

        assert policy_loss.requires_grad
        assert policy_loss.item() != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
