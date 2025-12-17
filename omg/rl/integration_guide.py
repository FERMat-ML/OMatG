"""
Integration guide: How to use trained residual models during generation.

This module shows how to modify the integration process to incorporate residual velocities.
"""

import torch
from typing import Callable, Union, Tuple, List
from torch_geometric.data import Data
from tqdm import trange

from omg.globals import SMALL_TIME, BIG_TIME
from omg.utils import reshape_t, DataField
from omg.rl.residual_model import ResidualModel


class ResidualIntegrator:
    """
    Modified integrator that combines base model with residual model.

    This class wraps the standard StochasticInterpolants integration
    and adds residual velocity corrections.
    """

    def __init__(
        self,
        base_model,
        residual_model: ResidualModel,
        stochastic_interpolants,
        use_residual: bool = True,
    ):
        """
        Constructor.

        :param base_model: Frozen base flow matching model
        :param residual_model: Trained residual model
        :param stochastic_interpolants: StochasticInterpolants instance
        :param use_residual: Whether to apply residual (can disable for comparison)
        """
        self.base_model = base_model
        self.residual_model = residual_model
        self.si = stochastic_interpolants
        self.use_residual = use_residual

        # Set models to eval mode
        self.base_model.eval()
        self.residual_model.eval()

    def integrate(
        self,
        x_0: Data,
        save_intermediate: bool = False,
    ) -> Union[Data, Tuple[Data, List[Data]]]:
        """
        Integrate using base + residual velocities.

        This is a modified version of StochasticInterpolants.integrate()
        that incorporates residual corrections.

        :param x_0: Initial structures
        :param save_intermediate: Whether to save intermediate steps

        :return: Final structures (and optionally intermediate steps)
        """
        times = torch.linspace(
            SMALL_TIME,
            BIG_TIME,
            self.si._integration_time_steps,
            device=x_0.pos.device
        )

        # Clone data for integration
        data_fields = [data_field.name for data_field in self.si._data_fields]
        x_t = x_0.clone(*data_fields)
        new_x_t = x_0.clone(*data_fields)
        x_t_dict = x_t.to_dict()
        new_x_t_dict = new_x_t.to_dict()

        if save_intermediate:
            inter_list = [x_t.clone()]
        else:
            inter_list = None

        # Integration loop
        for t_index in trange(
            1,
            len(times),
            desc='Integrating with residual',
            disable=not self.si._enable_progress_bar
        ):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]

            # Get batch size
            batch_size = len(x_t.n_atoms)
            t_batch = t.repeat(batch_size)

            # === Get velocities from base model ===
            with torch.no_grad():
                base_output = self.base_model(x_t, t_batch)

            # === Get velocities from residual model (if enabled) ===
            if self.use_residual:
                with torch.no_grad():
                    # residual_model.forward returns (output, log_prob, mean)
                    # During eval, output == mean and is deterministic
                    residual_output, _, _ = self.residual_model(
                        x_t,
                        t_batch,
                        return_log_prob=False
                    )
            else:
                residual_output = None

            # === Integrate each data field ===
            for stochastic_interpolant, data_field in zip(
                self.si._stochastic_interpolants,
                self.si._data_fields
            ):
                b_data_field = data_field.name + "_b"

                # Get base velocity
                if b_data_field not in base_output:
                    continue

                base_velocity = base_output[b_data_field]

                # Add residual if enabled
                if self.use_residual and residual_output is not None:
                    if b_data_field in residual_output:
                        total_velocity = base_velocity + residual_output[b_data_field]
                    else:
                        total_velocity = base_velocity
                else:
                    total_velocity = base_velocity

                # Reshape time
                reshaped_t = reshape_t(t_batch, x_t.n_atoms, data_field)

                # Integrate this field
                # This uses the stochastic interpolant's integration logic
                new_x = stochastic_interpolant.integrate_step(
                    x_t_dict[data_field.name],
                    total_velocity,
                    reshaped_t,
                    dt,
                    x_t.batch if data_field != DataField.cell else torch.arange(len(x_t.n_atoms))
                )

                # Update
                new_x_t_dict[data_field.name].copy_(new_x)

            # Copy new_x_t to x_t for next iteration
            for data_field in self.si._data_fields:
                x_t_dict[data_field.name].copy_(new_x_t_dict[data_field.name])

            if save_intermediate:
                inter_list.append(x_t.clone())

        if save_intermediate:
            return x_t, inter_list
        else:
            return x_t


# ============================================================================
# Example usage
# ============================================================================


def example_generation_with_residual():
    """
    Example: Generate structures using base + trained residual model.
    """
    import torch
    from pathlib import Path

    print("Loading models...")

    # 1. Load base model
    from omg.omg_lightning import OMGLightning
    base_lightning = OMGLightning.load_from_checkpoint("base_model.ckpt")

    # 2. Load trained residual model
    # First recreate architecture
    residual_architecture = create_your_residual_architecture()

    residual_model = ResidualModel(
        base_architecture=residual_architecture,
        noise_scale=0.1,  # Unused during inference
        regularization_weight=0.01,
    )

    # Load weights
    residual_model.load_state_dict(torch.load("residual_model.pt"))
    residual_model.eval()

    print("Creating integrator...")

    # 3. Create residual integrator
    integrator = ResidualIntegrator(
        base_model=base_lightning.model,
        residual_model=residual_model,
        stochastic_interpolants=base_lightning.si,
        use_residual=True,
    )

    # 4. Sample initial structures
    # You need some data to condition on (e.g., compositions)
    batch_data = get_your_batch_data()
    x_0 = base_lightning.sampler.sample_p_0(batch_data)

    print("Generating structures...")

    # 5. Generate
    x_final = integrator.integrate(x_0, save_intermediate=False)

    print(f"Generated {len(x_final.n_atoms)} structures!")

    # 6. Convert to ASE and save
    from ase import Atoms
    structures = []
    for i in range(len(x_final.n_atoms)):
        lower, upper = x_final.ptr[i], x_final.ptr[i + 1]
        atoms = Atoms(
            numbers=x_final.species[lower:upper],
            scaled_positions=x_final.pos[lower:upper, :],
            cell=x_final.cell[i, :, :],
            pbc=(1, 1, 1)
        )
        structures.append(atoms)

    # Save or analyze structures...
    print("Done!")

    return structures


def example_comparison():
    """
    Example: Compare generation with and without residual.
    """
    print("Comparing base model vs base + residual...")

    # Setup (same as above)
    base_lightning = OMGLightning.load_from_checkpoint("base_model.ckpt")
    residual_model = load_your_residual_model()

    # Create integrators
    integrator_base_only = ResidualIntegrator(
        base_model=base_lightning.model,
        residual_model=residual_model,
        stochastic_interpolants=base_lightning.si,
        use_residual=False,  # Disable residual
    )

    integrator_with_residual = ResidualIntegrator(
        base_model=base_lightning.model,
        residual_model=residual_model,
        stochastic_interpolants=base_lightning.si,
        use_residual=True,  # Enable residual
    )

    # Sample same initial structures
    x_0 = sample_initial_structures()

    # Generate with base only
    print("Generating with base model only...")
    x_base = integrator_base_only.integrate(x_0.clone())

    # Generate with base + residual
    print("Generating with base + residual...")
    x_residual = integrator_with_residual.integrate(x_0.clone())

    # Compare (e.g., compute rewards)
    from omg.rl.reward_functions import VolumeReward
    reward_fn = VolumeReward()

    structures_base = convert_to_ase(x_base)
    structures_residual = convert_to_ase(x_residual)

    rewards_base = reward_fn.compute(structures_base)
    rewards_residual = reward_fn.compute(structures_residual)

    print(f"Base model - Mean reward: {rewards_base.mean():.4f}")
    print(f"With residual - Mean reward: {rewards_residual.mean():.4f}")
    print(f"Improvement: {(rewards_residual.mean() - rewards_base.mean()):.4f}")


if __name__ == "__main__":
    print("""
    This file shows how to integrate residual models into generation.

    Key steps:
    1. Load base model (frozen)
    2. Load trained residual model
    3. Use ResidualIntegrator instead of standard integration
    4. Generate structures with combined velocities

    See example functions for complete workflows.
    """)
