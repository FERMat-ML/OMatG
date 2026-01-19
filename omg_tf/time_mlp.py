from enum import auto, Enum
from typing import Optional, Sequence
import torch
import torch.nn as nn


class GlobalInvariant(Enum):
    """
    Enumeration of global invariants that can be computed from a structure.
    """
    N_ATOMS = auto()
    """Number of atoms in the structure. Captures system size."""

    LOG_VOLUME_PER_ATOM = auto()
    """Log volume per atom: log(volume / n_atoms). Log-scale for stable MLP input range."""

    CELL_ANISOTROPY = auto()
    """Ratio of min to max cell vector length. Captures cell shape (1 = cubic, < 1 = elongated)."""

    MIN_NN_DISTANCE = auto()
    """Minimum nearest-neighbor distance. Captures closest atomic approach."""


class TimeMLP(nn.Module):
    @staticmethod
    def _build_mlp(hidden_dimension: int, number_hidden_layers: int, input_dimension: int, output_dimension: int,
                   initialize_zeros: bool) -> nn.Sequential:
        assert input_dimension > 0
        assert output_dimension > 0
        assert hidden_dimension > 0
        assert number_hidden_layers >= 0
        layers: list[nn.Module] = [nn.Linear(input_dimension, hidden_dimension), nn.SiLU()]
        for _ in range(number_hidden_layers):
            layers.append(nn.Linear(hidden_dimension, hidden_dimension))
            layers.append(nn.SiLU())
        final_layer = nn.Linear(hidden_dimension, output_dimension)
        if initialize_zeros:
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        return nn.Sequential(*layers)

    def __init__(self, hidden_dimension: int = 64, number_hidden_layers: int = 2, initialize_zeros: bool = True,
                 shared_trunk: bool = True, invariants: Sequence[str] = ()) -> None:
        """
        Initialize a TimeMLP for predicting scale factors.

        :param hidden_dimension: Hidden dimension of the MLP.
        :param number_hidden_layers: Number of hidden layers.
        :param initialize_zeros: If True, zero-initialize the output layer(s).
        :param shared_trunk: If True, use a shared trunk with separate heads for pos and cell.
        :param invariants: List of global invariants to condition on.
        """
        super().__init__()
        if not hidden_dimension > 0:
            raise ValueError("Hidden dimension must be positive.")
        if not number_hidden_layers >= 0:
            raise ValueError("Number of hidden layers must be non-negative.")

        # Parse invariants.
        try:
            self.invariants = [GlobalInvariant[inv.upper()] for inv in invariants]
        except KeyError as e:
            raise ValueError(f"Invalid invariant in {invariants}. Valid options are: "
                             f"{', '.join([inv.name.lower() for inv in GlobalInvariant])}.") from e

        input_dimension = 1 + len(self.invariants)  # Start with 1 for time.

        self.shared_trunk = shared_trunk
        if self.shared_trunk:
            # Zero-initialize the heads and not the trunk.
            self._trunk = self._build_mlp(hidden_dimension=hidden_dimension, number_hidden_layers=number_hidden_layers,
                                          input_dimension=input_dimension, output_dimension=hidden_dimension,
                                          initialize_zeros=False)
            self._pos_head = nn.Linear(hidden_dimension, 1)
            self._cell_head = nn.Linear(hidden_dimension, 1)
            if initialize_zeros:
                nn.init.zeros_(self._pos_head.weight)
                nn.init.zeros_(self._pos_head.bias)
                nn.init.zeros_(self._cell_head.weight)
                nn.init.zeros_(self._cell_head.bias)
        else:
            self._pos_mlp = self._build_mlp(hidden_dimension=hidden_dimension,
                                            number_hidden_layers=number_hidden_layers,
                                            input_dimension=input_dimension, output_dimension=1,
                                            initialize_zeros=initialize_zeros)
            self._cell_mlp = self._build_mlp(hidden_dimension=hidden_dimension,
                                             number_hidden_layers=number_hidden_layers,
                                             input_dimension=input_dimension, output_dimension=1,
                                             initialize_zeros=initialize_zeros)

    def _compute_invariants(self, x) -> Optional[torch.Tensor]:
        """
        Compute global invariants from the structure.

        :param x: OMGData batch with pos, cell, n_atoms, batch attributes.
        :return: Tensor of shape (batch_size, num_invariants) or None if no invariants.
        """
        invariant_list = []

        for invariant in self.invariants:
            if invariant == GlobalInvariant.N_ATOMS:
                # Normalize n_atoms to a reasonable scale (e.g., divide by 20 to center around 1).
                # TODO: NORMALIZATION FACTOR SHOULD BE PARAMETER
                n_atoms = x.n_atoms.float().unsqueeze(-1)  # Shape: (batch_size, 1).
                invariant_list.append(n_atoms / 20.0)
            elif invariant == GlobalInvariant.LOG_VOLUME_PER_ATOM:
                # Use slogdet for numerical stability (avoids computing det then log).
                _, logabsdet = torch.linalg.slogdet(x.cell)  # Both shape (batch_size,).
                log_volume = logabsdet.unsqueeze(-1)  # Shape: (batch_size, 1).
                log_n_atoms = torch.log(x.n_atoms.float()).unsqueeze(-1)  # Shape: (batch_size, 1).
                invariant_list.append(log_volume - log_n_atoms)
            elif invariant == GlobalInvariant.CELL_ANISOTROPY:
                cell_lengths = torch.norm(x.cell, dim=-1)  # Shape: (batch_size, 3).
                min_cell_length = torch.min(cell_lengths, dim=-1).values.unsqueeze(-1)  # Shape: (batch_size, 1).
                max_cell_length = torch.max(cell_lengths, dim=-1).values.unsqueeze(-1)  # Shape: (batch_size, 1).
                invariant_list.append(min_cell_length / max_cell_length)
            elif invariant == GlobalInvariant.MIN_NN_DISTANCE:
                # Compute minimum nearest-neighbor distance per structure.
                # This is more expensive as it requires computing pairwise distances.
                min_nn_distance = self._compute_min_nn_distance(x).unsqueeze(-1)  # Shape: (batch_size, 1).
                invariant_list.append(min_nn_distance)

        # Concatenate all invariants. Shape: (batch_size, num_invariants).
        return torch.cat(invariant_list, dim=-1)

    def _compute_min_nn_distance(self, x) -> torch.Tensor:
        """
        Compute the minimum nearest-neighbor distance for each structure in the batch,
        respecting periodic boundary conditions via minimum image convention.

        Vectorized implementation that pads structures to max_n_atoms and processes
        the entire batch in parallel.

        :param x: OMGData batch.
        :return: Tensor of shape (batch_size,) with min NN distance per structure.
        """
        assert torch.all(x.pos_is_fractional)

        batch_size = x.cell.shape[0]
        max_n = x.n_atoms.max().item()

        # Handle edge case where all structures have < 2 atoms.
        if max_n < 2:
            return torch.zeros(batch_size, device=x.pos.device, dtype=x.pos.dtype)

        # Pad positions to (batch_size, max_n, 3) - fully vectorized.
        # Compute local index of each atom within its structure.
        n_atoms_cumsum = x.n_atoms.cumsum(0)
        offsets = torch.cat([torch.zeros(1, device=x.pos.device, dtype=n_atoms_cumsum.dtype), n_atoms_cumsum[:-1]])
        local_idx = torch.arange(x.pos.shape[0], device=x.pos.device) - offsets[x.batch]
        # Scatter positions into padded tensor.
        padded_frac = torch.zeros(batch_size, max_n, 3, device=x.pos.device, dtype=x.pos.dtype)
        padded_frac[x.batch, local_idx] = x.pos

        # Pairwise fractional differences: (batch_size, max_n, max_n, 3).
        frac_diff = padded_frac.unsqueeze(2) - padded_frac.unsqueeze(1)
        frac_diff = frac_diff - torch.round(frac_diff)  # Wrap to [-0.5, 0.5).

        # Convert to Cartesian: (batch_size, max_n, max_n, 3).
        # cell is (batch_size, 3, 3), need to apply per-structure.
        cart_diff = torch.einsum('bijk,bkl->bijl', frac_diff, x.cell)

        # Compute distances: (batch_size, max_n, max_n).
        dist = torch.linalg.norm(cart_diff, dim=-1)

        # Create mask for valid atom pairs: (batch_size, max_n, max_n).
        # Valid if both i < n_atoms and j < n_atoms and i != j.
        atom_idx = torch.arange(max_n, device=x.pos.device)  # Shape (max_n,)
        valid_i = atom_idx.unsqueeze(0) < x.n_atoms.unsqueeze(1)  # (batch_size, max_n).
        valid_pair = valid_i.unsqueeze(2) & valid_i.unsqueeze(1)  # (batch_size, max_n, max_n).
        not_self = ~torch.eye(max_n, dtype=torch.bool, device=x.pos.device).unsqueeze(0)  # (1, max_n, max_n)
        valid_pair = valid_pair & not_self

        # Set invalid pairs to large value.
        dist = torch.where(valid_pair, dist, torch.tensor(float('inf'), device=dist.device, dtype=dist.dtype))

        # Min over all pairs per structure.
        min_distances = dist.reshape(batch_size, -1).min(dim=-1).values

        # Handle structures with < 2 atoms (no valid pairs -> inf -> set to 0).
        min_distances = torch.where(x.n_atoms < 2, torch.zeros_like(min_distances), min_distances)

        return min_distances

    def forward(self, x, t):
        """
        Forward pass.

        :param x: OMGData batch (used to compute invariants if enabled, otherwise ignored).
        :param t: Time tensor of shape (batch_size,).
        :return: Dictionary with "pos_s" and "cell_s" scale factors.
        """
        # Mirror the input of OMG Model class.
        t_in = t.unsqueeze(-1)  # Shape: (batch_size, 1).
        if len(self.invariants) > 0:
            invariants = self._compute_invariants(x)  # Shape: (batch_size, num_invariants).
            mlp_input = torch.cat([t_in, invariants], dim=-1)  # Shape: (batch_size, 1 + num_invariants).
        else:
            mlp_input = t_in  # Shape: (batch_size, 1).

        if self.shared_trunk:
            features = self._trunk(mlp_input)
            pos_s = self._pos_head(features).squeeze(-1)
            cell_s = self._cell_head(features).squeeze(-1)
        else:
            pos_s = self._pos_mlp(mlp_input).squeeze(-1)
            cell_s = self._cell_mlp(mlp_input).squeeze(-1)

        return {"pos_s": pos_s, "cell_s": cell_s}
