import torch
import torch.nn as nn


class ScaleMLP(nn.Module):
    """
    Multi-layer perceptron (MLP) for predicting time-dependent scale factors for positions and cell parameters in the
    OMatG-IRL framework.

    The MLP takes time t as input and outputs scale factors pos_s and cell_s. It can be configured to have a shared
    trunk with separate heads for pos and cell, or to have separate MLPs for each. The output layer(s) can be
    optionally initialized to zero to start with no scaling at the beginning of training.

    :param hidden_dimension:
        Hidden dimension of the MLP.
        Must be positive.
        Defaults to 64.
    :type hidden_dimension: int
    :param number_hidden_layers:
        Number of hidden layers in the MLP.
        Must be non-negative.
        Defaults to 2.
    :type number_hidden_layers: int
    :param initialize_zeros:
        If True, zero-initialize the output layer(s) to start with no scaling at the beginning of training.
    :type initialize_zeros: bool
    :param shared_trunk:
        If True, use a shared trunk with separate heads for pos and cell. If False, use separate MLPs for pos and cell.
        Defaults to True.
    :type shared_trunk: bool

    :raises ValueError:
        If hidden_dimension is not positive.
        If number_hidden_layers is negative.
    """

    def __init__(self, hidden_dimension: int = 64, number_hidden_layers: int = 2, initialize_zeros: bool = True,
                 shared_trunk: bool = True) -> None:
        """Constructor for TimeMLP."""
        super().__init__()
        if not hidden_dimension > 0:
            raise ValueError("Hidden dimension must be positive.")
        if not number_hidden_layers >= 0:
            raise ValueError("Number of hidden layers must be non-negative.")

        self.shared_trunk = shared_trunk
        if self.shared_trunk:
            # Zero-initialize the heads and not the trunk.
            self._trunk = self._build_mlp(hidden_dimension=hidden_dimension, number_hidden_layers=number_hidden_layers,
                                          input_dimension=1, output_dimension=hidden_dimension,
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
                                            input_dimension=1, output_dimension=1,
                                            initialize_zeros=initialize_zeros)
            self._cell_mlp = self._build_mlp(hidden_dimension=hidden_dimension,
                                             number_hidden_layers=number_hidden_layers,
                                             input_dimension=1, output_dimension=1,
                                             initialize_zeros=initialize_zeros)

    @staticmethod
    def _build_mlp(hidden_dimension: int, number_hidden_layers: int, input_dimension: int, output_dimension: int,
                   initialize_zeros: bool) -> nn.Sequential:
        """Helper function to build an MLP with the specified configuration."""
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

    def forward(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass of the TimeMLP.

        This method takes in a batch of time values t of shape (batch_size,) and outputs a dictionary containing the
        predicted scale factors for positions (pos_s) and cell parameters (cell_s). The output shapes are (batch_size,).

        :param t:
            Batch of time values of shape (batch_size,).
        :type t: torch.Tensor

        :return:
            Dictionary with keys "pos_s" and "cell_s" containing the predicted scale factors for
            positions and cell parameters, respectively. Both tensors have shape (batch_size,).
        :rtype: dict[str, torch.Tensor]
        """
        assert t.ndim == 1
        t_in = t.unsqueeze(-1)  # Shape: (batch_size, 1).

        if self.shared_trunk:
            features = self._trunk(t_in)
            pos_s = self._pos_head(features).squeeze(-1)
            cell_s = self._cell_head(features).squeeze(-1)
        else:
            pos_s = self._pos_mlp(t_in).squeeze(-1)
            cell_s = self._cell_mlp(t_in).squeeze(-1)

        return {"pos_s": pos_s, "cell_s": cell_s}
