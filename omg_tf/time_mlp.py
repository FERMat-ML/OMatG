import torch.nn as nn


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
                 shared_trunk: bool = True) -> None:
        super().__init__()
        if not hidden_dimension > 0:
            raise ValueError("Hidden dimension must be positive.")
        if not number_hidden_layers >= 0:
            raise ValueError("Number of hidden layers must be non-negative.")
        self.shared_trunk = shared_trunk
        if self.shared_trunk:
            # Zero-initialize the heads and not the trunk.
            self._trunk = self._build_mlp(hidden_dimension=hidden_dimension, number_hidden_layers=number_hidden_layers,
                                          input_dimension=1, output_dimension=hidden_dimension, initialize_zeros=False)
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

    def forward(self, x, t):
        # Mirror the input of OMG Model class.
        t_in = t.unsqueeze(-1)
        if self.shared_trunk:
            features = self._trunk(t_in)
            pos_s = self._pos_head(features).squeeze(-1)
            cell_s = self._cell_head(features).squeeze(-1)
        else:
            pos_s = self._pos_mlp(t_in).squeeze(-1)
            cell_s = self._cell_mlp(t_in).squeeze(-1)
        return {"pos_s": pos_s, "cell_s": cell_s}


if __name__ == '__main__':
    import torch

    model = TimeMLP(initialize_zeros=True, shared_trunk=True)
    t = torch.tensor([0.0, 0.5, 1.0])
    print(model(None, t))
