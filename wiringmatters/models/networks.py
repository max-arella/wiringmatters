"""BioMLP and BioRNN: neural network architectures with fixed connectivity masks."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from wiringmatters.models.layers import MaskedLinear, MaskedRNN


class BioMLP(nn.Module):
    """Multi-layer perceptron with optional biological connectivity masks.

    Stacks MaskedLinear layers with non-linearities. Masks are applied
    to square (hidden-to-hidden) layers only; rectangular layers are dense.

    Args:
        layer_sizes: Layer dimensions, e.g. [64, 448, 448, 10].
        masks: One binary mask per layer, or None for a dense layer.
               Pass None (not a list) for a fully dense network.
        activation: Activation function name ('relu', 'tanh', 'elu', etc.).
        bias: Whether to include bias in each layer.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        masks: Optional[List[torch.Tensor]] = None,
        activation: str = "relu",
        bias: bool = True
    ) -> None:
        super().__init__()

        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements")

        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.activation_fn = self._get_activation(activation)

        num_layers = len(layer_sizes) - 1
        if masks is not None:
            if len(masks) != num_layers:
                raise ValueError(
                    f"Expected {num_layers} masks but got {len(masks)}. "
                    "Use None for dense layers."
                )
            for i, mask in enumerate(masks):
                if mask is None:
                    continue
                expected = (layer_sizes[i + 1], layer_sizes[i])
                if mask.shape != expected:
                    raise ValueError(
                        f"Mask {i} has shape {mask.shape}, expected {expected}"
                    )

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_f, out_f = layer_sizes[i], layer_sizes[i + 1]
            m = masks[i] if (masks is not None and masks[i] is not None) \
                else torch.ones((out_f, in_f))
            self.layers.append(MaskedLinear(in_f, out_f, m, bias))

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        options = {
            "relu": nn.ReLU(), "tanh": nn.Tanh(), "elu": nn.ELU(),
            "sigmoid": nn.Sigmoid(), "gelu": nn.GELU(), "leaky_relu": nn.LeakyReLU(),
        }
        if name not in options:
            raise ValueError(f"Unknown activation '{name}'. Options: {list(options)}")
        return options[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation_fn(x)
        return x

    def get_network_density(self) -> float:
        return sum(l.density for l in self.layers) / len(self.layers)

    def get_num_active_params(self) -> int:
        return sum(l.num_active_params for l in self.layers)

    def get_num_total_params(self) -> int:
        return sum(l.num_total_params for l in self.layers)

    def get_num_all_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"layer_sizes={self.layer_sizes}, activation={self.activation_name}, "
            f"density={self.get_network_density():.4f}"
        )

    @classmethod
    def from_connectome(
        cls,
        adjacency_matrix: torch.Tensor,
        layer_sizes: List[int],
        mask_type: str = "bio",
        activation: str = "relu",
        bias: bool = True
    ) -> "BioMLP":
        """Create BioMLP from a connectome adjacency matrix.

        Args:
            adjacency_matrix: Connectome adjacency matrix (numpy or tensor).
            layer_sizes: Network layer dimensions.
            mask_type: 'bio', 'uniform_sparse', or 'dense'.
            activation: Activation function name.
            bias: Whether to include bias.
        """
        if mask_type not in ("bio", "uniform_sparse", "dense"):
            raise ValueError(
                f"mask_type must be 'bio', 'uniform_sparse', or 'dense', got '{mask_type}'"
            )

        import numpy as np
        from wiringmatters.topology.masks import bio_mask as _bio_mask, uniform_sparse_mask

        adj_np = adjacency_matrix.numpy() if isinstance(adjacency_matrix, torch.Tensor) \
            else np.array(adjacency_matrix, dtype=np.float32)

        num_layers = len(layer_sizes) - 1

        if mask_type == "bio":
            masks = []
            for i in range(num_layers):
                out_f, in_f = layer_sizes[i + 1], layer_sizes[i]
                masks.append(_bio_mask(adj_np, target_shape=out_f) if out_f == in_f else None)
            square = [m for m in masks if m is not None]
            bio_density = float(square[0].float().mean().item()) if square else 0.5

        elif mask_type == "uniform_sparse":
            ref = _bio_mask(adj_np, target_shape=layer_sizes[1])
            density = float(ref.float().mean().item())
            masks = [
                uniform_sparse_mask((layer_sizes[i+1], layer_sizes[i]), density=density)
                for i in range(num_layers)
            ]

        else:
            masks = None

        return cls(layer_sizes=layer_sizes, masks=masks, activation=activation, bias=bias)


class BioRNN(nn.Module):
    """RNN with masked recurrent connections and a linear readout.

    The biological mask constrains the hidden-to-hidden weight matrix.
    The input-to-hidden projection is always dense.

    Args:
        input_size: Input feature dimension per timestep.
        hidden_size: Hidden state dimension.
        output_size: Output feature dimension.
        mask_ih: Binary mask for input-to-hidden weights.
        mask_hh: Binary mask for hidden-to-hidden weights.
        num_layers: Number of stacked RNN layers.
        bias: Whether to include bias terms.
        batch_first: If True, input/output are (batch, seq, features).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        mask_ih: torch.Tensor,
        mask_hh: torch.Tensor,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.rnn = MaskedRNN(
            input_size, hidden_size, mask_ih, mask_hh,
            num_layers=num_layers, bias=bias, batch_first=batch_first
        )
        self.fc = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        rnn_out, h_n = self.rnn(x, h0)

        if self.batch_first:
            b, s, h = rnn_out.shape
            output = self.fc(rnn_out.reshape(-1, h)).reshape(b, s, self.output_size)
        else:
            s, b, h = rnn_out.shape
            output = self.fc(rnn_out.reshape(-1, h)).reshape(s, b, self.output_size)

        return (output, h_n) if return_hidden else output

    def get_sequence_output(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return the readout at the final timestep (for sequence classification)."""
        output, _ = self(x, h0, return_hidden=True)
        return output[:, -1, :] if self.batch_first else output[-1, :, :]

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"output_size={self.output_size}, batch_first={self.batch_first}"
        )
