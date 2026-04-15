"""PyTorch layers with fixed binary connectivity masks."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Module):
    """Linear layer with a fixed binary connectivity mask.

    The mask is registered as a non-trainable buffer and applied
    element-wise during the forward pass: W_eff = W * M.
    Gradients at masked positions are exactly zero.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        mask: Binary tensor of shape (out_features, in_features).
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
        bias: bool = True
    ) -> None:
        super().__init__()

        if mask.shape != (out_features, in_features):
            raise ValueError(
                f"Mask shape {mask.shape} does not match "
                f"({out_features}, {in_features})"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.register_buffer("mask", mask.float())

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_buffer("bias", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=nn.init.calculate_gain("linear"))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    @property
    def density(self) -> float:
        total = self.mask.numel()
        return float(self.mask.sum().item() / total) if total > 0 else 0.0

    @property
    def num_active_params(self) -> int:
        return int(self.mask.sum().item())

    @property
    def num_total_params(self) -> int:
        return self.weight.numel()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, density={self.density:.4f}"
        )

    @classmethod
    def from_topology(
        cls,
        adjacency_matrix: torch.Tensor,
        in_features: int,
        out_features: int,
        bias: bool = True
    ) -> "MaskedLinear":
        """Create a MaskedLinear from a binary adjacency matrix."""
        if adjacency_matrix.shape != (out_features, in_features):
            raise ValueError(
                f"Adjacency matrix shape {adjacency_matrix.shape} does not match "
                f"({out_features}, {in_features})"
            )
        mask = (adjacency_matrix != 0).float()
        return cls(in_features, out_features, mask, bias)


class MaskedRNNCell(nn.Module):
    """RNN cell with masked input-to-hidden and hidden-to-hidden weights.

    Computes: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
    Both weight matrices are masked by fixed binary tensors.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        mask_ih: Binary mask of shape (hidden_size, input_size).
        mask_hh: Binary mask of shape (hidden_size, hidden_size).
        bias: Whether to include bias terms.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mask_ih: torch.Tensor,
        mask_hh: torch.Tensor,
        bias: bool = True
    ) -> None:
        super().__init__()

        if mask_ih.shape != (hidden_size, input_size):
            raise ValueError(
                f"mask_ih shape {mask_ih.shape} does not match ({hidden_size}, {input_size})"
            )
        if mask_hh.shape != (hidden_size, hidden_size):
            raise ValueError(
                f"mask_hh shape {mask_hh.shape} does not match ({hidden_size}, {hidden_size})"
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty((hidden_size, input_size)))
        self.weight_hh = nn.Parameter(torch.empty((hidden_size, hidden_size)))
        self.register_buffer("mask_ih", mask_ih.float())
        self.register_buffer("mask_hh", mask_hh.float())

        if bias:
            self.bias_ih = nn.Parameter(torch.empty(hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_buffer("bias_ih", None)
            self.register_buffer("bias_hh", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std = 1.0 / (self.hidden_size ** 0.5)
        for w in [self.weight_ih, self.weight_hh]:
            nn.init.uniform_(w, -std, std)
        if self.bias_ih is not None:
            nn.init.uniform_(self.bias_ih, -std, std)
            nn.init.uniform_(self.bias_hh, -std, std)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gi = F.linear(x, self.weight_ih * self.mask_ih, self.bias_ih)
        gh = F.linear(h, self.weight_hh * self.mask_hh, self.bias_hh)
        return torch.tanh(gi + gh)

    @property
    def density_ih(self) -> float:
        total = self.mask_ih.numel()
        return float(self.mask_ih.sum().item() / total) if total > 0 else 0.0

    @property
    def density_hh(self) -> float:
        total = self.mask_hh.numel()
        return float(self.mask_hh.sum().item() / total) if total > 0 else 0.0

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"density_ih={self.density_ih:.4f}, density_hh={self.density_hh:.4f}"
        )


class MaskedRNN(nn.Module):
    """Multi-layer RNN with masked recurrent connections.

    Wraps MaskedRNNCells to process full sequences. Returns outputs at
    every timestep and the final hidden state.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        mask_ih: Binary mask for input-to-hidden connections (layer 0).
        mask_hh: Binary mask for hidden-to-hidden connections.
        num_layers: Number of stacked RNN layers.
        bias: Whether to include bias terms.
        batch_first: If True, input/output tensors are (batch, seq, features).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mask_ih: torch.Tensor,
        mask_hh: torch.Tensor,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cells.append(MaskedRNNCell(
                input_size=layer_input_size,
                hidden_size=hidden_size,
                mask_ih=mask_ih if i == 0 else mask_hh,
                mask_hh=mask_hh,
                bias=bias,
            ))

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()

        if h0 is None:
            h0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                dtype=x.dtype, device=x.device
            )

        output = []
        h = [h0[i] for i in range(self.num_layers)]

        for t in range(seq_len):
            x_t = x[t]
            for i, cell in enumerate(self.cells):
                h[i] = cell(x_t, h[i])
                x_t = h[i]
            output.append(h[-1])

        output = torch.stack(output, dim=0)
        h_n = torch.stack(h, dim=0)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, h_n

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, batch_first={self.batch_first}"
        )
