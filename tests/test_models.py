"""
Tests for wiringmatters.models
--------------------------------
Run with: pytest tests/test_models.py -v
"""

import torch
import numpy as np
import pytest

from wiringmatters.models import MaskedLinear, MaskedRNN, BioMLP, BioRNN
from wiringmatters.topology import bio_mask, uniform_sparse_mask, dense_mask


# Fixtures

@pytest.fixture
def simple_mask() -> torch.Tensor:
    """A simple 8x8 binary mask with ~50% connections."""
    torch.manual_seed(0)
    mask = (torch.rand(8, 8) > 0.5).float()
    return mask


@pytest.fixture
def full_mask() -> torch.Tensor:
    return dense_mask((8, 8))


@pytest.fixture
def batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 8)  # batch_size=4, input=8


# MaskedLinear tests

class TestMaskedLinear:

    def test_output_shape(self, simple_mask, batch):
        layer = MaskedLinear(in_features=8, out_features=8, mask=simple_mask)
        out = layer(batch)
        assert out.shape == (4, 8)

    def test_mask_is_not_parameter(self, simple_mask):
        layer = MaskedLinear(in_features=8, out_features=8, mask=simple_mask)
        param_names = [n for n, _ in layer.named_parameters()]
        assert "mask" not in param_names

    def test_mask_is_buffer(self, simple_mask):
        layer = MaskedLinear(in_features=8, out_features=8, mask=simple_mask)
        buffer_names = [n for n, _ in layer.named_buffers()]
        assert "mask" in buffer_names

    def test_mask_enforced_at_forward(self, batch):
        """Outputs at masked positions must be zero when inputs are positive."""
        mask = torch.zeros(8, 8)
        mask[0, 0] = 1.0  # only connection [0,0] active
        layer = MaskedLinear(in_features=8, out_features=8, mask=mask, bias=False)
        # Force weight to ones for deterministic test
        with torch.no_grad():
            layer.weight.fill_(1.0)
        out = layer(batch)
        # Only output[0] should be nonzero (receives input[0] only)
        # All other outputs should be 0 (no active connections)
        for i in range(1, 8):
            assert out[:, i].abs().sum().item() == pytest.approx(0.0)

    def test_dense_mask_matches_linear(self, full_mask, batch):
        """MaskedLinear with all-ones mask must behave like nn.Linear."""
        torch.manual_seed(1)
        masked = MaskedLinear(in_features=8, out_features=8, mask=full_mask, bias=False)
        linear = torch.nn.Linear(in_features=8, out_features=8, bias=False)
        # Copy weights
        with torch.no_grad():
            linear.weight.copy_(masked.weight)
        assert torch.allclose(masked(batch), linear(batch), atol=1e-6)

    def test_density_property(self, simple_mask):
        layer = MaskedLinear(in_features=8, out_features=8, mask=simple_mask)
        assert 0.0 <= layer.density <= 1.0

    def test_gradients_only_for_active_weights(self, simple_mask, batch):
        """Gradients must be zero for weights at masked positions."""
        layer = MaskedLinear(in_features=8, out_features=8, mask=simple_mask)
        out = layer(batch).sum()
        out.backward()
        # Gradient at masked positions should be 0
        masked_grads = layer.weight.grad * (1 - simple_mask)
        assert masked_grads.abs().max().item() == pytest.approx(0.0, abs=1e-6)

    def test_device_transfer(self, simple_mask, batch):
        """Mask must follow the model to the same device via .to()."""
        layer = MaskedLinear(in_features=8, out_features=8, mask=simple_mask)
        # CPU transfer (no-op but checks the API)
        layer = layer.to("cpu")
        assert layer.mask.device.type == "cpu"


# BioMLP tests

class TestBioMLP:

    def test_dense_forward(self):
        model = BioMLP(layer_sizes=[8, 16, 4], masks=None)
        x = torch.randn(5, 8)
        out = model(x)
        assert out.shape == (5, 4)

    def test_masked_forward(self, simple_mask):
        mask_16 = uniform_sparse_mask((16, 16), density=0.5, seed=42)
        # layer_sizes=[8, 16, 16, 4] has 3 transitions: 8→16, 16→16, 16→4.
        # The bio mask is 16×16 (square), so it only applies to the 16→16 layer.
        # Non-square input/output projections get None (dense).
        model = BioMLP(layer_sizes=[8, 16, 16, 4], masks=[None, mask_16, None])
        x = torch.randn(5, 8)
        out = model(x)
        assert out.shape == (5, 4)

    def test_output_shape_multiclass(self):
        model = BioMLP(layer_sizes=[64, 128, 10], masks=None)
        x = torch.randn(32, 64)
        out = model(x)
        assert out.shape == (32, 10)

    def test_from_connectome_dense(self):
        adj = np.random.randint(0, 2, (10, 10)).astype(np.float32)
        model = BioMLP.from_connectome(adj, layer_sizes=[8, 16, 4], mask_type="dense")
        x = torch.randn(3, 8)
        out = model(x)
        assert out.shape == (3, 4)

    def test_from_connectome_bio(self):
        adj = np.random.randint(0, 2, (10, 10)).astype(np.float32)
        model = BioMLP.from_connectome(adj, layer_sizes=[8, 16, 4], mask_type="bio")
        x = torch.randn(3, 8)
        out = model(x)
        assert out.shape == (3, 4)

    def test_from_connectome_uniform_sparse(self):
        adj = np.random.randint(0, 2, (10, 10)).astype(np.float32)
        model = BioMLP.from_connectome(adj, layer_sizes=[8, 16, 4], mask_type="uniform_sparse")
        x = torch.randn(3, 8)
        out = model(x)
        assert out.shape == (3, 4)

    def test_gradients_flow(self):
        model = BioMLP(layer_sizes=[4, 8, 2], masks=None)
        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_training_loop(self):
        """Smoke test: loss decreases over a few steps."""
        model = BioMLP(layer_sizes=[4, 16, 2], masks=None)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(20, 4)
        y = torch.randint(0, 2, (20,))

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease over 20 steps
        assert losses[-1] < losses[0]


# MaskedRNN tests

class TestMaskedRNN:

    def test_output_shape(self):
        mask_ih = dense_mask((8, 4))   # hidden x input
        mask_hh = dense_mask((8, 8))   # hidden x hidden
        rnn = MaskedRNN(input_size=4, hidden_size=8, mask_ih=mask_ih, mask_hh=mask_hh)
        x = torch.randn(10, 3, 4)  # seq_len=10, batch=3, input=4
        output, h_n = rnn(x)
        assert output.shape == (10, 3, 8)
        assert h_n.shape == (1, 3, 8)

    def test_mask_not_trainable(self):
        mask_ih = dense_mask((8, 4))
        mask_hh = dense_mask((8, 8))
        rnn = MaskedRNN(input_size=4, hidden_size=8, mask_ih=mask_ih, mask_hh=mask_hh)
        param_names = [n for n, _ in rnn.named_parameters()]
        assert not any("mask" in n for n in param_names)

    def test_bio_rnn_output_shape(self):
        mask_ih = uniform_sparse_mask((16, 8), density=0.3, seed=42)
        mask_hh = uniform_sparse_mask((16, 16), density=0.3, seed=42)
        model = BioRNN(input_size=8, hidden_size=16, output_size=4,
                       mask_ih=mask_ih, mask_hh=mask_hh)
        x = torch.randn(5, 3, 8)  # seq=5, batch=3, input=8
        # get_sequence_output returns only the final timestep: (batch, output_size)
        out = model.get_sequence_output(x)
        assert out.shape == (3, 4)  # batch x output
