"""
WiringMatters models module.

Provides PyTorch layers and networks with biological connectivity masking.

Exports:
    - MaskedLinear: Linear layer with fixed connectivity mask
    - MaskedRNNCell: RNN cell with masked recurrent weights
    - MaskedRNN: Multi-layer RNN with masked connections
    - BioMLP: Multi-layer perceptron with optional connectivity masks
    - BioRNN: Recurrent network with masked connections and linear readout
"""

from wiringmatters.models.layers import (
    MaskedLinear,
    MaskedRNNCell,
    MaskedRNN,
)
from wiringmatters.models.networks import (
    BioMLP,
    BioRNN,
)

__all__ = [
    "MaskedLinear",
    "MaskedRNNCell",
    "MaskedRNN",
    "BioMLP",
    "BioRNN",
]
