"""Binary connectivity mask generation for biological topology constraints.

Masks are applied as element-wise multiplication: W_eff = W * M.
Masks are registered as non-trainable buffers and stay fixed during training.

Scaling strategy (used consistently across all experiments):
  N = connectome size (448): use full matrix directly.
  N < 448: crop the top-left N×N submatrix.
  N > 448: tile with wrap-around.
"""

from typing import Union

import numpy as np
import torch


def bio_mask(
    adjacency_matrix: Union[np.ndarray, torch.Tensor],
    target_shape: Union[tuple, int],
) -> torch.Tensor:
    """Generate a binary mask from a biological adjacency matrix.

    Scales the connectome topology to the target layer size using the
    project's fixed scaling rule (tile/crop/direct).

    Args:
        adjacency_matrix: Binary or weighted square adjacency matrix.
            Will be binarized before use.
        target_shape: Target size as int n (produces n×n mask) or tuple (n, n).

    Returns:
        Binary float32 tensor of shape (n, n).

    Raises:
        ValueError: If the matrix is not square or target_shape is invalid.
    """
    if isinstance(adjacency_matrix, torch.Tensor):
        adj = adjacency_matrix.cpu().numpy()
    else:
        adj = np.asarray(adjacency_matrix)

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {adj.shape}")

    if isinstance(target_shape, int):
        n = target_shape
    elif isinstance(target_shape, (tuple, list)) and len(target_shape) == 2:
        if target_shape[0] != target_shape[1]:
            raise ValueError(f"Target shape must be square, got {target_shape}")
        n = target_shape[0]
    else:
        raise ValueError(f"Invalid target_shape: {target_shape}")

    binary = np.where(adj > 0, 1.0, 0.0)
    src = binary.shape[0]

    if n == src:
        result = binary.copy()
    elif n > src:
        result = _tile_adjacency(binary, n)
    else:
        result = binary[:n, :n].copy()

    return torch.tensor(result, dtype=torch.float32)


def _tile_adjacency(adjacency: np.ndarray, target_size: int) -> np.ndarray:
    """Tile an adjacency matrix to a larger size using block replication."""
    scale = (target_size + adjacency.shape[0] - 1) // adjacency.shape[0]
    tiled = np.tile(adjacency, (scale, scale))
    return tiled[:target_size, :target_size]


def uniform_sparse_mask(
    target_shape: Union[tuple, int],
    density: float,
    seed: int = None,
) -> torch.Tensor:
    """Generate a random binary mask at a given density.

    Used as the density-matched control condition: same sparsity as the
    biological mask but with no specific wiring pattern.

    Args:
        target_shape: Output size as int n or tuple (rows, cols).
        density: Fraction of active connections in [0, 1].
        seed: Optional random seed for reproducibility.

    Returns:
        Binary float32 tensor.

    Raises:
        ValueError: If density is outside [0, 1] or target_shape is invalid.
    """
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"Density must be in [0, 1], got {density}")

    if isinstance(target_shape, int):
        rows, cols = target_shape, target_shape
    elif isinstance(target_shape, (tuple, list)) and len(target_shape) == 2:
        rows, cols = int(target_shape[0]), int(target_shape[1])
    else:
        raise ValueError(f"Invalid target_shape: {target_shape}")

    if seed is not None:
        np.random.seed(seed)

    mask = np.random.binomial(1, density, size=(rows, cols)).astype(np.float32)
    return torch.tensor(mask, dtype=torch.float32)


def dense_mask(target_shape: Union[tuple, int]) -> torch.Tensor:
    """Create an all-ones mask (fully connected baseline).

    Args:
        target_shape: Output size as int n or tuple (rows, cols).

    Returns:
        Binary float32 tensor of all ones.
    """
    if isinstance(target_shape, int):
        rows, cols = target_shape, target_shape
    elif isinstance(target_shape, (tuple, list)) and len(target_shape) == 2:
        rows, cols = int(target_shape[0]), int(target_shape[1])
    else:
        raise ValueError(f"Invalid target_shape: {target_shape}")

    return torch.ones((rows, cols), dtype=torch.float32)


def magnitude_mask(weight: torch.Tensor, density: float) -> torch.Tensor:
    """Create a binary mask keeping the top-k% weights by absolute magnitude.

    Used for the magnitude-pruning baseline: after training a dense model,
    keep the connections it relied on most and retrain from scratch with
    that fixed topology. Tests whether a data-derived sparse topology
    beats a biologically-derived one.

    Args:
        weight: Trained 2D weight matrix (must be square).
        density: Fraction of weights to keep, in (0, 1].

    Returns:
        Binary float32 tensor with 1 where |weight| >= threshold.

    Raises:
        ValueError: If density is out of range or weight is not 2D.
    """
    if not 0.0 < density <= 1.0:
        raise ValueError(f"Density must be in (0, 1], got {density}")
    if weight.ndim != 2:
        raise ValueError(f"Weight must be 2D, got shape {weight.shape}")

    flat = weight.detach().abs().flatten()
    k = max(1, int(density * flat.numel()))
    threshold = flat.topk(k).values.min()
    return (weight.detach().abs() >= threshold).to(torch.float32)


def mask_density(mask: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute the fraction of non-zero entries in a binary mask.

    Args:
        mask: 2D binary array or tensor.

    Returns:
        Float in [0, 1].
    """
    if isinstance(mask, torch.Tensor):
        arr = mask.cpu().numpy()
    else:
        arr = np.asarray(mask)

    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {arr.shape}")

    total = arr.size
    return float(np.count_nonzero(arr) / total) if total > 0 else 0.0
