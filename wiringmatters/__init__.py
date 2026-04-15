"""
WiringMatters — Biologically-Constrained Neural Networks
=========================================================

WiringMatters is an open-source research package that applies real biological
connectome topology (C. elegans, Drosophila) as fixed binary masks on fully
trainable neural networks, and measures the impact on generalization,
robustness, and data efficiency compared to dense or uniformly sparse baselines.

Research question:
    Does forcing the connectivity topology of a real biological connectome as a
    fixed binary mask on an artificial neural network change its performance —
    in terms of generalization, robustness, and data efficiency?

Quick start:
    >>> import wiringmatters as wm
    >>> G, adj = wm.load_celegans(as_matrix=True)
    >>> mask = wm.bio_mask(adj, target_shape=(256, 256))
    >>> model = wm.BioMLP(layer_sizes=[784, 256, 256, 10], masks=[None, mask, None])

Modules:
    loaders   — Load connectome data (C. elegans, Drosophila)
    topology  — Extract topological properties and build connectivity masks
    models    — MaskedLinear, MaskedRNN, BioMLP, BioRNN PyTorch layers

Version: 0.1.0 (Stage 1 — Topology only)
Author:  Maxence Arella
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Maxence Arella"
__email__ = "maxencearella@gmail.com"
__license__ = "MIT"

# Loaders
from wiringmatters.loaders import load_celegans, load_drosophila

# Topology
from wiringmatters.topology import (
    # Mask construction
    bio_mask,
    uniform_sparse_mask,
    dense_mask,
    magnitude_mask,
    mask_density,
    # Topological analysis
    topological_summary,
    compute_clustering,
    compute_modularity,
    compute_path_length,
    compute_hub_scores,
    compute_small_world_sigma,
    # Utilities
    adjacency_to_graph,
    graph_to_adjacency,
    binarize,
)

# Models
from wiringmatters.models import (
    MaskedLinear,
    MaskedRNNCell,
    MaskedRNN,
    BioMLP,
    BioRNN,
)

__all__ = [
    # Loaders
    "load_celegans",
    "load_drosophila",
    # Mask construction
    "bio_mask",
    "uniform_sparse_mask",
    "dense_mask",
    "magnitude_mask",
    "mask_density",
    # Topological analysis
    "topological_summary",
    "compute_clustering",
    "compute_modularity",
    "compute_path_length",
    "compute_hub_scores",
    "compute_small_world_sigma",
    # Utilities
    "adjacency_to_graph",
    "graph_to_adjacency",
    "binarize",
    # Models
    "MaskedLinear",
    "MaskedRNNCell",
    "MaskedRNN",
    "BioMLP",
    "BioRNN",
]
