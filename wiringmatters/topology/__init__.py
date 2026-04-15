"""WiringMatters topology module.

This module provides tools for extracting and applying topological properties
from biological connectomes to neural network architectures.

Core functionality:
- Topological analysis: clustering, modularity, hub structure, small-world properties
- Mask generation: binary connectivity masks that enforce biological constraints
- Utilities: graph conversions, binarization, density computation
"""

from .analysis import (
    compute_clustering,
    compute_hub_scores,
    compute_modularity,
    compute_path_length,
    compute_small_world_sigma,
    topological_summary,
)
from .masks import (
    bio_mask,
    dense_mask,
    magnitude_mask,
    mask_density,
    uniform_sparse_mask,
)
from .utils import (
    adjacency_to_graph,
    binarize,
    graph_to_adjacency,
)

__all__ = [
    # Analysis functions
    "compute_clustering",
    "compute_modularity",
    "compute_path_length",
    "compute_hub_scores",
    "compute_small_world_sigma",
    "topological_summary",
    # Mask functions
    "bio_mask",
    "uniform_sparse_mask",
    "dense_mask",
    "magnitude_mask",
    "mask_density",
    # Utility functions
    "adjacency_to_graph",
    "graph_to_adjacency",
    "binarize",
]
