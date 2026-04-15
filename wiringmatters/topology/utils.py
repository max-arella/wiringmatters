"""Utility functions for topology operations.

This module provides conversion utilities between different graph representations
and basic matrix operations for working with adjacency matrices.
"""

from typing import Union

import networkx as nx
import numpy as np


def adjacency_to_graph(matrix: np.ndarray) -> nx.Graph:
    """Convert an adjacency matrix to a NetworkX graph object.

    Converts a weighted or binary adjacency matrix into a NetworkX graph.
    The graph is undirected; the input matrix should be symmetric for
    meaningful interpretation. Self-loops are excluded.

    Args:
        matrix: Adjacency matrix of shape (n, n). Can be weighted (float) or
            binary (0/1). Symmetric for undirected graphs.

    Returns:
        A NetworkX Graph object. Edge weights are preserved from the input matrix.

    Raises:
        ValueError: If the matrix is not square.

    Examples:
        Create a simple adjacency matrix and convert to graph:

        >>> adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        >>> G = adjacency_to_graph(adj)
        >>> G.number_of_nodes()
        3
        >>> G.number_of_edges()
        2
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Adjacency matrix must be square, got shape {matrix.shape}"
        )

    G = nx.Graph()
    n = matrix.shape[0]
    G.add_nodes_from(range(n))

    # Add edges with weights, excluding self-loops
    for i in range(n):
        for j in range(i + 1, n):
            weight = matrix[i, j]
            if weight != 0:
                G.add_edge(i, j, weight=weight)

    return G


def graph_to_adjacency(graph: nx.Graph) -> np.ndarray:
    """Convert a NetworkX graph to an adjacency matrix.

    Creates a dense adjacency matrix from a NetworkX graph. Edge weights
    are preserved; unweighted edges default to weight 1.

    Args:
        graph: A NetworkX Graph object. Can be weighted or unweighted.

    Returns:
        Adjacency matrix of shape (n, n) as a numpy array. Symmetric for
        undirected graphs. Data type is float64.

    Examples:
        Create a simple graph and convert to adjacency matrix:

        >>> G = nx.cycle_graph(4)
        >>> adj = graph_to_adjacency(G)
        >>> adj.shape
        (4, 4)
        >>> adj[0, 1]
        1.0
    """
    n = graph.number_of_nodes()
    adj = np.zeros((n, n), dtype=np.float64)

    for i, j, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)
        adj[i, j] = weight
        adj[j, i] = weight  # Ensure symmetry for undirected graphs

    return adj


def binarize(
    matrix: np.ndarray, threshold: float = 0.0
) -> np.ndarray:
    """Convert a weighted matrix to binary by thresholding.

    Applies a simple threshold: values > threshold become 1, else 0.
    Useful for converting weighted connectomes to binary connectivity masks.

    Args:
        matrix: Input matrix (any shape) with numeric values.
        threshold: Threshold value. Elements > threshold map to 1. Default 0.

    Returns:
        Binary matrix (0 and 1 values) of the same shape as input.
        Data type is float64.

    Examples:
        Binarize a weighted adjacency matrix:

        >>> weights = np.array([[0, 0.5, 0.1], [0.5, 0, 0.8], [0.1, 0.8, 0]])
        >>> binary = binarize(weights, threshold=0.3)
        >>> binary
        array([[0., 1., 0.],
               [1., 0., 1.],
               [0., 1., 0.]])

        With different threshold:

        >>> binary = binarize(weights, threshold=0.6)
        >>> binary
        array([[0., 0., 0.],
               [0., 0., 1.],
               [0., 1., 0.]])
    """
    return np.where(matrix > threshold, 1.0, 0.0)
