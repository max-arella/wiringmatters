"""
Tests for wiringmatters.topology
---------------------------------
Run with: pytest tests/test_topology.py -v
"""

import numpy as np
import networkx as nx
import torch
import pytest

from wiringmatters.topology import (
    bio_mask,
    uniform_sparse_mask,
    dense_mask,
    mask_density,
    adjacency_to_graph,
    graph_to_adjacency,
    binarize,
    topological_summary,
    compute_clustering,
    compute_modularity,
)


# Fixtures

@pytest.fixture
def small_adjacency() -> np.ndarray:
    """4x4 binary adjacency matrix representing a tiny connectome."""
    return np.array([
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 0],
    ], dtype=np.float32)


@pytest.fixture
def small_graph(small_adjacency) -> nx.DiGraph:
    return adjacency_to_graph(small_adjacency)


# Mask tests

class TestMasks:

    def test_dense_mask_shape(self):
        mask = dense_mask((8, 8))
        assert mask.shape == (8, 8)

    def test_dense_mask_all_ones(self):
        mask = dense_mask((8, 8))
        assert mask.sum().item() == 64

    def test_dense_mask_binary(self):
        mask = dense_mask((8, 8))
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    def test_uniform_sparse_mask_shape(self):
        mask = uniform_sparse_mask((16, 16), density=0.3)
        assert mask.shape == (16, 16)

    def test_uniform_sparse_mask_density(self):
        mask = uniform_sparse_mask((100, 100), density=0.3, seed=42)
        actual_density = mask_density(mask)
        # Allow ±5% tolerance
        assert abs(actual_density - 0.3) < 0.05

    def test_uniform_sparse_mask_binary(self):
        mask = uniform_sparse_mask((8, 8), density=0.5)
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    def test_bio_mask_shape_larger_than_connectome(self, small_adjacency):
        """Target larger than the 4x4 connectome — should tile."""
        mask = bio_mask(small_adjacency, target_shape=(8, 8))
        assert mask.shape == (8, 8)

    def test_bio_mask_shape_smaller_than_connectome(self, small_adjacency):
        """Target smaller than connectome — should crop."""
        mask = bio_mask(small_adjacency, target_shape=(2, 2))
        assert mask.shape == (2, 2)

    def test_bio_mask_exact_shape(self, small_adjacency):
        mask = bio_mask(small_adjacency, target_shape=(4, 4))
        assert mask.shape == (4, 4)
        assert torch.allclose(mask, torch.tensor(small_adjacency))

    def test_bio_mask_binary(self, small_adjacency):
        mask = bio_mask(small_adjacency, target_shape=(8, 8))
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    def test_bio_mask_not_trainable(self, small_adjacency):
        """Masks must not be trainable — they are fixed topology."""
        mask = bio_mask(small_adjacency, target_shape=(4, 4))
        assert not mask.requires_grad

    def test_mask_density_dense(self):
        mask = dense_mask((10, 10))
        assert mask_density(mask) == pytest.approx(1.0)

    def test_mask_density_sparse(self):
        mask = torch.zeros(10, 10)
        mask[0, 0] = 1.0
        assert mask_density(mask) == pytest.approx(0.01)


# Graph utility tests

class TestGraphUtils:

    def test_adjacency_to_graph(self, small_adjacency):
        G = adjacency_to_graph(small_adjacency)
        assert isinstance(G, (nx.DiGraph, nx.Graph))
        assert G.number_of_nodes() == 4

    def test_graph_to_adjacency(self, small_graph):
        adj = graph_to_adjacency(small_graph)
        assert isinstance(adj, np.ndarray)
        assert adj.shape == (4, 4)

    def test_roundtrip(self):
        """adjacency → graph → adjacency must be lossless for symmetric binary matrices.

        adjacency_to_graph builds an undirected Graph, so only symmetric
        (undirected) adjacency matrices survive the round-trip unchanged.
        """
        sym = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], dtype=np.float32)
        G = adjacency_to_graph(sym)
        adj2 = graph_to_adjacency(G)
        np.testing.assert_array_almost_equal(sym, adj2)

    def test_binarize(self):
        M = np.array([[0.0, 0.5, -0.1, 1.0]])
        B = binarize(M, threshold=0.0)
        expected = np.array([[0.0, 1.0, 0.0, 1.0]])
        np.testing.assert_array_equal(B, expected)

    def test_binarize_threshold(self):
        M = np.array([[0.0, 0.5, 0.8, 1.0]])
        B = binarize(M, threshold=0.6)
        expected = np.array([[0.0, 0.0, 1.0, 1.0]])
        np.testing.assert_array_equal(B, expected)


# Topological analysis tests

class TestTopologicalAnalysis:

    def test_clustering_returns_dict(self, small_graph):
        result = compute_clustering(small_graph)
        assert "global_clustering" in result
        assert "per_node" in result

    def test_clustering_range(self, small_graph):
        result = compute_clustering(small_graph)
        assert 0.0 <= result["global_clustering"] <= 1.0

    def test_modularity_returns_dict(self, small_graph):
        result = compute_modularity(small_graph)
        assert "modularity" in result
        assert "n_communities" in result

    def test_topological_summary_keys(self, small_graph):
        summary = topological_summary(small_graph)
        expected_keys = [
            "clustering_coefficient",
            "modularity",
            "n_communities",
            "density",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_topological_summary_values_in_range(self, small_graph):
        summary = topological_summary(small_graph)
        assert 0.0 <= summary["clustering_coefficient"] <= 1.0
        assert 0.0 <= summary["density"] <= 1.0
