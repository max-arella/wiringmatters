"""
Topological analysis functions for connectomes and networks.

This module computes key topological properties of graphs: clustering coefficients,
modularity, path lengths, hub structure, and small-world properties.
All functions operate on NetworkX graph objects and return plain dicts for easy
logging and JSON serialization.
"""

from typing import Any

import networkx as nx
import numpy as np


def compute_clustering(graph: nx.Graph) -> dict[str, Any]:
    """
    Compute local and global clustering coefficients.

    The clustering coefficient of a node measures the degree to which its
    neighbors are also connected to each other. High local clustering combined
    with short path lengths is the hallmark of small-world topology.

    Args:
        graph: A NetworkX Graph or DiGraph.

    Returns:
        dict with keys:
          - "global_clustering" (float): Average clustering coefficient across all nodes.
          - "per_node" (dict): Mapping from node ID to local clustering coefficient.

    Example:
        >>> G = nx.complete_graph(4)
        >>> result = compute_clustering(G)
        >>> result["global_clustering"]
        1.0
    """
    # Convert DiGraph to undirected for clustering (standard approach)
    ug = graph.to_undirected() if graph.is_directed() else graph
    per_node = dict(nx.clustering(ug))
    global_c = float(np.mean(list(per_node.values()))) if per_node else 0.0
    return {"global_clustering": global_c, "per_node": per_node}


def compute_modularity(graph: nx.Graph) -> dict[str, Any]:
    """
    Detect communities and compute modularity Q.

    Modularity Q ranges from -1 to 1. Q > 0.3 indicates strong community structure.
    Community detection uses Louvain if available, otherwise falls back to greedy
    modularity optimization from NetworkX.

    Args:
        graph: A NetworkX Graph or DiGraph.

    Returns:
        dict with keys:
          - "modularity" (float): Modularity score Q.
          - "n_communities" (int): Number of detected communities.
          - "community_sizes" (list[int]): Size of each community.
          - "partition" (dict): Mapping from node ID to community ID.

    Example:
        >>> G = nx.barbell_graph(5, 0)
        >>> result = compute_modularity(G)
        >>> result["n_communities"]
        2
    """
    ug = graph.to_undirected() if graph.is_directed() else graph

    if ug.number_of_nodes() == 0:
        return {"modularity": 0.0, "n_communities": 0, "community_sizes": [], "partition": {}}

    # Try Louvain (best quality), fallback to greedy
    try:
        from community import community_louvain  # python-louvain
        partition = community_louvain.best_partition(ug)
    except ImportError:
        from networkx.algorithms import community as nx_community
        sets = list(nx_community.greedy_modularity_communities(ug))
        partition = {node: i for i, s in enumerate(sets) for node in s}

    # Ensure all nodes are assigned
    for node in ug.nodes():
        if node not in partition:
            partition[node] = 0

    # Build communities dict
    communities: dict[int, set] = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, set()).add(node)

    # Compute modularity Q
    try:
        Q = nx.algorithms.community.modularity(ug, list(communities.values()))
    except Exception:
        Q = 0.0

    return {
        "modularity": float(Q),
        "n_communities": len(communities),
        "community_sizes": sorted([len(s) for s in communities.values()], reverse=True),
        "partition": partition,
    }


def compute_path_length(graph: nx.Graph) -> dict[str, Any]:
    """
    Compute average shortest path length.

    For disconnected graphs, uses the largest connected component.

    Args:
        graph: A NetworkX Graph or DiGraph.

    Returns:
        dict with key:
          - "avg_path_length" (float): Average shortest path length. 0.0 if
            the graph has fewer than 2 nodes or is fully disconnected.

    Example:
        >>> G = nx.complete_graph(5)
        >>> result = compute_path_length(G)
        >>> result["avg_path_length"]
        1.0
    """
    ug = graph.to_undirected() if graph.is_directed() else graph

    if ug.number_of_nodes() < 2:
        return {"avg_path_length": 0.0}

    try:
        if nx.is_connected(ug):
            L = nx.average_shortest_path_length(ug)
        else:
            largest_cc = max(nx.connected_components(ug), key=len)
            sub = ug.subgraph(largest_cc)
            L = nx.average_shortest_path_length(sub) if sub.number_of_nodes() >= 2 else 0.0
    except Exception:
        L = 0.0

    return {"avg_path_length": float(L)}


def compute_hub_scores(graph: nx.Graph) -> dict[str, Any]:
    """
    Compute degree centrality and betweenness centrality.

    Hub neurons (nodes) are those with disproportionately high centrality. They
    are key features of biological connectomes — a small number of highly
    connected neurons integrate information across the network.

    Args:
        graph: A NetworkX Graph or DiGraph.

    Returns:
        dict with keys:
          - "degree_centrality" (dict): Node → degree centrality.
          - "betweenness_centrality" (dict): Node → betweenness centrality.
          - "max_degree_centrality" (float): Maximum degree centrality.
          - "avg_betweenness" (float): Mean betweenness centrality.

    Example:
        >>> G = nx.star_graph(4)
        >>> result = compute_hub_scores(G)
        >>> result["degree_centrality"][0]  # center node
        1.0
    """
    deg = nx.degree_centrality(graph)
    between = nx.betweenness_centrality(graph)
    return {
        "degree_centrality": deg,
        "betweenness_centrality": between,
        "max_degree_centrality": float(max(deg.values())) if deg else 0.0,
        "avg_betweenness": float(np.mean(list(between.values()))) if between else 0.0,
    }


def compute_small_world_sigma(graph: nx.Graph) -> dict[str, Any]:
    """
    Compute the small-world coefficient sigma.

    sigma = (C / C_rand) / (L / L_rand)

    C is clustering coefficient, L is average path length. Subscript _rand
    refers to a random graph with the same degree distribution.
    sigma > 1 indicates small-world properties.

    Args:
        graph: A connected NetworkX Graph with at least 3 nodes.

    Returns:
        dict with key:
          - "small_world_sigma" (float): Small-world coefficient. 0.0 if graph
            is too small, disconnected, or computation fails.

    Example:
        >>> G = nx.watts_strogatz_graph(30, 4, 0.3, seed=42)
        >>> result = compute_small_world_sigma(G)
        >>> result["small_world_sigma"] > 1.0
        True
    """
    ug = graph.to_undirected() if graph.is_directed() else graph

    if ug.number_of_nodes() < 3 or not nx.is_connected(ug):
        return {"small_world_sigma": 0.0}

    try:
        C_actual = nx.average_clustering(ug)
        L_actual = nx.average_shortest_path_length(ug)

        degree_seq = [d for _, d in ug.degree()]
        rg = nx.configuration_model(degree_seq, seed=42)
        rg = nx.Graph(rg)
        rg.remove_edges_from(nx.selfloop_edges(rg))

        if not nx.is_connected(rg):
            largest_cc = max(nx.connected_components(rg), key=len)
            rg = rg.subgraph(largest_cc).copy()

        C_rand = nx.average_clustering(rg)
        L_rand = nx.average_shortest_path_length(rg)

        if C_rand == 0 or L_rand == 0:
            return {"small_world_sigma": 0.0}

        sigma = (C_actual / C_rand) / (L_actual / L_rand)
        return {"small_world_sigma": float(sigma)}
    except Exception:
        return {"small_world_sigma": 0.0}


def topological_summary(graph: nx.Graph) -> dict[str, Any]:
    """
    Compute a comprehensive topological summary of a connectome graph.

    Aggregates all topology metrics into a single dict. This is the primary
    function used in notebooks and experiment logs.

    Args:
        graph: A NetworkX Graph or DiGraph (e.g., from load_celegans()).

    Returns:
        dict with keys:
          n_nodes, n_edges, density, avg_degree,
          clustering_coefficient, avg_path_length, small_world_sigma,
          modularity, n_communities,
          max_degree_centrality, avg_betweenness_centrality

    Example:
        >>> G, _ = load_celegans(as_matrix=True)
        >>> summary = topological_summary(G)
        >>> print(f"Modularity: {summary['modularity']:.3f}")
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    density = float(nx.density(graph))
    avg_degree = float(2 * n_edges / n_nodes) if n_nodes > 0 else 0.0

    clustering   = compute_clustering(graph)
    path_info    = compute_path_length(graph)
    sigma_info   = compute_small_world_sigma(graph)
    mod_info     = compute_modularity(graph)
    hub_info     = compute_hub_scores(graph)

    return {
        "n_nodes":                    n_nodes,
        "n_edges":                    n_edges,
        "density":                    density,
        "avg_degree":                 avg_degree,
        "clustering_coefficient":     clustering["global_clustering"],
        "avg_path_length":            path_info["avg_path_length"],
        "small_world_sigma":          sigma_info["small_world_sigma"],
        "modularity":                 mod_info["modularity"],
        "n_communities":              mod_info["n_communities"],
        "max_degree_centrality":      hub_info["max_degree_centrality"],
        "avg_betweenness_centrality": hub_info["avg_betweenness"],
    }
