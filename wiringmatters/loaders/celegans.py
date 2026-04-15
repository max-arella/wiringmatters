"""
C. elegans connectome loader.

Loads connectome data from the OpenWorm project (https://www.openworm.org/).

The canonical C. elegans hermaphrodite has 302 somatic neurons. However, the
full connectome dataset (herm_full_edgelist.csv) contains 448 unique nodes
because it includes the 20 pharyngeal neurons plus muscle cells and other
non-neuronal cells that receive chemical synapses. WiringMatters uses all 448
nodes to preserve the richer topology of the complete wiring diagram.

Data sources (in order of priority):
  1. OpenWorm CElegansNeuroML — herm_full_edgelist.csv (edge list format, 448 nodes)
  2. Varshney et al. 2011 via ivan-ea/celegans_connectome (adjacency matrix)

Both are downloaded on first use and cached locally in ~/.wiringmatters/data/.
"""

import csv
import io
import logging
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import networkx as nx
import numpy as np
import requests

logger = logging.getLogger(__name__)

# Data sources

# Primary: OpenWorm edge list (most canonical, maintained by OpenWorm project)
OPENWORM_EDGELIST_URL = (
    "https://raw.githubusercontent.com/openworm/CElegansNeuroML/"
    "master/herm_full_edgelist.csv"
)

# Fallback: Varshney et al. 2011 adjacency matrices (stable academic source)
VARSHNEY_CHEM_URL = (
    "https://raw.githubusercontent.com/ivan-ea/celegans_connectome/"
    "master/Chem.csv"
)
VARSHNEY_GAP_URL = (
    "https://raw.githubusercontent.com/ivan-ea/celegans_connectome/"
    "master/Gap.csv"
)
VARSHNEY_LABELS_URL = (
    "https://raw.githubusercontent.com/ivan-ea/celegans_connectome/"
    "master/Labels.csv"
)

CACHE_DIR = Path.home() / ".wiringmatters" / "data" / "celegans"
TIMEOUT = 30


# Cache helpers

def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, filename: str, force: bool = False) -> Optional[Path]:
    """Download a file and cache it. Returns Path on success, None on failure."""
    _ensure_cache_dir()
    filepath = CACHE_DIR / filename

    if filepath.exists() and not force:
        logger.debug(f"Using cached: {filepath}")
        return filepath

    logger.info(f"Downloading {filename} from {url}")
    try:
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        filepath.write_text(response.text, encoding="utf-8")
        logger.info(f"Cached to: {filepath}")
        return filepath
    except requests.RequestException as e:
        logger.warning(f"Download failed ({url}): {e}")
        return None


# Parser 1 — OpenWorm edge list

def _parse_edgelist(filepath: Path) -> Tuple[list, np.ndarray]:
    """
    Parse OpenWorm herm_full_edgelist.csv.

    Expected columns (flexible): pre, post, [sections], [synapses/weight/num_synapses]
    Rows with a 'type' or 'synapse_type' column equal to 'EJ' are gap junctions.
    All other rows are treated as chemical synapses.

    Returns:
        (neuron_names, full_adj)  where full_adj[:,0] is chem, full_adj[:,1] is gap.
        Actually returns (neuron_names, chem_matrix, gap_matrix).
    """
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Edge list file is empty")

    fieldnames = [f.strip().lower() for f in reader.fieldnames or []]
    logger.debug(f"Edge list columns: {fieldnames}")

    # Collect all neuron names
    pre_col  = _find_col(fieldnames, ["pre", "from", "source", "neuron1"])
    post_col = _find_col(fieldnames, ["post", "to", "target", "neuron2"])
    wt_col   = _find_col(fieldnames, ["synapses", "weight", "sections",
                                      "num_synapses", "n_synapses", "value"])
    type_col = _find_col(fieldnames, ["type", "synapse_type", "connection_type"])

    if pre_col is None or post_col is None:
        raise ValueError(
            f"Cannot find pre/post columns in edge list. "
            f"Available columns: {fieldnames}"
        )

    orig_fields = reader.fieldnames or []
    pre_orig  = orig_fields[pre_col]
    post_orig = orig_fields[post_col]

    all_neurons = sorted(set(
        [r[pre_orig].strip() for r in rows] +
        [r[post_orig].strip() for r in rows]
    ))
    n = len(all_neurons)
    idx = {name: i for i, name in enumerate(all_neurons)}

    chem_matrix = np.zeros((n, n), dtype=np.float32)
    gap_matrix  = np.zeros((n, n), dtype=np.float32)

    for row in rows:
        pre  = row[pre_orig].strip()
        post = row[post_orig].strip()
        if not pre or not post:
            continue

        # Synaptic weight
        weight = 1.0
        if wt_col is not None:
            wt_field = orig_fields[wt_col]
            try:
                weight = abs(float(row[wt_field])) or 1.0
            except (ValueError, TypeError):
                weight = 1.0

        # Synapse type
        is_gap = False
        if type_col is not None:
            t_field = orig_fields[type_col]
            t_val = row.get(t_field, "").strip().upper()
            is_gap = t_val in ("EJ", "GAP", "ELECTRICAL", "GAP_JUNCTION", "GJ")

        i, j = idx[pre], idx[post]
        if is_gap:
            gap_matrix[i, j] += weight
            gap_matrix[j, i] += weight  # gap junctions are bidirectional
        else:
            chem_matrix[i, j] += weight

    logger.info(
        f"Parsed edge list: {n} neurons, "
        f"{int(np.count_nonzero(chem_matrix))} chem edges, "
        f"{int(np.count_nonzero(gap_matrix))} gap edges"
    )
    return all_neurons, chem_matrix, gap_matrix


def _find_col(fieldnames: list, candidates: list) -> Optional[int]:
    """Return the index of the first matching column name (case-insensitive)."""
    for i, f in enumerate(fieldnames):
        if f in candidates:
            return i
    return None


# Parser 2 — Varshney adjacency matrix

def _parse_varshney_matrix(matrix_path: Path, labels_path: Path) -> Tuple[list, np.ndarray]:
    """
    Parse Varshney et al. 2011 adjacency matrix CSV (no header, pure numbers)
    alongside a separate Labels.csv for neuron names.
    """
    # Labels: one neuron name per line
    with open(labels_path, encoding="utf-8") as f:
        labels_raw = [line.strip() for line in f if line.strip()]
    # Drop header row if it looks like "name" or "neuron"
    if labels_raw and labels_raw[0].lower() in ("name", "neuron", "label"):
        labels_raw = labels_raw[1:]

    # Matrix: numeric CSV
    matrix_data = []
    with open(matrix_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = []
            for val in line.split(","):
                try:
                    row.append(float(val.strip()))
                except ValueError:
                    pass
            if row:
                matrix_data.append(row)

    adj = np.array(matrix_data, dtype=np.float32)
    n = adj.shape[0]

    # If labels count mismatches, use generic names
    if len(labels_raw) != n:
        logger.warning(
            f"Labels count ({len(labels_raw)}) != matrix size ({n}). "
            "Using generic neuron names."
        )
        labels_raw = [f"N{i:03d}" for i in range(n)]

    logger.info(f"Parsed Varshney matrix: {n} neurons, {int(np.count_nonzero(adj))} edges")
    return labels_raw, adj


# Graph builder

def _to_digraph(neuron_names: list, adj_matrix: np.ndarray) -> nx.DiGraph:
    """Convert adjacency matrix + names to a weighted NetworkX DiGraph."""
    G = nx.DiGraph()
    for name in neuron_names:
        G.add_node(name)
    rows, cols = np.nonzero(adj_matrix)
    for i, j in zip(rows, cols):
        G.add_edge(neuron_names[i], neuron_names[j], weight=float(adj_matrix[i, j]))
    logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# Public API

def load_celegans(
    synapse_type: Literal["chemical", "gap", "all"] = "chemical",
    as_matrix: bool = False,
    force_download: bool = False,
    include_metadata: bool = False,
) -> Union[nx.DiGraph, Tuple[nx.DiGraph, np.ndarray]]:
    """
    Load the C. elegans connectome.

    Downloads data automatically on first call and caches it in
    ~/.wiringmatters/data/celegans/. Subsequent calls use the cache.

    Two data sources are tried in order:
      1. OpenWorm herm_full_edgelist.csv (edge list, most up-to-date)
      2. Varshney et al. 2011 adjacency matrices (stable fallback)

    Args:
        synapse_type: Which synapses to load.
            "chemical"  — chemical synapses only (default, ~2000 edges)
            "gap"       — gap junctions (electrical synapses) only
            "all"       — chemical + gap junctions combined
        as_matrix: If True, return (graph, adjacency_matrix).
                   If False, return graph only.
        force_download: Re-download data even if already cached.
        include_metadata: Unused, kept for API compatibility.

    Returns:
        NetworkX DiGraph, or (DiGraph, numpy adjacency matrix) if as_matrix=True.

    Raises:
        ValueError: If synapse_type is not one of "chemical", "gap", or "all".
        RuntimeError: If all data sources fail (network error or parse error).

    Examples:
        >>> G = load_celegans()
        >>> print(G.number_of_nodes(), G.number_of_edges())

        >>> G, adj = load_celegans(synapse_type="chemical", as_matrix=True)
        >>> print(adj.shape)   # (n_neurons, n_neurons)
    """
    logger.info(f"Loading C. elegans connectome (synapse_type={synapse_type})")

    if synapse_type not in ("chemical", "gap", "all"):
        raise ValueError(f"synapse_type must be 'chemical', 'gap', or 'all', got '{synapse_type}'")

    chem_matrix = gap_matrix = neuron_names = None

    # Strategy 1: OpenWorm edge list
    edgelist_path = _download_file(
        OPENWORM_EDGELIST_URL, "herm_full_edgelist.csv", force_download
    )
    if edgelist_path is not None:
        try:
            neuron_names, chem_matrix, gap_matrix = _parse_edgelist(edgelist_path)
            logger.info("Using OpenWorm edge list data.")
        except Exception as e:
            logger.warning(f"Edge list parse failed: {e}. Trying Varshney fallback.")
            neuron_names = chem_matrix = gap_matrix = None

    # Strategy 2: Varshney adjacency matrices
    if chem_matrix is None:
        logger.info("Falling back to Varshney et al. 2011 data.")
        chem_path   = _download_file(VARSHNEY_CHEM_URL,   "varshney_chem.csv",   force_download)
        gap_path    = _download_file(VARSHNEY_GAP_URL,    "varshney_gap.csv",    force_download)
        labels_path = _download_file(VARSHNEY_LABELS_URL, "varshney_labels.csv", force_download)

        if chem_path and labels_path:
            try:
                neuron_names, chem_matrix = _parse_varshney_matrix(chem_path, labels_path)
                if gap_path:
                    _, gap_matrix = _parse_varshney_matrix(gap_path, labels_path)
                else:
                    gap_matrix = np.zeros_like(chem_matrix)
                logger.info("Using Varshney et al. data.")
            except Exception as e:
                raise RuntimeError(
                    f"Both data sources failed. Last error: {e}\n"
                    "Check your internet connection and try force_download=True."
                ) from e
        else:
            raise RuntimeError(
                "Could not download C. elegans data from any source. "
                "Check your internet connection."
            )

    # Build binarized adjacency matrix for the requested synapse type
    if synapse_type == "chemical":
        final_matrix = (chem_matrix > 0).astype(np.float32)
    elif synapse_type == "gap":
        final_matrix = (gap_matrix > 0).astype(np.float32) if gap_matrix is not None \
                       else np.zeros_like(chem_matrix)
    else:  # "all"
        combined = chem_matrix + (gap_matrix if gap_matrix is not None else 0)
        final_matrix = (combined > 0).astype(np.float32)

    # Build graph
    G = _to_digraph(neuron_names, final_matrix)

    if as_matrix:
        return G, final_matrix
    return G
