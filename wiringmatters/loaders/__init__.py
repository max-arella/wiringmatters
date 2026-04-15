"""
Loaders module for WiringMatters.

Provides functions to load connectome data from various biological sources.
Currently supports:
- C. elegans (Caenorhabditis elegans): 302 somatic neurons, plus 20 pharyngeal
  neurons and ~126 muscle/non-neuronal cells in the full connectome dataset.
  Total: 448 unique nodes in the complete wiring diagram, ~7000 chemical synapses.
- Drosophila (coming in future versions)

Example:
    >>> from wiringmatters.loaders import load_celegans
    >>> G = load_celegans(synapse_type="chemical")
    >>> print(G.number_of_nodes(), G.number_of_edges())
    448 4681
"""

import logging

from wiringmatters.loaders.celegans import load_celegans
from wiringmatters.loaders.drosophila import load_drosophila

__all__ = [
    "load_celegans",
    "load_drosophila",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
