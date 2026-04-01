"""
Lambda Experiment Package

Contains topology generators and metrics for studying TD(λ) performance
across different maze topologies.
"""

from .topology_generators import (
    make_corridor,
    make_procedural_maze,
    ALL_TOPOLOGIES,
    generate_topology
)

from .topology_metrics import (
    compute_node_types,
    compute_mean_corridor_length,
    compute_junction_density,
    compute_corr_dec_ratio,
    compute_all_metrics
)

__all__ = [
    'make_corridor',
    'make_procedural_maze',
    'ALL_TOPOLOGIES',
    'generate_topology',
    'compute_node_types',
    'compute_mean_corridor_length',
    'compute_junction_density',
    'compute_corr_dec_ratio',
    'compute_all_metrics',
]
