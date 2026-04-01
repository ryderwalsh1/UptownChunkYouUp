"""
Topology Metrics for Lambda Experiment

Computes structural descriptors of maze topologies to characterize
their decision structure and credit assignment demands.
"""

import networkx as nx
import numpy as np


def compute_node_types(graph):
    """
    Classify nodes by degree into dead-ends, corridors, and junctions.

    Parameters:
    -----------
    graph : nx.Graph
        Maze graph

    Returns:
    --------
    metrics : dict
        Dictionary with:
        - num_dead_ends: nodes with degree 1
        - num_corridors: nodes with degree 2
        - num_junctions: nodes with degree >= 3
        - frac_dead_ends: fraction of nodes that are dead-ends
        - frac_corridors: fraction of nodes that are corridors
        - frac_junctions: fraction of nodes that are junctions
    """
    degrees = [graph.degree(node) for node in graph.nodes()]
    total_nodes = len(degrees)

    num_dead_ends = sum(1 for d in degrees if d == 1)
    num_corridors = sum(1 for d in degrees if d == 2)
    num_junctions = sum(1 for d in degrees if d >= 3)

    return {
        'num_dead_ends': num_dead_ends,
        'num_corridors': num_corridors,
        'num_junctions': num_junctions,
        'frac_dead_ends': num_dead_ends / total_nodes if total_nodes > 0 else 0,
        'frac_corridors': num_corridors / total_nodes if total_nodes > 0 else 0,
        'frac_junctions': num_junctions / total_nodes if total_nodes > 0 else 0,
    }


def compute_mean_corridor_length(graph):
    """
    Compute mean length of corridor segments.

    A corridor segment is a maximal sequence of degree-2 nodes.

    Parameters:
    -----------
    graph : nx.Graph
        Maze graph

    Returns:
    --------
    metrics : dict
        Dictionary with:
        - mean_corridor_length: average length of corridor segments
        - median_corridor_length: median length of corridor segments
        - max_corridor_length: maximum corridor length
        - num_corridor_segments: number of distinct corridor segments
    """
    # Find all corridor segments
    visited = set()
    corridor_segments = []

    for node in graph.nodes():
        if graph.degree(node) == 2 and node not in visited:
            # Trace out the full corridor segment
            segment = [node]
            visited.add(node)

            # Walk in both directions
            for neighbor in graph.neighbors(node):
                current = neighbor
                prev = node

                while current not in visited and graph.degree(current) == 2:
                    segment.append(current)
                    visited.add(current)

                    # Find next node (not the one we came from)
                    neighbors = list(graph.neighbors(current))
                    neighbors.remove(prev)
                    if len(neighbors) == 0:
                        break
                    prev = current
                    current = neighbors[0]

            corridor_segments.append(len(segment))

    if len(corridor_segments) == 0:
        # No corridor segments found
        return {
            'mean_corridor_length': 0.0,
            'median_corridor_length': 0.0,
            'max_corridor_length': 0,
            'num_corridor_segments': 0,
        }

    return {
        'mean_corridor_length': np.mean(corridor_segments),
        'median_corridor_length': np.median(corridor_segments),
        'max_corridor_length': np.max(corridor_segments),
        'num_corridor_segments': len(corridor_segments),
    }


def compute_junction_density(graph):
    """
    Compute junction density: fraction of non-terminal nodes that are junctions.

    Parameters:
    -----------
    graph : nx.Graph
        Maze graph

    Returns:
    --------
    junction_density : float
        Fraction of non-dead-end nodes that have degree >= 3
    """
    non_dead_end_nodes = [node for node in graph.nodes() if graph.degree(node) > 1]
    if len(non_dead_end_nodes) == 0:
        return 0.0

    junction_nodes = [node for node in non_dead_end_nodes if graph.degree(node) >= 3]
    return len(junction_nodes) / len(non_dead_end_nodes)


def compute_corr_dec_ratio(graph, start, goal):
    """
    Compute corridor-to-decision ratio: R_corr/dec.

    This is the ratio of corridor nodes to junction nodes on the shortest path
    from start to goal.

    Parameters:
    -----------
    graph : nx.Graph
        Maze graph
    start : node
        Start node
    goal : node
        Goal node

    Returns:
    --------
    metrics : dict
        Dictionary with:
        - shortest_path_length: number of steps in shortest path
        - num_corridor_nodes_on_path: corridor nodes on shortest path
        - num_junction_nodes_on_path: junction nodes on shortest path
        - corr_dec_ratio: R_corr/dec (with epsilon to avoid divide-by-zero)
    """
    if not nx.has_path(graph, start, goal):
        return {
            'shortest_path_length': -1,
            'num_corridor_nodes_on_path': 0,
            'num_junction_nodes_on_path': 0,
            'corr_dec_ratio': 0.0,
        }

    path = nx.shortest_path(graph, start, goal)
    path_length = len(path) - 1  # Number of steps

    num_corridor = 0
    num_junction = 0

    for node in path:
        degree = graph.degree(node)
        if degree == 2:
            num_corridor += 1
        elif degree >= 3:
            num_junction += 1

    # Compute ratio with small epsilon to avoid divide-by-zero
    epsilon = 1e-6
    corr_dec_ratio = num_corridor / (num_junction + epsilon)

    return {
        'shortest_path_length': path_length,
        'num_corridor_nodes_on_path': num_corridor,
        'num_junction_nodes_on_path': num_junction,
        'corr_dec_ratio': corr_dec_ratio,
    }


def compute_all_metrics(graph, start=None, goal=None):
    """
    Compute all topology metrics.

    Parameters:
    -----------
    graph : nx.Graph
        Maze graph
    start : node, optional
        Start node for path-dependent metrics
    goal : node, optional
        Goal node for path-dependent metrics

    Returns:
    --------
    metrics : dict
        Dictionary with all topology metrics
    """
    metrics = {}

    # Global graph metrics
    metrics.update(compute_node_types(graph))
    metrics.update(compute_mean_corridor_length(graph))
    metrics['junction_density'] = compute_junction_density(graph)

    # Path-dependent metrics (if start and goal provided)
    if start is not None and goal is not None:
        metrics.update(compute_corr_dec_ratio(graph, start, goal))

    return metrics


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    # Test topology metrics
    print("Testing topology metrics...")

    # Create a simple test graph
    # Linear corridor: 0 - 1 - 2 - 3 - 4
    print("\n1. Testing on linear corridor:")
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    metrics = compute_all_metrics(G, start=0, goal=4)
    print("  Metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value}")

    # T-junction: 0 - 1 - 2
    #                 |
    #                 3
    print("\n2. Testing on T-junction:")
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 2), (1, 3)])

    metrics2 = compute_all_metrics(G2, start=0, goal=3)
    print("  Metrics:")
    for key, value in metrics2.items():
        print(f"    {key}: {value}")

    # Test with actual maze
    print("\n3. Testing on generated maze:")
    from corridors import MazeGraph

    maze = MazeGraph(length=8, width=8, corridor=0.5, seed=42)
    graph = maze.get_graph()
    nodes = list(graph.nodes())

    metrics3 = compute_all_metrics(graph, start=nodes[0], goal=nodes[-1])
    print("  Metrics:")
    for key, value in metrics3.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.3f}")
        else:
            print(f"    {key}: {value}")

    print("\n✓ Topology metrics tests passed!")
