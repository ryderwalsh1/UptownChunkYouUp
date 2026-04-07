"""
Topology Metrics for Lambda Experiment

Computes structural descriptors of maze topologies to characterize
their decision structure and credit assignment demands.
"""

import networkx as nx
import numpy as np
from scipy.spatial.distance import jensenshannon


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


def compute_global_corr_dec_ratio(graph):
    """
    Compute average corridor-to-decision ratio across all dead-end pairs.

    This provides a global, topology-wide measure of credit assignment demand
    that is independent of any specific start-goal pair.

    Parameters:
    -----------
    graph : nx.Graph
        Maze graph

    Returns:
    --------
    metrics : dict
        Dictionary with:
        - mean_global_corr_dec_ratio: average R_corr/dec across all dead-end pairs
        - median_global_corr_dec_ratio: median ratio
        - std_global_corr_dec_ratio: standard deviation
        - num_dead_end_pairs: number of pairs analyzed
    """
    # Find all dead-end nodes (degree == 1)
    dead_ends = [node for node in graph.nodes() if graph.degree(node) == 1]

    if len(dead_ends) < 2:
        # Not enough dead-ends to compute pairs
        return {
            'mean_global_corr_dec_ratio': 0.0,
            'median_global_corr_dec_ratio': 0.0,
            'std_global_corr_dec_ratio': 0.0,
            'num_dead_end_pairs': 0,
        }

    # Compute corridor/decision ratio for all unique dead-end pairs
    ratios = []
    epsilon = 1e-6

    for i in range(len(dead_ends)):
        for j in range(i + 1, len(dead_ends)):
            start = dead_ends[i]
            goal = dead_ends[j]

            # Check if path exists
            if not nx.has_path(graph, start, goal):
                continue

            # Get shortest path
            path = nx.shortest_path(graph, start, goal)

            # Count corridor and junction nodes on path
            num_corridor = 0
            num_junction = 0

            for node in path:
                degree = graph.degree(node)
                if degree == 2:
                    num_corridor += 1
                elif degree >= 3:
                    num_junction += 1

            # Compute ratio
            ratio = num_corridor / (num_junction + epsilon)
            ratios.append(ratio)

    if len(ratios) == 0:
        return {
            'mean_global_corr_dec_ratio': 0.0,
            'median_global_corr_dec_ratio': 0.0,
            'std_global_corr_dec_ratio': 0.0,
            'num_dead_end_pairs': 0,
        }

    return {
        'mean_global_corr_dec_ratio': np.mean(ratios),
        'median_global_corr_dec_ratio': np.median(ratios),
        'std_global_corr_dec_ratio': np.std(ratios),
        'num_dead_end_pairs': len(ratios),
    }


def compute_spatial_homogeneity(graph, maze, window_size=3):
    """
    Compute spatial homogeneity using overlapping window analysis.

    Measures whether the maze has consistent local topology composition throughout
    (homogeneous) vs. patchy regions with different mixtures (heterogeneous).

    Algorithm:
    1. Slide 4×4 window across maze with stride=1
    2. Compute composition vector [dead_end_frac, corridor_frac, junction_frac] per window
    3. Compute mean composition across all windows
    4. For each window, compute Jensen-Shannon divergence from mean
    5. Average JS divergence = heterogeneity
    6. Return homogeneity = 1 - heterogeneity_normalized

    Parameters:
    -----------
    graph : nx.Graph
        Maze graph with (x,y) tuple nodes
    maze : MazeGraph
        Original maze object (for dimensions)
    window_size : int
        Size of sliding window (default: 4)

    Returns:
    --------
    metrics : dict
        - spatial_homogeneity: homogeneity score [0, 1], 1=homogeneous, 0=heterogeneous
        - spatial_heterogeneity: raw heterogeneity score (mean JS divergence)
        - num_windows: number of windows analyzed
    """
    # Check if nodes are 2D grid coordinates
    sample_nodes = list(graph.nodes())[:5]
    if not all(isinstance(n, tuple) and len(n) == 2 for n in sample_nodes):
        # Not a 2D grid graph, cannot compute spatial metrics
        return {
            'spatial_homogeneity': 1.0,  # Default to homogeneous if not grid
            'spatial_heterogeneity': 0.0,
            'num_windows': 0
        }

    # Get maze dimensions
    max_x = max(node[0] for node in graph.nodes())
    max_y = max(node[1] for node in graph.nodes())

    # Collect compositions from all overlapping windows
    window_compositions = []

    # Slide window across maze (stride=1 for full overlap)
    for start_x in range(max_x - window_size + 2):  # +2 to include boundary
        for start_y in range(max_y - window_size + 2):
            # Define window bounds
            end_x = start_x + window_size
            end_y = start_y + window_size

            # Get nodes in this window
            window_nodes = [
                node for node in graph.nodes()
                if isinstance(node, tuple) and len(node) == 2
                and start_x <= node[0] < end_x
                and start_y <= node[1] < end_y
            ]

            if len(window_nodes) < 3:  # Skip tiny windows
                continue

            # Count node types in window
            num_dead_ends = sum(1 for n in window_nodes if graph.degree(n) == 1)
            num_corridors = sum(1 for n in window_nodes if graph.degree(n) == 2)
            num_junctions = sum(1 for n in window_nodes if graph.degree(n) >= 3)

            total = len(window_nodes)

            # Composition vector (probability distribution)
            composition = np.array([
                num_dead_ends / total,
                num_corridors / total,
                num_junctions / total
            ])

            window_compositions.append(composition)

    if len(window_compositions) < 2:
        # Not enough windows
        return {
            'spatial_homogeneity': 1.0,
            'spatial_heterogeneity': 0.0,
            'num_windows': len(window_compositions)
        }

    # Compute mean composition across all windows
    mean_composition = np.mean(window_compositions, axis=0)

    # Method 1: Jensen-Shannon divergence (distribution-based)
    js_divergences = []
    for composition in window_compositions:
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        comp_safe = composition + eps
        mean_safe = mean_composition + eps

        # Normalize to ensure valid probability distributions
        comp_safe = comp_safe / comp_safe.sum()
        mean_safe = mean_safe / mean_safe.sum()

        # jensenshannon returns sqrt of JS divergence, so square it
        js_div = jensenshannon(comp_safe, mean_safe) ** 2
        js_divergences.append(js_div)

    # Use 95th percentile instead of mean for more sensitivity
    # This makes the metric more discriminatory - even one very different region matters
    heterogeneity_js = np.percentile(js_divergences, 95)

    # Method 2: Standard deviation of composition components (variance-based)
    # More sensitive to local variations than JS divergence
    window_comps_array = np.array(window_compositions)

    # Compute std for each composition dimension: [dead_end, corridor, junction]
    std_dead_end = np.std(window_comps_array[:, 0])
    std_corridor = np.std(window_comps_array[:, 1])
    std_junction = np.std(window_comps_array[:, 2])

    # Theoretical max std for a binary [0,1] variable is 0.5 (at 50/50 split)
    # Normalize by this to get [0,1] range
    max_theoretical_std = 0.5
    heterogeneity_std = np.mean([
        std_dead_end / max_theoretical_std,
        std_corridor / max_theoretical_std,
        std_junction / max_theoretical_std
    ])

    # Combined heterogeneity: weighted average of both methods
    # Std-based is more sensitive, so weight it higher
    heterogeneity = 0.3 * heterogeneity_js + 0.7 * heterogeneity_std

    # Homogeneity = 1 - heterogeneity
    homogeneity = 1.0 - heterogeneity

    return {
        'spatial_homogeneity': homogeneity,
        'spatial_heterogeneity': heterogeneity,
        'spatial_heterogeneity_js': heterogeneity_js,
        'spatial_heterogeneity_std': heterogeneity_std,
        'num_windows': len(window_compositions)
    }


def compute_all_metrics(graph, start=None, goal=None, maze=None):
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
    maze : MazeGraph, optional
        Original maze object (needed for spatial metrics)

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
    metrics.update(compute_global_corr_dec_ratio(graph))

    # Spatial homogeneity metrics (if maze object provided)
    if maze is not None:
        metrics.update(compute_spatial_homogeneity(graph, maze))

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
