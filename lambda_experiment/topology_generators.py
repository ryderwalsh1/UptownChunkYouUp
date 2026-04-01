"""
Topology Generators for Lambda Experiment

Generates various maze topology families for testing TD(λ) performance.
"""

import networkx as nx
from corridors import MazeGraph


def make_corridor(length, seed=None):
    """
    Generate Family A: Pure corridor (linear chain).

    Creates a 1×length maze which should produce a simple linear path.

    Parameters:
    -----------
    length : int
        Length of the corridor (number of nodes)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    maze : MazeGraph
        A 1×length maze with corridor parameter = 1.0
    """
    # Use corridor=1.0 to maximize corridor-likeness (minimize branching)
    # With 1×length grid, this should create a linear path
    maze = MazeGraph(length=1, width=length, corridor=1.0, seed=seed)
    return maze


def make_procedural_maze(length, width, corridor, seed=None):
    """
    Generate Family D: Procedural maze with parameterized corridor-likeness.

    Parameters:
    -----------
    length : int
        Height of the maze grid
    width : int
        Width of the maze grid
    corridor : float
        Corridor parameter from 0 to 1
        0 = maximally junction-heavy
        1 = maximally corridor-like
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    maze : MazeGraph
        A maze with specified corridor parameter
    """
    maze = MazeGraph(length=length, width=width, corridor=corridor, seed=seed)
    return maze


def make_single_branch(pre_branch_length, post_branch_length, seed=None):
    """
    Generate Family B: Single branch maze.

    Creates a corridor leading to one branch point, with one correct and one wrong branch.

    NOTE: This is a simplified implementation. For more complex branching structures,
    you may need to manually construct the graph.

    Parameters:
    -----------
    pre_branch_length : int
        Length of corridor before branch
    post_branch_length : int
        Length of branches after the branch point
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    graph : nx.Graph
        A NetworkX graph representing the single branch maze
    """
    # TODO: Implement manual graph construction for single branch
    # For now, return a simple procedural maze with low corridor parameter
    # This is a placeholder until manual graph construction is implemented
    raise NotImplementedError("Single branch topology not yet implemented")


def make_intersection(num_arms, arm_length, seed=None):
    """
    Generate Family C: Intersection/crossing maze.

    Creates a central hub with multiple arms extending from it.

    Parameters:
    -----------
    num_arms : int
        Number of arms extending from central hub
    arm_length : int
        Length of each arm
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    graph : nx.Graph
        A NetworkX graph representing the intersection maze
    """
    # TODO: Implement manual graph construction for intersection
    # This is a placeholder until manual graph construction is implemented
    raise NotImplementedError("Intersection topology not yet implemented")


# Configuration for elementary topologies
# ELEMENTARY_TOPOLOGIES = {
#     'corridor_short': {'type': 'corridor', 'length': 5},
#     'corridor_medium': {'type': 'corridor', 'length': 10},
#     'corridor_long': {'type': 'corridor', 'length': 15},
#     'corridor_verylong': {'type': 'corridor', 'length': 20},
# }

# Configuration for procedural topologies
PROCEDURAL_TOPOLOGIES = {
    'maze_0.0': {'type': 'procedural', 'length': 5, 'width': 5, 'corridor': 0.0},
    'maze_0.1': {'type': 'procedural', 'length': 5, 'width': 5, 'corridor': 0.1},
    'maze_0.2': {'type': 'procedural', 'length': 5, 'width': 5, 'corridor': 0.2},
    'maze_0.3': {'type': 'procedural', 'length': 5, 'width': 5, 'corridor': 0.3},
    'maze_0.4': {'type': 'procedural', 'length': 5, 'width': 5, 'corridor': 0.4},
    'maze_0.5': {'type': 'procedural', 'length': 5, 'width': 5, 'corridor': 0.5},
    'maze_0.6': {'type': 'procedural', 'length': 5, 'width': 5, 'corridor': 0.6},
    'maze_0.7': {'type': 'procedural', 'length': 5, 'width': 5, 'corridor': 0.7},
    'maze_0.8': {'type': 'procedural', 'length': 5, 'width': 5, 'corridor': 0.8},
    'maze_0.9': {'type': 'procedural', 'length': 5, 'width': 5, 'corridor': 0.9},
    'maze_1.0': {'type': 'procedural', 'length': 5, 'width': 5, 'corridor': 1.0},
}

# Combined topology catalog
ALL_TOPOLOGIES = {**PROCEDURAL_TOPOLOGIES}


def generate_topology(topology_config, seed=None):
    """
    Generate a maze topology from configuration.

    Parameters:
    -----------
    topology_config : dict
        Configuration dictionary with 'type' and topology-specific parameters
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    maze : MazeGraph
        Generated maze
    """
    topo_type = topology_config['type']

    if topo_type == 'corridor':
        return make_corridor(topology_config['length'], seed=seed)
    elif topo_type == 'procedural':
        return make_procedural_maze(
            topology_config['length'],
            topology_config['width'],
            topology_config['corridor'],
            seed=seed
        )
    elif topo_type == 'single_branch':
        return make_single_branch(
            topology_config['pre_branch_length'],
            topology_config['post_branch_length'],
            seed=seed
        )
    elif topo_type == 'intersection':
        return make_intersection(
            topology_config['num_arms'],
            topology_config['arm_length'],
            seed=seed
        )
    else:
        raise ValueError(f"Unknown topology type: {topo_type}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    # Test topology generators
    print("Testing topology generators...")

    # Test corridor
    print("\n1. Testing corridor topology:")
    for name, config in ELEMENTARY_TOPOLOGIES.items():
        maze = generate_topology(config, seed=42)
        stats = maze.get_stats()
        print(f"  {name}: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

    # Test procedural
    print("\n2. Testing procedural topology:")
    for name, config in PROCEDURAL_TOPOLOGIES.items():
        maze = generate_topology(config, seed=42)
        stats = maze.get_stats()
        print(f"  {name}: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
        print(f"    Dead-ends: {stats['nodes_with_1_connection']}, "
              f"Corridors: {stats['nodes_with_2_connections']}, "
              f"Junctions: {stats['nodes_with_3plus_connections']}")

    print("\n✓ Topology generators tests passed!")
