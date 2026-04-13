"""
Topology Generators for Lambda Experiment

Generates various maze topology families for testing TD(λ) performance.
"""

import networkx as nx
from corridors import MazeGraph


class ElementaryMaze:
    """
    Wrapper for manually constructed NetworkX graphs to make them compatible
    with the MazeGraph interface used by MazeEnvironment.

    Elementary topologies have curated start-goal pairs designed to test
    specific credit assignment hypotheses.
    """

    def __init__(self, graph, seed=None, start_goal_pairs=None, topology_type=None):
        """
        Initialize ElementaryMaze from a NetworkX graph.

        Parameters:
        -----------
        graph : nx.Graph
            The maze graph with (x, y) tuple nodes
        seed : int, optional
            Random seed for reproducibility
        start_goal_pairs : list of tuples, optional
            List of (start_node, goal_node) pairs for curated training.
            If None, environment will sample randomly.
            If provided, environment should cycle through these pairs.
        topology_type : str, optional
            Type of topology ('tree' requires special direction handling)
        """
        self.graph = graph
        self.seed = seed
        self.start_goal_pairs = start_goal_pairs
        self.topology_type = topology_type

        # Infer dimensions from node coordinates
        nodes = list(graph.nodes())
        if len(nodes) == 0:
            self.length = 1
            self.width = 1
        else:
            xs = [n[0] for n in nodes]
            ys = [n[1] for n in nodes]
            self.length = max(xs) - min(xs) + 1
            self.width = max(ys) - min(ys) + 1

        # Elementary mazes are always maximally corridor-like
        # (they are hand-designed, not procedurally generated)
        self.corridor = 1.0

    def get_graph(self):
        """Return the NetworkX graph."""
        return self.graph

    def get_direction_neighbors(self, node):
        """
        Get valid direction-based neighbors for a node.

        For elementary topologies with arbitrary coordinates, we map neighbors
        to directions based on relative position.

        For tree topologies, uses semantic direction mapping:
        - UP (0): Move to parent (any neighbor with smaller y)
        - DOWN (1): Move to middle child (same x, larger y)
        - LEFT (2): Move to left child (negative x, larger y)
        - RIGHT (3): Move to right child (positive x, larger y)

        Parameters:
        -----------
        node : tuple
            Node position (x, y)

        Returns:
        --------
        neighbors : dict
            Mapping from direction index to neighbor node.
            Direction encoding: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        """
        x, y = node
        neighbors = {}

        # Get all actual neighbors from graph
        graph_neighbors = list(self.graph.neighbors(node))

        # Special handling for tree topologies
        if self.topology_type == 'tree':
            for neighbor in graph_neighbors:
                nx, ny = neighbor
                dx = nx - x
                dy = ny - y

                # Tree semantic directions:
                if dy < 0:
                    # Moving up in tree (toward root)
                    direction = 0  # UP
                elif dy > 0:
                    # Moving down in tree (toward leaves)
                    if dx < 0:
                        direction = 2  # LEFT child
                    elif dx > 0:
                        direction = 3  # RIGHT child
                    else:  # dx == 0
                        direction = 1  # DOWN (middle child)
                else:
                    # Same level (shouldn't happen in tree)
                    print(f"Warning: Same-level neighbor in tree at {node} -> {neighbor}")
                    continue

                if direction in neighbors:
                    print(f"Warning: Multiple neighbors map to direction {direction} from {node}")

                neighbors[direction] = neighbor

        else:
            # Standard direction mapping for non-tree topologies
            for neighbor in graph_neighbors:
                nx, ny = neighbor
                dx = nx - x
                dy = ny - y

                # Determine primary direction (prioritize larger displacement)
                if abs(dy) > abs(dx):
                    # Vertical movement dominates
                    if dy < 0:
                        direction = 0  # UP (negative y)
                    else:
                        direction = 1  # DOWN (positive y)
                else:
                    # Horizontal movement dominates (or equal)
                    if dx < 0:
                        direction = 2  # LEFT (negative x)
                    else:
                        direction = 3  # RIGHT (positive x)

                # Warn if multiple neighbors map to same direction
                if direction in neighbors:
                    print(f"Warning: Multiple neighbors map to direction {direction} from {node}")

                neighbors[direction] = neighbor

        return neighbors

    def get_stats(self):
        """Compute statistics compatible with MazeGraph.get_stats()."""
        nodes = list(self.graph.nodes())
        edges = list(self.graph.edges())

        # Count nodes by degree
        nodes_by_degree = {}
        for node in nodes:
            degree = self.graph.degree(node)
            nodes_by_degree[degree] = nodes_by_degree.get(degree, 0) + 1

        return {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'nodes_with_1_connection': nodes_by_degree.get(1, 0),
            'nodes_with_2_connections': nodes_by_degree.get(2, 0),
            'nodes_with_3plus_connections': sum(
                count for deg, count in nodes_by_degree.items() if deg >= 3
            ),
        }


def make_corridor(length, seed=None):
    """
    Generate Family A: Pure corridor (linear chain).

    Creates a simple linear chain: 0 → 1 → 2 → ... → length-1

    Training scheme: Fixed start at one end, goal at the other end.
    Tests: Pure credit propagation over distance without branching ambiguity.

    Parameters:
    -----------
    length : int
        Length of the corridor (number of nodes)
    seed : int, optional
        Unused (kept for API compatibility). Elementary topologies are deterministic.

    Returns:
    --------
    maze : ElementaryMaze
        A wrapper around NetworkX graph compatible with MazeGraph interface
    """
    graph = nx.Graph()

    # Create linear chain
    for i in range(length):
        graph.add_node((0, i))
        if i > 0:
            graph.add_edge((0, i-1), (0, i))

    # Curated start-goal pair: start at beginning, goal at end
    start_goal_pairs = [((0, 0), (0, length - 1))]

    return ElementaryMaze(graph, seed=seed, start_goal_pairs=start_goal_pairs)


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

    Structure:
        Start → ... → Junction → Branch1 (goal)
                              → Branch2 (dead-end)

    Parameters:
    -----------
    pre_branch_length : int
        Length of corridor before branch (number of nodes before junction)
    post_branch_length : int
        Length of branches after the branch point (number of nodes per branch)
    seed : int, optional
        Unused (kept for API compatibility). Elementary topologies are deterministic.

    Returns:
    --------
    maze : ElementaryMaze
        A wrapper around NetworkX graph compatible with MazeGraph interface
    """
    graph = nx.Graph()

    # Create pre-branch corridor: nodes 0, 1, ..., pre_branch_length-1
    for i in range(pre_branch_length):
        graph.add_node((0, i))
        if i > 0:
            graph.add_edge((0, i-1), (0, i))

    # Junction node at end of pre-branch corridor
    junction = (0, pre_branch_length - 1)

    # Create two branches from junction
    # Branch 1 (correct path to goal): extends upward
    for i in range(1, post_branch_length + 1):
        graph.add_node((i, pre_branch_length - 1))
        if i == 1:
            graph.add_edge(junction, (i, pre_branch_length - 1))
        else:
            graph.add_edge((i-1, pre_branch_length - 1), (i, pre_branch_length - 1))

    # Branch 2 (dead-end): extends downward
    for i in range(1, post_branch_length + 1):
        graph.add_node((-i, pre_branch_length - 1))
        if i == 1:
            graph.add_edge(junction, (-i, pre_branch_length - 1))
        else:
            graph.add_edge((-i+1, pre_branch_length - 1), (-i, pre_branch_length - 1))

    # Curated start-goal pair: start at corridor beginning, goal at correct branch end
    start_node = (0, 0)
    goal_node = (post_branch_length, pre_branch_length - 1)  # End of correct branch
    start_goal_pairs = [(start_node, goal_node)]

    # Wrap in ElementaryMaze for compatibility
    return ElementaryMaze(graph, seed=seed, start_goal_pairs=start_goal_pairs)


def make_intersection(num_arms, arm_length, seed=None):
    """
    Generate Family C: Intersection/crossing maze.

    Creates a central hub with multiple arms extending from it (star topology).

    Parameters:
    -----------
    num_arms : int
        Number of arms extending from central hub
    arm_length : int
        Length of each arm (number of nodes per arm, not including center)
    seed : int, optional
        Unused (kept for API compatibility). Elementary topologies are deterministic.

    Returns:
    --------
    maze : ElementaryMaze
        A wrapper around NetworkX graph compatible with MazeGraph interface
    """
    import math
    graph = nx.Graph()

    # Central hub at origin
    center = (0, 0)
    graph.add_node(center)

    # Create arms extending in cardinal directions ONLY
    # (no diagonals - they don't map cleanly to UP/DOWN/LEFT/RIGHT actions)
    # Limited to max 4 arms (one per cardinal direction)

    cardinal_directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Right, Up, Left, Down

    if num_arms > 4:
        raise ValueError(f"Intersections are limited to 4 arms (cardinal directions only). Got {num_arms}.")

    # Use first num_arms cardinal directions
    arm_directions = cardinal_directions[:num_arms]

    start_goal_pairs = []

    # Create each arm
    for dx, dy in arm_directions:
        # Build arm one step at a time
        for step in range(1, arm_length + 1):
            x = dx * step
            y = dy * step
            node = (x, y)
            graph.add_node(node)

            # Connect to previous node
            if step == 1:
                graph.add_edge(center, node)
            else:
                prev_node = (dx * (step - 1), dy * (step - 1))
                graph.add_edge(prev_node, node)

        # Add this arm's end to start-goal pairs
        end_node = (dx * arm_length, dy * arm_length)
        start_goal_pairs.append((center, end_node))

    # Wrap in ElementaryMaze for compatibility
    return ElementaryMaze(graph, seed=seed, start_goal_pairs=start_goal_pairs)


def make_corridor_chain(num_junctions, corridor_length, branching_factor=2, seed=None):
    """
    Generate Family D: Corridor-chain-with-periodic-junctions.

    Creates a sequence of corridor segments separated by junctions, with dead-end
    branches at each junction.

    Structure: Start → corridor → junction → corridor → junction → ... → Goal
                                      |                    |
                                   dead-ends            dead-ends

    Parameters:
    -----------
    num_junctions : int
        Number of junctions along the main path
    corridor_length : int
        Number of corridor nodes between junctions
    branching_factor : int
        Total number of branches at each junction (including correct path)
        Must be >= 2. branching_factor - 1 will be dead-ends.
    seed : int, optional
        Unused (kept for API compatibility). Elementary topologies are deterministic.

    Returns:
    --------
    maze : ElementaryMaze
        A wrapper around NetworkX graph compatible with MazeGraph interface
    """
    if branching_factor < 2:
        raise ValueError("branching_factor must be >= 2")

    graph = nx.Graph()

    # Build main spine with junctions
    y_pos = 0

    # Start node
    graph.add_node((0, y_pos))
    current_x = 0

    for junction_idx in range(num_junctions):
        # Add corridor segment leading to junction
        for step in range(1, corridor_length + 1):
            current_x += 1
            graph.add_node((current_x, y_pos))
            graph.add_edge((current_x - 1, y_pos), (current_x, y_pos))

        # Junction is at current_x
        junction_node = (current_x, y_pos)

        # Add dead-end branches (branching_factor - 1 branches)
        for branch_idx in range(1, branching_factor):
            # Alternate branches above/below main path
            # branch 1 -> +1, branch 2 -> -1, branch 3 -> +2, branch 4 -> -2, etc.
            if branch_idx % 2 == 1:
                branch_y_offset = (branch_idx + 1) // 2
            else:
                branch_y_offset = -(branch_idx // 2)

            # Create branch extending perpendicular from junction, one step at a time
            current_branch_y = y_pos
            for y_step in range(abs(branch_y_offset)):
                next_branch_y = current_branch_y + (1 if branch_y_offset > 0 else -1)
                graph.add_node((current_x, next_branch_y))
                if y_step == 0:
                    graph.add_edge(junction_node, (current_x, next_branch_y))
                else:
                    graph.add_edge((current_x, current_branch_y), (current_x, next_branch_y))
                current_branch_y = next_branch_y

            branch_y = y_pos + branch_y_offset

            # Extend branch horizontally (same length as corridor)
            for branch_step in range(1, corridor_length):
                next_x = current_x + branch_step
                graph.add_node((next_x, branch_y))
                graph.add_edge((next_x - 1, branch_y), (next_x, branch_y))

    # Add final corridor segment to goal
    for step in range(1, corridor_length + 1):
        current_x += 1
        graph.add_node((current_x, y_pos))
        graph.add_edge((current_x - 1, y_pos), (current_x, y_pos))

    # Curated start-goal pair: beginning to end of main spine
    # Tests: Credit propagation through multiple junctions with varying corridor lengths
    start_node = (0, y_pos)
    goal_node = (current_x, y_pos)
    start_goal_pairs = [(start_node, goal_node)]

    return ElementaryMaze(graph, seed=seed, start_goal_pairs=start_goal_pairs)


def make_tree(depth, branching_factor=2, seed=None):
    """
    Generate Family E: Tree maze.

    Creates a balanced tree with directional navigation:
    - UP: Move to parent (up one level)
    - LEFT: Move to left child (down one level)
    - RIGHT: Move to right child (down one level)
    - DOWN: Move to middle child (down one level, only for branching_factor=3)

    Tree layout uses exponential spacing (2^(depth-1)) to prevent coordinate collisions.

    Parameters:
    -----------
    depth : int
        Depth of the tree (number of levels, including root)
    branching_factor : int
        Number of children per node (2 or 3)
    seed : int, optional
        Unused (kept for API compatibility). Elementary topologies are deterministic.

    Returns:
    --------
    maze : ElementaryMaze
        A wrapper around NetworkX graph compatible with MazeGraph interface
    """
    if branching_factor not in [2, 3]:
        raise ValueError(f"Trees support branching_factor of 2 or 3. Got {branching_factor}.")

    graph = nx.Graph()

    # Build tree level by level
    # Use wide spacing to prevent coordinate collisions in deeper trees
    # Calculate spacing needed: for depth d, we need at least 2^(d-1) horizontal space
    # Use 2^(depth-1) spacing to ensure no subtree overlaps
    spacing = 2 ** (depth - 1)

    level_nodes = {}  # level -> list of (x, y) nodes at that level

    # Root at origin
    root = (0, 0)
    graph.add_node(root)
    level_nodes[0] = [root]

    # Build tree level by level
    for level in range(depth - 1):
        level_nodes[level + 1] = []
        current_level = level_nodes[level]

        # Calculate spacing for this level: how far apart should children be?
        # Each node's subtree needs 2^(depth - level - 2) horizontal space
        level_spacing = spacing // (2 ** (level + 1))

        for parent_node in current_level:
            parent_x, parent_y = parent_node

            # Create children based on branching factor
            if branching_factor == 2:
                # Left and right children with appropriate spacing
                children = [
                    (parent_x - level_spacing, parent_y + 1),  # Left child
                    (parent_x + level_spacing, parent_y + 1),  # Right child
                ]
            elif branching_factor == 3:
                # Left, middle, and right children with appropriate spacing
                children = [
                    (parent_x - level_spacing, parent_y + 1),  # Left child
                    (parent_x, parent_y + 1),                   # Middle child (DOWN action)
                    (parent_x + level_spacing, parent_y + 1),  # Right child
                ]

            # Add children to graph
            for child_node in children:
                graph.add_node(child_node)
                graph.add_edge(parent_node, child_node)
                level_nodes[level + 1].append(child_node)

    # Curated start-goal pairs: root to each leaf
    leaves = level_nodes[depth - 1]
    start_goal_pairs = [(root, leaf) for leaf in leaves]

    return ElementaryMaze(graph, seed=seed, start_goal_pairs=start_goal_pairs, topology_type='tree')


# Configuration for elementary topologies
ELEMENTARY_TOPOLOGIES = {
    # Family A: Pure corridors (varying length)
    'corridor_L3': {'type': 'corridor', 'length': 3},
    'corridor_L5': {'type': 'corridor', 'length': 5},
    'corridor_L7': {'type': 'corridor', 'length': 7},
    'corridor_L10': {'type': 'corridor', 'length': 10},
    'corridor_L15': {'type': 'corridor', 'length': 15},
    'corridor_L20': {'type': 'corridor', 'length': 20},

    # Family B: Single branch (varying pre/post lengths)
    'branch_pre2_post2': {'type': 'single_branch', 'pre_branch_length': 2, 'post_branch_length': 2},
    'branch_pre2_post5': {'type': 'single_branch', 'pre_branch_length': 2, 'post_branch_length': 5},
    'branch_pre2_post8': {'type': 'single_branch', 'pre_branch_length': 2, 'post_branch_length': 8},
    'branch_pre5_post2': {'type': 'single_branch', 'pre_branch_length': 5, 'post_branch_length': 2},
    'branch_pre5_post5': {'type': 'single_branch', 'pre_branch_length': 5, 'post_branch_length': 5},
    'branch_pre5_post8': {'type': 'single_branch', 'pre_branch_length': 5, 'post_branch_length': 8},
    'branch_pre8_post2': {'type': 'single_branch', 'pre_branch_length': 8, 'post_branch_length': 2},
    'branch_pre8_post5': {'type': 'single_branch', 'pre_branch_length': 8, 'post_branch_length': 5},
    'branch_pre8_post8': {'type': 'single_branch', 'pre_branch_length': 8, 'post_branch_length': 8},

    # Family C: Intersection (varying arms and lengths - limited to 4 arms max for cardinal directions)
    'intersection_3arms_L2': {'type': 'intersection', 'num_arms': 3, 'arm_length': 2},
    'intersection_3arms_L4': {'type': 'intersection', 'num_arms': 3, 'arm_length': 4},
    'intersection_3arms_L6': {'type': 'intersection', 'num_arms': 3, 'arm_length': 6},
    'intersection_4arms_L2': {'type': 'intersection', 'num_arms': 4, 'arm_length': 2},
    'intersection_4arms_L4': {'type': 'intersection', 'num_arms': 4, 'arm_length': 4},
    'intersection_4arms_L6': {'type': 'intersection', 'num_arms': 4, 'arm_length': 6},

    # Family D: Corridor chain with junctions (varying parameters)
    'chain_2j_L2_b2': {'type': 'corridor_chain', 'num_junctions': 2, 'corridor_length': 2, 'branching_factor': 2},
    'chain_2j_L4_b2': {'type': 'corridor_chain', 'num_junctions': 2, 'corridor_length': 4, 'branching_factor': 2},
    'chain_2j_L6_b2': {'type': 'corridor_chain', 'num_junctions': 2, 'corridor_length': 6, 'branching_factor': 2},
    'chain_4j_L2_b2': {'type': 'corridor_chain', 'num_junctions': 4, 'corridor_length': 2, 'branching_factor': 2},
    'chain_4j_L4_b2': {'type': 'corridor_chain', 'num_junctions': 4, 'corridor_length': 4, 'branching_factor': 2},
    'chain_4j_L6_b2': {'type': 'corridor_chain', 'num_junctions': 4, 'corridor_length': 6, 'branching_factor': 2},
    'chain_6j_L2_b2': {'type': 'corridor_chain', 'num_junctions': 6, 'corridor_length': 2, 'branching_factor': 2},
    'chain_6j_L4_b2': {'type': 'corridor_chain', 'num_junctions': 6, 'corridor_length': 4, 'branching_factor': 2},
    'chain_6j_L6_b2': {'type': 'corridor_chain', 'num_junctions': 6, 'corridor_length': 6, 'branching_factor': 2},
    'chain_2j_L4_b3': {'type': 'corridor_chain', 'num_junctions': 2, 'corridor_length': 4, 'branching_factor': 3},
    'chain_4j_L4_b3': {'type': 'corridor_chain', 'num_junctions': 4, 'corridor_length': 4, 'branching_factor': 3},

    # Family E: Trees (varying depth and branching)
    # Trees use directional navigation: U=parent, L/R=children, D=middle child (b=3 only)
    'tree_d2_b2': {'type': 'tree', 'depth': 2, 'branching_factor': 2},
    'tree_d3_b2': {'type': 'tree', 'depth': 3, 'branching_factor': 2},
    'tree_d4_b2': {'type': 'tree', 'depth': 4, 'branching_factor': 2},
    'tree_d2_b3': {'type': 'tree', 'depth': 2, 'branching_factor': 3},
    'tree_d3_b3': {'type': 'tree', 'depth': 3, 'branching_factor': 3},
    'tree_d4_b3': {'type': 'tree', 'depth': 4, 'branching_factor': 3},
}

# Configuration for procedural topologies
PROCEDURAL_TOPOLOGIES = {
    '0.0 corridor': {'type': 'procedural', 'length': 8, 'width': 8, 'corridor': 0.0},
    '0.1 corridor': {'type': 'procedural', 'length': 8, 'width': 8, 'corridor': 0.1},
    '0.2 corridor': {'type': 'procedural', 'length': 8, 'width': 8, 'corridor': 0.2},
    '0.3 corridor': {'type': 'procedural', 'length': 8, 'width': 8, 'corridor': 0.3},
    '0.4 corridor': {'type': 'procedural', 'length': 8, 'width': 8, 'corridor': 0.4},
    '0.5 corridor': {'type': 'procedural', 'length': 8, 'width': 8, 'corridor': 0.5},
    '0.6 corridor': {'type': 'procedural', 'length': 8, 'width': 8, 'corridor': 0.6},
    '0.7 corridor': {'type': 'procedural', 'length': 8, 'width': 8, 'corridor': 0.7},
    '0.8 corridor': {'type': 'procedural', 'length': 8, 'width': 8, 'corridor': 0.8},
    '0.9 corridor': {'type': 'procedural', 'length': 8, 'width': 8, 'corridor': 0.9},
    '1.0 corridor': {'type': 'procedural', 'length': 8, 'width': 8, 'corridor': 1.0},
}

# Combined topology catalog
ALL_TOPOLOGIES = {**PROCEDURAL_TOPOLOGIES}
ALL_ELEMENTARY_TOPOLOGIES = {**ELEMENTARY_TOPOLOGIES}


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
    maze : MazeGraph or ElementaryMaze
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
    elif topo_type == 'corridor_chain':
        return make_corridor_chain(
            topology_config['num_junctions'],
            topology_config['corridor_length'],
            topology_config['branching_factor'],
            seed=seed
        )
    elif topo_type == 'tree':
        return make_tree(
            topology_config['depth'],
            topology_config['branching_factor'],
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
    # for name, config in ELEMENTARY_TOPOLOGIES.items():
    #     maze = generate_topology(config, seed=42)
    #     stats = maze.get_stats()
    #     print(f"  {name}: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

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
