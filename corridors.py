import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx


class MazeGraph:
    def __init__(self, length, width, corridor, seed=None):
        """
        Initialize a maze graph with parameterized corridor-likeness.

        Parameters:
        -----------
        length : int
            Height of the grid (number of rows)
        width : int
            Width of the grid (number of columns)
        corridor : float
            Corridor parameter from 0 to 1.
            0 = maximally junction-heavy (many branching points)
            1 = maximally corridor-like (mostly single paths, minimal branching)
        seed : int, optional
            Random seed for reproducibility
        """
        if not 0 <= corridor <= 1:
            raise ValueError("corridor parameter must be between 0 and 1")

        self.length = length
        self.width = width
        self.corridor = corridor
        self.seed = seed

        if seed is not None:
            random.seed(seed)

        # Graph representation: NetworkX undirected graph
        # Each node is represented as (row, col) tuple
        self.graph = nx.Graph()

        # Initialize all nodes
        self.nodes = [(r, c) for r in range(length) for c in range(width)]
        self.graph.add_nodes_from(self.nodes)

        # Generate the maze
        self._generate_maze()

    def _get_neighbors(self, node):
        """Get all 4-connected neighbors of a node (up, down, left, right)."""
        r, c = node
        neighbors = []

        # Up, Down, Left, Right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.length and 0 <= nc < self.width:
                neighbors.append((nr, nc))

        return neighbors

    def _count_connections(self, node):
        """Count how many connections a node currently has."""
        return self.graph.degree(node)

    def _generate_maze(self):
        """
        Generate maze using modified Prim's algorithm with corridor bias.

        Strategy:
        - Start from a random cell
        - Maintain a frontier of walls (potential edges to add)
        - When selecting which wall to add, bias selection based on corridor parameter:
          * corridor=0: Prefer adding edges that create junctions (connect to well-connected nodes)
          * corridor=1: Prefer adding edges that extend corridors (connect to nodes with few connections)
        """
        # Start from random node
        start = random.choice(self.nodes)
        visited = {start}

        # Frontier: list of (node, neighbor) pairs representing potential edges
        frontier = []
        for neighbor in self._get_neighbors(start):
            frontier.append((start, neighbor))

        while frontier:
            # Select a wall from frontier based on corridor parameter
            edge = self._select_edge(frontier, visited)
            node1, node2 = edge

            # If node2 hasn't been visited, add the edge
            if node2 not in visited:
                self.graph.add_edge(node1, node2)
                visited.add(node2)

                # Add new walls to frontier
                for neighbor in self._get_neighbors(node2):
                    if neighbor not in visited:
                        frontier.append((node2, neighbor))

            # Remove the selected edge from frontier
            frontier.remove(edge)

    def _select_edge(self, frontier, visited):
        """
        Select an edge from frontier based on corridor parameter.

        corridor=0: Prefer edges that connect to nodes with more existing connections (creates junctions)
        corridor=1: Prefer edges that connect to nodes with fewer connections (creates corridors)
        """
        if self.corridor == 0:
            # Pure junction mode: completely random (unbiased)
            return random.choice(frontier)
        elif self.corridor == 1:
            # Pure corridor mode: extremely strong preference for nodes with exactly 1 connection
            weights = []
            for node1, node2 in frontier:
                if node2 not in visited:
                    # Count connections of node1 (the visited node)
                    conn_count = self._count_connections(node1)
                    # Very strong preference for extending from nodes with exactly 1 connection
                    if conn_count == 1:
                        weight = 10000  # Extremely prefer corridor endpoints
                    elif conn_count == 0:
                        weight = 10     # Allow starting new corridors
                    else:
                        weight = 0.001  # Almost completely avoid branching
                    weights.append(weight)
                else:
                    weights.append(1)
            return random.choices(frontier, weights=weights)[0]
        else:
            # Interpolate between the two strategies using extreme exponential scaling
            weights = []
            for node1, node2 in frontier:
                if node2 not in visited:
                    conn_count = self._count_connections(node1)
                    # Use corridor^12 for very dramatic effect - keeps 0.5 weak but 0.9+ very strong
                    corridor_factor = self.corridor*0.4 + 0.35
                    corridor_factor = corridor_factor ** 12

                    if conn_count == 0:
                        weight = 1
                    elif conn_count == 1:
                        # Exponentially prefer single-connection nodes as corridor increases
                        weight = 1 + corridor_factor * 9999
                    else:
                        # Exponentially penalize multi-connection nodes as corridor increases
                        weight = max(0.001, 1 - corridor_factor * 0.999)
                    weights.append(weight)
                else:
                    weights.append(1)
            return random.choices(frontier, weights=weights)[0]

    def visualize(self, title=None, figsize=(10, 10), save_path=None):
        """
        Visualize the maze graph with aesthetic styling.

        Shows nodes as circles and edges as connecting lines with pastel colors.
        """
        # Set up figure with clean white background
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('#F8F9FA')  # Light gray background

        # Aesthetic color palette (soft pastels)
        node_color = '#A8DADC'      # Soft teal
        edge_color = '#457B9D'      # Muted blue
        highlight_color = '#E63946'  # Soft red for dead ends
        wall_color = '#2D2D2D'      # Dark gray for walls

        # Node positions (center of each grid cell)
        pos = {(r, c): (c + 0.5, r + 0.5) for r in range(self.length) for c in range(self.width)}

        # Draw walls first (thin black lines for missing edges)
        wall_thickness = 1.2
        for r in range(self.length):
            for c in range(self.width):
                node = (r, c)

                # Check each direction and draw wall if no edge exists
                # Right wall
                if c < self.width - 1:
                    neighbor = (r, c + 1)
                    if not self.graph.has_edge(node, neighbor):
                        ax.plot([c + 1, c + 1], [r, r + 1],
                               color=wall_color, linewidth=wall_thickness, alpha=0.7, zorder=0)
                else:
                    # Outer boundary
                    ax.plot([c + 1, c + 1], [r, r + 1],
                           color=wall_color, linewidth=wall_thickness, alpha=0.7, zorder=0)

                # Bottom wall
                if r < self.length - 1:
                    neighbor = (r + 1, c)
                    if not self.graph.has_edge(node, neighbor):
                        ax.plot([c, c + 1], [r + 1, r + 1],
                               color=wall_color, linewidth=wall_thickness, alpha=0.7, zorder=0)
                else:
                    # Outer boundary
                    ax.plot([c, c + 1], [r + 1, r + 1],
                           color=wall_color, linewidth=wall_thickness, alpha=0.7, zorder=0)

                # Top wall (only for top row)
                if r == 0:
                    ax.plot([c, c + 1], [r, r],
                           color=wall_color, linewidth=wall_thickness, alpha=0.7, zorder=0)

                # Left wall (only for left column)
                if c == 0:
                    ax.plot([c, c], [r, r + 1],
                           color=wall_color, linewidth=wall_thickness, alpha=0.7, zorder=0)

        # Draw edges (connections between nodes)
        for edge in self.graph.edges():
            node1, node2 = edge
            x1, y1 = pos[node1]
            x2, y2 = pos[node2]
            ax.plot([x1, x2], [y1, y2],
                   color=edge_color,
                   linewidth=2.5,
                   alpha=0.6,
                   solid_capstyle='round',
                   zorder=1)

        # Draw nodes
        node_sizes = []
        node_colors = []

        for node in self.graph.nodes():
            degree = self.graph.degree(node)

            # Color nodes by their connectivity
            if degree == 1:
                # Dead ends - highlight with different color
                node_colors.append(highlight_color)
                node_sizes.append(120)
            elif degree == 2:
                # Corridor nodes
                node_colors.append(node_color)
                node_sizes.append(80)
            else:
                # Junction nodes - larger
                node_colors.append('#1D3557')  # Dark blue for junctions
                node_sizes.append(150)

        # Plot all nodes
        x_coords = [pos[node][0] for node in self.graph.nodes()]
        y_coords = [pos[node][1] for node in self.graph.nodes()]

        ax.scatter(x_coords, y_coords,
                  s=node_sizes,
                  c=node_colors,
                  alpha=0.8,
                  edgecolors='white',
                  linewidths=1.5,
                  zorder=2)

        # Set axis properties
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.length)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

        # Add title with clean typography
        # if title is None:
        #     title = f'Maze Graph (corridor = {self.corridor})'

        ax.text(self.width / 2, -0.5, title,
               fontsize=16,
               fontweight='500',
               ha='center',
               va='top',
               fontfamily='sans-serif')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        # plt.show()

    def get_direction_neighbors(self, node):
        """
        Get valid direction-based neighbors for a node.

        Parameters:
        -----------
        node : tuple
            Node position (row, col)

        Returns:
        --------
        neighbors : dict
            Mapping from direction index to neighbor node.
            Direction encoding: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        """
        r, c = node
        neighbors = {}

        # Direction mappings
        directions = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1)    # RIGHT
        }

        for direction_idx, (dr, dc) in directions.items():
            neighbor = (r + dr, c + dc)
            # Check if neighbor exists and has an edge in the graph
            if neighbor in self.graph.nodes() and self.graph.has_edge(node, neighbor):
                neighbors[direction_idx] = neighbor

        return neighbors

    def get_graph(self):
        """Return the NetworkX graph object."""
        return self.graph

    def get_stats(self):
        """Get statistics about the maze structure."""
        connection_counts = [self.graph.degree(node) for node in self.graph.nodes()]

        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'avg_connections': sum(connection_counts) / len(connection_counts),
            'max_connections': max(connection_counts),
            'min_connections': min(connection_counts),
            'nodes_with_1_connection': sum(1 for c in connection_counts if c == 1),
            'nodes_with_2_connections': sum(1 for c in connection_counts if c == 2),
            'nodes_with_3plus_connections': sum(1 for c in connection_counts if c >= 3),
        }


# Example usage
if __name__ == "__main__":
    import numpy as np
    # Test different corridor values
    corridor_vals = np.arange(11) / 10.0  # Test 11 values from 0 to 1
    # corridor_vals = [0.5]
    all_stats = []

    for corridor_val in corridor_vals:
        maze = MazeGraph(length=5, width=5, corridor=corridor_val, seed=180)
        stats = maze.get_stats()
        all_stats.append(stats)

        print(f"\nCorridor={corridor_val}:")
        print(f"  Avg connections per node: {stats['avg_connections']:.2f}")
        print(f"  Nodes with 1 connection (dead ends): {stats['nodes_with_1_connection']}")
        print(f"  Nodes with 2 connections (corridors): {stats['nodes_with_2_connections']}")
        print(f"  Nodes with 3+ connections (junctions): {stats['nodes_with_3plus_connections']}")

        maze.visualize(save_path=f'results/graphs/corridors/5x5/maze_corridor_{corridor_val:.1f}.png')
        # maze.visualize()

    # Plot statistics across corridor values
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    dead_ends = [s['nodes_with_1_connection'] for s in all_stats]
    corridors = [s['nodes_with_2_connections'] for s in all_stats]
    junctions = [s['nodes_with_3plus_connections'] for s in all_stats]

    x = range(len(corridor_vals))
    width = 0.25

    if True:
        # Aesthetic colors matching the visualization
        ax.bar([i - width for i in x], dead_ends, width, label='Dead ends (degree 1)',
            color='#E63946', alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.bar(x, corridors, width, label='Corridors (degree 2)',
            color='#A8DADC', alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.bar([i + width for i in x], junctions, width, label='Junctions (degree 3+)',
            color='#1D3557', alpha=0.8, edgecolor='white', linewidth=1.5)

        ax.set_xlabel('Corridor Parameter', fontsize=12, fontweight='500')
        ax.set_ylabel('Number of Nodes', fontsize=12, fontweight='500')
        ax.set_title('Node Type Distribution by Corridor Parameter', fontsize=14, fontweight='500')
        ax.set_xticks(x)
        ax.set_xticklabels(corridor_vals)
        ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
        ax.set_facecolor('#F8F9FA')

        plt.tight_layout()
        plt.savefig('results/graphs/corridors/5x5/node_distribution_by_corridor.png', dpi=300, bbox_inches='tight', facecolor='white')
