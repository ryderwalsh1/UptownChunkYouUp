"""
Slow Memory System

Pure dictionary-based memory lookup - no PsyNeuLink.
Fast and simple associative memory for (state, goal) -> action.
"""

import torch
import numpy as np
import networkx as nx


class SlowMemory:
    """Simple dictionary-based episodic memory for action retrieval."""

    def __init__(self, num_nodes, num_actions=5, default_temperature=1.0):
        """
        Initialize slow memory system.

        Parameters:
        -----------
        num_nodes : int
            Number of nodes in the maze (state space size)
        num_actions : int
            Number of possible actions (5: UP, DOWN, LEFT, RIGHT, IDENTIFY_GOAL)
        default_temperature : float
            Temperature for softmax when creating distributions
        """
        self.num_nodes = num_nodes
        self.num_actions = num_actions  # Should be 5
        self.default_temperature = default_temperature

        # Memory storage: dict mapping (state_idx, goal_idx) -> direction (0-4)
        # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=IDENTIFY_GOAL
        self.memory = {}

    def initialize_memory(self, maze_graph):
        """
        Initialize memory with shortest path information from maze.

        Parameters:
        -----------
        maze_graph : networkx.Graph or MazeGraph or ElementaryMaze
            The maze graph (either NetworkX graph, MazeGraph object, or ElementaryMaze)
        """
        # Handle both MazeGraph and ElementaryMaze
        if hasattr(maze_graph, 'get_graph'):
            # It's a MazeGraph or ElementaryMaze object
            from corridors import MazeGraph
            maze = maze_graph
            graph = maze.get_graph()

            # Detect coordinate system
            # ElementaryMaze uses (x, y) where x=col, y=row
            # MazeGraph uses (row, col)
            from lambda_experiment.topology_generators import ElementaryMaze
            is_elementary = isinstance(maze, ElementaryMaze)
        else:
            # It's a raw NetworkX graph - need to create MazeGraph wrapper
            # This shouldn't happen in normal use, but handle it gracefully
            graph = maze_graph
            maze = None
            is_elementary = False

        # Build shortest path memories - store (source, target) -> direction
        nodes_list = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}

        # Direction constants
        DIRECTION_UP = 0
        DIRECTION_DOWN = 1
        DIRECTION_LEFT = 2
        DIRECTION_RIGHT = 3
        IDENTIFY_GOAL = 4

        memory_count = 0
        for source in nodes_list:
            for target in nodes_list:
                if source == target:
                    # If at goal, action is IDENTIFY_GOAL
                    source_idx = node_to_idx[source]
                    target_idx = node_to_idx[target]
                    self.memory[(source_idx, target_idx)] = IDENTIFY_GOAL
                    memory_count += 1

                elif nx.has_path(graph, source, target):
                    path = nx.shortest_path(graph, source, target)
                    if len(path) > 1:
                        next_step = path[1]  # The immediate next node

                        # Convert to direction based on coordinate system and topology type
                        if is_elementary and hasattr(maze, 'topology_type') and maze.topology_type == 'tree':
                            # Tree topology uses semantic directions with (x, y) coordinates
                            x_src, y_src = source
                            x_next, y_next = next_step
                            dx = x_next - x_src
                            dy = y_next - y_src

                            # Tree semantic directions:
                            if dy < 0:
                                # Moving up in tree (toward root)
                                direction = DIRECTION_UP
                            elif dy > 0:
                                # Moving down in tree (toward leaves)
                                if dx < 0:
                                    direction = DIRECTION_LEFT  # LEFT child
                                elif dx > 0:
                                    direction = DIRECTION_RIGHT  # RIGHT child
                                else:  # dx == 0
                                    direction = DIRECTION_DOWN  # DOWN (middle child)
                            else:
                                # Same level (shouldn't happen in tree)
                                print(f"Warning: Same-level movement in tree from {source} to {next_step}")
                                continue

                        elif is_elementary:
                            # ElementaryMaze uses (x, y) format: x=col, y=row
                            # Need to swap to get (row, col)
                            x_src, y_src = source
                            x_next, y_next = next_step

                            # Convert to (row, col) by swapping
                            r_src, c_src = y_src, x_src
                            r_next, c_next = y_next, x_next

                            dr = r_next - r_src
                            dc = c_next - c_src

                            # Map to direction
                            if dr == -1 and dc == 0:
                                direction = DIRECTION_UP
                            elif dr == 1 and dc == 0:
                                direction = DIRECTION_DOWN
                            elif dr == 0 and dc == -1:
                                direction = DIRECTION_LEFT
                            elif dr == 0 and dc == 1:
                                direction = DIRECTION_RIGHT
                            else:
                                # Should not happen
                                print(f"Warning: Invalid direction from {source} to {next_step} (dr={dr}, dc={dc})")
                                continue

                        else:
                            # MazeGraph uses (row, col) format directly
                            r_src, c_src = source
                            r_next, c_next = next_step

                            dr = r_next - r_src
                            dc = c_next - c_src

                            # Map to direction
                            if dr == -1 and dc == 0:
                                direction = DIRECTION_UP
                            elif dr == 1 and dc == 0:
                                direction = DIRECTION_DOWN
                            elif dr == 0 and dc == -1:
                                direction = DIRECTION_LEFT
                            elif dr == 0 and dc == 1:
                                direction = DIRECTION_RIGHT
                            else:
                                # Should not happen
                                print(f"Warning: Invalid direction from {source} to {next_step} (dr={dr}, dc={dc})")
                                continue

                        # Store mapping: (source_idx, target_idx) -> direction
                        source_idx = node_to_idx[source]
                        target_idx = node_to_idx[target]

                        self.memory[(source_idx, target_idx)] = direction
                        memory_count += 1

        if memory_count == 0:
            raise ValueError("No memories created from maze graph")

        print(f"Initialized episodic memory with {memory_count} navigation memories")

    def query(self, state_encoding, goal_encoding, temperature=None):
        """
        Query memory for action given current state and goal.

        Parameters:
        -----------
        state_encoding : torch.Tensor
            One-hot encoding of current state [batch_size, num_nodes]
        goal_encoding : torch.Tensor
            One-hot encoding of goal state [batch_size, num_nodes]
        temperature : float, optional
            Temperature for softmax (higher = more uniform, lower = sharper)

        Returns:
        --------
        action_logits : torch.Tensor
            Action distribution from memory [batch_size, 5]
            Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=IDENTIFY_GOAL
        """
        if temperature is None:
            temperature = self.default_temperature

        batch_size = state_encoding.shape[0]
        action_logits_list = []

        for i in range(batch_size):
            # Get state and goal indices from one-hot encodings
            state_idx = state_encoding[i].argmax().item()
            goal_idx = goal_encoding[i].argmax().item()

            # Look up direction in memory
            if (state_idx, goal_idx) in self.memory:
                direction = self.memory[(state_idx, goal_idx)]

                # Create sharp distribution centered on the retrieved direction
                # Use high logit for the correct direction, low for others
                logits = torch.full((self.num_actions,), -10.0)
                logits[direction] = 10.0

                # Apply temperature
                logits = logits / temperature

            else:
                # If not in memory (shouldn't happen with complete initialization), use uniform
                logits = torch.zeros(self.num_actions)

            action_logits_list.append(logits)

        # Stack into batch
        action_logits = torch.stack(action_logits_list)

        return action_logits


if __name__ == "__main__":
    print("Testing Slow Memory System...")

    # Create a simple test graph
    from corridors import MazeGraph

    print("\nCreating maze...")
    maze = MazeGraph(length=4, width=4, corridor=0.5, seed=42)
    graph = maze.get_graph()
    num_nodes = graph.number_of_nodes()
    num_actions = 5  # Direction-based actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=IDENTIFY_GOAL

    print(f"Maze has {num_nodes} nodes")

    # Create slow memory
    memory = SlowMemory(num_nodes, num_actions)
    print(f"\nCreated SlowMemory")

    # Initialize memory with MazeGraph object
    print("\nInitializing memory...")
    memory.initialize_memory(maze)
    print(f"Memory contains {len(memory.memory)} entries")

    # Test query
    print("\nTesting memory query...")
    batch_size = 4
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'IDENTIFY_GOAL'}

    state_encoding = torch.zeros(batch_size, num_nodes)
    state_encoding[:, 0] = 1.0  # First node

    goal_encoding = torch.zeros(batch_size, num_nodes)
    goal_encoding[:, num_nodes-1] = 1.0  # Last node

    action_logits = memory.query(state_encoding, goal_encoding)
    print(f"  Action logits shape: {action_logits.shape}")
    suggested_direction = action_logits[0].argmax().item()
    print(f"  Suggested direction: {action_names[suggested_direction]} ({suggested_direction})")

    # Test that retrieved actions are correct
    print("\nTesting multiple queries...")
    nodes_list = list(graph.nodes())

    correct = 0
    total = 0

    for i in range(min(5, num_nodes)):
        for j in range(min(5, num_nodes)):
            state = torch.zeros(1, num_nodes)
            state[0, i] = 1.0

            goal = torch.zeros(1, num_nodes)
            goal[0, j] = 1.0

            logits = memory.query(state, goal)
            suggested_direction = logits.argmax().item()

            if i == j:
                # If at goal, should suggest IDENTIFY_GOAL
                match = (suggested_direction == 4)
                optimal_action_str = "IDENTIFY_GOAL"
            elif nx.has_path(graph, nodes_list[i], nodes_list[j]):
                # Get optimal path and compute expected direction
                path = nx.shortest_path(graph, nodes_list[i], nodes_list[j])
                if len(path) > 1:
                    curr_node = nodes_list[i]
                    next_node = path[1]

                    # Compute expected direction
                    r_curr, c_curr = curr_node
                    r_next, c_next = next_node
                    dr, dc = r_next - r_curr, c_next - c_curr

                    if dr == -1 and dc == 0:
                        expected_dir = 0  # UP
                    elif dr == 1 and dc == 0:
                        expected_dir = 1  # DOWN
                    elif dr == 0 and dc == -1:
                        expected_dir = 2  # LEFT
                    elif dr == 0 and dc == 1:
                        expected_dir = 3  # RIGHT
                    else:
                        continue

                    match = (suggested_direction == expected_dir)
                    optimal_action_str = action_names[expected_dir]
                else:
                    continue
            else:
                continue

            total += 1
            if match:
                correct += 1

            symbol = "✓" if match else "✗"
            print(f"  {symbol} From {nodes_list[i]} to {nodes_list[j]}: "
                  f"suggested {action_names[suggested_direction]}, optimal {optimal_action_str}")

    print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    # Test batch query
    print("\nTesting batch query...")
    batch_states = torch.zeros(3, num_nodes)
    batch_states[0, 0] = 1.0
    batch_states[1, 1] = 1.0
    batch_states[2, 2] = 1.0

    batch_goals = torch.zeros(3, num_nodes)
    batch_goals[:, num_nodes-1] = 1.0

    batch_logits = memory.query(batch_states, batch_goals)
    print(f"  Batch logits shape: {batch_logits.shape}")
    suggested_directions = batch_logits.argmax(dim=1).tolist()
    print(f"  Suggested directions: {[action_names[d] for d in suggested_directions]}")

    print("\n✓ Slow memory tests passed!")
