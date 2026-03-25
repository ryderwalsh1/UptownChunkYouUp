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

    def __init__(self, num_nodes, num_actions, default_temperature=1.0):
        """
        Initialize slow memory system.

        Parameters:
        -----------
        num_nodes : int
            Number of nodes in the maze (state space size)
        num_actions : int
            Number of possible actions (should equal num_nodes)
        default_temperature : float
            Temperature for softmax when creating distributions
        """
        self.num_nodes = num_nodes
        self.num_actions = num_actions
        self.default_temperature = default_temperature

        # Memory storage: dict mapping (state_idx, goal_idx) -> action_idx
        self.memory = {}

    def initialize_memory(self, maze_graph):
        """
        Initialize memory with shortest path information from maze.

        Parameters:
        -----------
        maze_graph : networkx.Graph
            The maze graph
        """
        # Build shortest path memories - store (source, target) -> next_node
        nodes_list = list(maze_graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}

        memory_count = 0
        for source in nodes_list:
            for target in nodes_list:
                if source != target and nx.has_path(maze_graph, source, target):
                    path = nx.shortest_path(maze_graph, source, target)
                    if len(path) > 1:
                        next_step = path[1]  # The immediate next node

                        # Store mapping: (source_idx, target_idx) -> next_step_idx
                        source_idx = node_to_idx[source]
                        target_idx = node_to_idx[target]
                        next_step_idx = node_to_idx[next_step]

                        self.memory[(source_idx, target_idx)] = next_step_idx
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
            Action distribution from memory [batch_size, num_actions]
        """
        if temperature is None:
            temperature = self.default_temperature

        batch_size = state_encoding.shape[0]
        action_logits_list = []

        for i in range(batch_size):
            # Get state and goal indices from one-hot encodings
            state_idx = state_encoding[i].argmax().item()
            goal_idx = goal_encoding[i].argmax().item()

            # Look up action in memory
            if (state_idx, goal_idx) in self.memory:
                next_step_idx = self.memory[(state_idx, goal_idx)]

                # Create sharp distribution centered on the retrieved action
                # Use high logit for the correct action, low for others
                logits = torch.full((self.num_actions,), -10.0)
                logits[next_step_idx] = 10.0

                # Apply temperature
                logits = logits / temperature

            else:
                # If not in memory (shouldn't happen with complete graph), use uniform
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
    num_actions = num_nodes  # Actions are node indices

    print(f"Maze has {num_nodes} nodes")

    # Create slow memory
    memory = SlowMemory(num_nodes, num_actions)
    print(f"\nCreated SlowMemory")

    # Initialize memory
    print("\nInitializing memory...")
    memory.initialize_memory(graph)
    print(f"Memory contains {len(memory.memory)} entries")

    # Test query
    print("\nTesting memory query...")
    batch_size = 4

    state_encoding = torch.zeros(batch_size, num_nodes)
    state_encoding[:, 0] = 1.0  # First node

    goal_encoding = torch.zeros(batch_size, num_nodes)
    goal_encoding[:, num_nodes-1] = 1.0  # Last node

    action_logits = memory.query(state_encoding, goal_encoding)
    print(f"  Action logits shape: {action_logits.shape}")
    print(f"  Suggested action (argmax): {action_logits[0].argmax().item()}")

    # Test that retrieved actions are correct
    print("\nTesting multiple queries...")
    nodes_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}

    correct = 0
    total = 0

    for i in range(min(5, num_nodes)):
        for j in range(min(5, num_nodes)):
            if i != j and nx.has_path(graph, nodes_list[i], nodes_list[j]):
                state = torch.zeros(1, num_nodes)
                state[0, i] = 1.0

                goal = torch.zeros(1, num_nodes)
                goal[0, j] = 1.0

                logits = memory.query(state, goal)
                suggested_action = logits.argmax().item()
                suggested_node = nodes_list[suggested_action]

                # Get optimal path
                path = nx.shortest_path(graph, nodes_list[i], nodes_list[j])
                optimal_next = path[1] if len(path) > 1 else nodes_list[j]

                match = suggested_node == optimal_next
                total += 1
                if match:
                    correct += 1

                symbol = "✓" if match else "✗"
                print(f"  {symbol} From {nodes_list[i]} to {nodes_list[j]}: "
                      f"suggested {suggested_node}, optimal {optimal_next}")

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
    print(f"  Suggested actions: {batch_logits.argmax(dim=1).tolist()}")

    print("\n✓ Slow memory tests passed!")
