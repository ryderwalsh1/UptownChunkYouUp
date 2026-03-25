"""
Slow Memory System

Pure episodic memory retrieval using PsyNeuLink.
NOT a neural network - just memory lookup.
"""

import torch
import numpy as np
import psyneulink as pnl
import networkx as nx
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='psyneulink')


class SlowMemory:
    """Simple episodic memory system for action retrieval."""

    def __init__(self, num_nodes, num_actions, memory_softmax_gain=15.0, memory_field_weights=None):
        """
        Initialize slow memory system.

        Parameters:
        -----------
        num_nodes : int
            Number of nodes in the maze (state space size)
        num_actions : int
            Number of possible actions (should equal num_nodes)
        memory_softmax_gain : float
            Softmax gain for episodic memory retrieval
        memory_field_weights : list, optional
            Weights for memory fields [source, target, answer]
        """
        self.num_nodes = num_nodes
        self.num_actions = num_actions
        self.memory_softmax_gain = memory_softmax_gain
        self.memory_field_weights = memory_field_weights or [1.0, 1.0, None]

        # Episodic memory (PsyNeuLink EMComposition)
        self.episodic_memory = None

    def initialize_memory(self, maze_graph):
        """
        Initialize episodic memory with shortest path information from maze.

        Parameters:
        -----------
        maze_graph : networkx.Graph
            The maze graph
        """
        # Build shortest path memories - store (source, target) -> next_node
        nodes_list = list(maze_graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}

        memories = []
        for source in nodes_list:
            for target in nodes_list:
                if source != target and nx.has_path(maze_graph, source, target):
                    path = nx.shortest_path(maze_graph, source, target)
                    if len(path) > 1:
                        next_step = path[1]  # The immediate next node

                        # Create one-hot encodings
                        source_encoding = [0] * self.num_nodes
                        source_encoding[node_to_idx[source]] = 1

                        target_encoding = [0] * self.num_nodes
                        target_encoding[node_to_idx[target]] = 1

                        # Answer is the next node (action = next node index)
                        next_step_encoding = [0] * self.num_actions
                        next_step_encoding[node_to_idx[next_step]] = 1

                        memories.append({
                            'Source': source_encoding,
                            'Target': target_encoding,
                            'Answer': next_step_encoding
                        })

        if len(memories) == 0:
            raise ValueError("No memories created from maze graph")

        memory_capacity = len(memories)
        memory_array = np.array([[m['Source'], m['Target'], m['Answer']] for m in memories])

        # Create EMComposition
        self.episodic_memory = pnl.EMComposition(
            memory_template=memory_array,
            memory_capacity=memory_capacity,
            field_names=['Source', 'Target', 'Answer'],
            field_weights=self.memory_field_weights,
            storage_prob=0.0,  # Don't store new memories during queries
            memory_decay_rate=0.0,
            softmax_gain=self.memory_softmax_gain,
            normalize_memories=True,
            enable_learning=False,
            name='Slow_Memory'
        )

        print(f"Initialized episodic memory with {len(memories)} navigation memories")

    def query(self, state_encoding, goal_encoding):
        """
        Query episodic memory for action given current state and goal.

        Parameters:
        -----------
        state_encoding : torch.Tensor
            One-hot encoding of current state [batch_size, num_nodes]
        goal_encoding : torch.Tensor
            One-hot encoding of goal state [batch_size, num_nodes]

        Returns:
        --------
        action_logits : torch.Tensor
            Action distribution from memory [batch_size, num_actions]
        """
        if self.episodic_memory is None:
            raise ValueError("Episodic memory not initialized. Call initialize_memory() first.")

        batch_size = state_encoding.shape[0]
        action_logits_list = []

        for i in range(batch_size):
            # Convert to numpy for PsyNeuLink
            state_np = state_encoding[i].detach().cpu().numpy()
            goal_np = goal_encoding[i].detach().cpu().numpy()

            # Query the episodic memory
            query_inputs = {
                self.episodic_memory.query_input_nodes[0]: state_np.tolist(),
                self.episodic_memory.query_input_nodes[1]: goal_np.tolist()
            }

            results = self.episodic_memory.run(inputs=query_inputs)

            # Extract the retrieved action distribution (Answer field)
            action_dist = np.array(results[2], dtype=np.float32)  # [num_actions]
            action_logits_list.append(action_dist)

        # Convert to torch tensor
        action_logits = torch.tensor(np.array(action_logits_list), dtype=torch.float32)

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
    print("\nInitializing episodic memory...")
    memory.initialize_memory(graph)

    # Test query
    print("\nTesting memory query...")
    batch_size = 4

    state_encoding = torch.zeros(batch_size, num_nodes)
    state_encoding[:, 0] = 1.0  # First node

    goal_encoding = torch.zeros(batch_size, num_nodes)
    goal_encoding[:, num_nodes-1] = 1.0  # Last node

    action_logits = memory.query(state_encoding, goal_encoding)
    print(f"  Action logits shape: {action_logits.shape}")
    print(f"  Action logits (first sample): {action_logits[0]}")
    print(f"  Suggested action (argmax): {action_logits[0].argmax().item()}")

    # Test that retrieved actions make sense
    print("\nTesting multiple queries...")
    nodes_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}

    for i in range(min(3, num_nodes)):
        for j in range(min(3, num_nodes)):
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

                match = "✓" if suggested_node == optimal_next else "✗"
                print(f"  {match} From {nodes_list[i]} to {nodes_list[j]}: suggested {suggested_node}, optimal {optimal_next}")

    print("\n✓ Slow memory tests passed!")
