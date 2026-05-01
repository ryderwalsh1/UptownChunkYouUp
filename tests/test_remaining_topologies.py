"""
Test script to verify SlowMemory works correctly with corridor chain and tree topologies.
"""

import torch
import networkx as nx
from slow import SlowMemory
from lambda_experiment.topology_generators import (
    make_corridor_chain, make_tree
)


def test_corridor_chain():
    """Test corridor chain topology."""
    print("\n" + "="*80)
    print("Testing corridor chain topology...")
    print("="*80)

    maze = make_corridor_chain(num_junctions=2, corridor_length=3, branching_factor=2, seed=42)
    graph = maze.get_graph()
    num_nodes = graph.number_of_nodes()

    print(f"Maze has {num_nodes} nodes")
    print(f"Start-goal pairs: {maze.start_goal_pairs}")

    # Create and initialize memory
    memory = SlowMemory(num_nodes, num_actions=5)
    memory.initialize_memory(maze)

    # Test path from start to goal
    start, goal = maze.start_goal_pairs[0]
    print(f"\nTesting path from {start} to {goal}")

    # Get shortest path
    shortest_path = nx.shortest_path(graph, start, goal)
    print(f"Shortest path: {shortest_path}")
    print(f"Path length: {len(shortest_path) - 1} steps")

    # Get node indices
    nodes_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}

    # Follow memory recommendations
    current = start
    memory_path = [current]
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'IDENTIFY_GOAL'}

    for step in range(len(shortest_path) + 5):  # Extra steps for safety
        # Create state encoding
        state_encoding = torch.zeros(1, num_nodes)
        state_encoding[0, node_to_idx[current]] = 1.0

        goal_encoding = torch.zeros(1, num_nodes)
        goal_encoding[0, node_to_idx[goal]] = 1.0

        # Query memory
        action_logits = memory.query(state_encoding, goal_encoding)
        action = action_logits.argmax().item()

        print(f"  Step {step}: At {current}, memory suggests {action_names[action]} ({action})")

        if current == goal:
            if action == 4:  # IDENTIFY_GOAL
                print(f"  ✓ Correctly reached goal!")
                break
            else:
                print(f"  ✗ At goal but memory didn't suggest IDENTIFY_GOAL")
                return False

        # Get next node from maze
        direction_neighbors = maze.get_direction_neighbors(current)

        if action in direction_neighbors:
            current = direction_neighbors[action]
            memory_path.append(current)
        else:
            print(f"  ✗ Invalid action: {action_names[action]} not available from {current}")
            print(f"    Available directions: {list(direction_neighbors.keys())}")
            print(f"    Available neighbors: {[direction_neighbors[d] for d in direction_neighbors]}")
            return False

    print(f"\nMemory path: {memory_path}")
    print(f"Optimal path: {shortest_path}")

    if memory_path == shortest_path:
        print("✓ Memory path matches optimal path!")
        return True
    else:
        print("✗ Memory path does NOT match optimal path")
        return False


def test_tree():
    """Test tree topology."""
    print("\n" + "="*80)
    print("Testing tree topology...")
    print("="*80)

    maze = make_tree(depth=3, branching_factor=2, seed=42)
    graph = maze.get_graph()
    num_nodes = graph.number_of_nodes()

    print(f"Maze has {num_nodes} nodes")
    print(f"Number of start-goal pairs: {len(maze.start_goal_pairs)}")
    print(f"Testing first pair: {maze.start_goal_pairs[0]}")

    # Create and initialize memory
    memory = SlowMemory(num_nodes, num_actions=5)
    memory.initialize_memory(maze)

    # Test first start-goal pair (root to leftmost leaf)
    start, goal = maze.start_goal_pairs[0]
    print(f"\nTesting path from {start} to {goal}")

    # Get shortest path
    shortest_path = nx.shortest_path(graph, start, goal)
    print(f"Shortest path: {shortest_path}")
    print(f"Path length: {len(shortest_path) - 1} steps")

    # Get node indices
    nodes_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}

    # Follow memory recommendations
    current = start
    memory_path = [current]
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'IDENTIFY_GOAL'}

    for step in range(len(shortest_path) + 5):  # Extra steps for safety
        # Create state encoding
        state_encoding = torch.zeros(1, num_nodes)
        state_encoding[0, node_to_idx[current]] = 1.0

        goal_encoding = torch.zeros(1, num_nodes)
        goal_encoding[0, node_to_idx[goal]] = 1.0

        # Query memory
        action_logits = memory.query(state_encoding, goal_encoding)
        action = action_logits.argmax().item()

        print(f"  Step {step}: At {current}, memory suggests {action_names[action]} ({action})")

        if current == goal:
            if action == 4:  # IDENTIFY_GOAL
                print(f"  ✓ Correctly reached goal!")
                break
            else:
                print(f"  ✗ At goal but memory didn't suggest IDENTIFY_GOAL")
                return False

        # Get next node from maze
        direction_neighbors = maze.get_direction_neighbors(current)

        if action in direction_neighbors:
            current = direction_neighbors[action]
            memory_path.append(current)
        else:
            print(f"  ✗ Invalid action: {action_names[action]} not available from {current}")
            print(f"    Available directions: {list(direction_neighbors.keys())}")
            print(f"    Available neighbors: {[direction_neighbors[d] for d in direction_neighbors]}")
            return False

    print(f"\nMemory path: {memory_path}")
    print(f"Optimal path: {shortest_path}")

    if memory_path == shortest_path:
        print("✓ Memory path matches optimal path!")
        return True
    else:
        print("✗ Memory path does NOT match optimal path")
        return False


if __name__ == "__main__":
    print("Testing SlowMemory with Corridor Chain and Tree Topologies")
    print("=" * 80)

    results = []

    # Test corridor chain
    results.append(("Corridor Chain", test_corridor_chain()))

    # Test tree
    results.append(("Tree", test_tree()))

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
