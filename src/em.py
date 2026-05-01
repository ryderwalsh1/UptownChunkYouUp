import numpy as np
import psyneulink as pnl
import warnings
import networkx as nx
from corridors import MazeGraph
warnings.filterwarnings("ignore", category=UserWarning, module='psyneulink')

# Generate maze graph
MAZE_LENGTH = 8
MAZE_WIDTH = 8
CORRIDOR = 0.5
SEED = 60

print(f"Generating maze graph {MAZE_LENGTH}x{MAZE_WIDTH} with corridor={CORRIDOR}...")
maze = MazeGraph(length=MAZE_LENGTH, width=MAZE_WIDTH, corridor=CORRIDOR, seed=SEED)
G = maze.get_graph()
print(f"Generated maze with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Create node to index mapping
nodes_list = list(G.nodes())
node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}
num_nodes = len(nodes_list)

# Build shortest path memories: for each pair of nodes, store (source, target, next_step)
memories = []
for source in nodes_list:
    for target in nodes_list:
        if source != target and nx.has_path(G, source, target):
            path = nx.shortest_path(G, source, target)
            if len(path) > 1:
                next_step = path[1]  # The immediate next node in the shortest path

                # Create one-hot encodings
                source_encoding = [0] * num_nodes
                source_encoding[node_to_idx[source]] = 1

                target_encoding = [0] * num_nodes
                target_encoding[node_to_idx[target]] = 1

                next_step_encoding = [0] * num_nodes
                next_step_encoding[node_to_idx[next_step]] = 1

                memories.append({
                    'Source': source_encoding,
                    'Target': target_encoding,
                    'Answer': next_step_encoding
                })

memory_capacity = len(memories)
memory_array = np.array([[m['Source'], m['Target'], m['Answer']] for m in memories])

print(f"Generated {len(memories)} navigation memories from maze")
print(f"Each memory has {num_nodes} node encodings")

# Create EMComposition oracle
em_oracle = pnl.EMComposition(
    memory_template=memory_array,
    memory_capacity=memory_capacity,
    field_names=['Source', 'Target', 'Answer'],
    field_weights=[1.0, 1.0, None],
    storage_prob=0.0,  # Don't store new memories during queries
    memory_decay_rate=0.0,
    softmax_gain=15.0,
    normalize_memories=True,
    enable_learning=False,
    name='Maze_Navigator'
)

print(f"\nEM Oracle initialized")
print(f"Total memories stored: {len(em_oracle.memory)}")

# Helper function to convert index back to node
def idx_to_node(idx):
    """Convert encoding index back to node."""
    return nodes_list[idx]


# Query function
def query_next_step(source_node, target_node, debug=False):
    """
    Query for the next step to take from source toward target.

    Args:
        source_node: Source node (e.g., (0, 0))
        target_node: Target/goal node (e.g., (7, 7))
        debug: Whether to print debug info

    Returns:
        Retrieved next step (as node) and full results
    """
    # Create one-hot encodings
    source_encoding = [0] * num_nodes
    source_encoding[node_to_idx[source_node]] = 1

    target_encoding = [0] * num_nodes
    target_encoding[node_to_idx[target_node]] = 1

    # Query the oracle
    query_inputs = {
        em_oracle.query_input_nodes[0]: source_encoding,
        em_oracle.query_input_nodes[1]: target_encoding
    }

    if debug:
        print(f"\nDEBUG query_next_step:")
        print(f"  Source node: {source_node}, Target node: {target_node}")

    results = em_oracle.run(inputs=query_inputs)

    # Extract the next step from Answer field
    next_step_encoding = results[2]  # Answer field
    next_step_idx = next_step_encoding.argmax()
    next_step_node = idx_to_node(next_step_idx)

    if debug:
        print(f"  Retrieved next step: {next_step_node}")
        print(f"  Answer distribution: {next_step_encoding}")

    return next_step_node, results

print("\n" + "=" * 70)
print("TESTING EM ORACLE - NEXT STEP RETRIEVAL")
print("=" * 70)

# Test all stored (source, target) -> next_step memories
print(f"\nTesting all {len(memories)} stored navigation memories...")
errors = []

for memory in memories:
    # Decode the source, target, and expected next step from one-hot encodings
    source_idx = memory['Source'].index(1)
    target_idx = memory['Target'].index(1)
    expected_next_step_idx = memory['Answer'].index(1)

    source_node = idx_to_node(source_idx)
    target_node = idx_to_node(target_idx)
    expected_next_step = idx_to_node(expected_next_step_idx)

    # Query the oracle
    retrieved_next_step, results = query_next_step(source_node, target_node)

    if retrieved_next_step != expected_next_step:
        errors.append({
            'source': source_node,
            'target': target_node,
            'expected_next': expected_next_step,
            'retrieved_next': retrieved_next_step,
            'distribution': results[2]
        })

print(f"\nResults:")
print(f"  Total memories tested: {len(memories)}")
print(f"  Correct retrievals: {len(memories) - len(errors)}")
print(f"  Incorrect retrievals: {len(errors)}")
print(f"  Accuracy: {100 * (len(memories) - len(errors)) / len(memories):.2f}%")

if errors:
    print(f"\nErrors found (showing first 10):")
    for i, err in enumerate(errors[:10]):
        print(f"  {i+1}. Source={err['source']}, Target={err['target']}")
        print(f"     Expected next step: {err['expected_next']}")
        print(f"     Retrieved next step: {err['retrieved_next']}")
        print(f"     Distribution: {err['distribution']}")
else:
    print("\n✓ All next steps retrieved correctly!")

print("\n" + "=" * 70)
print("STEP-BY-STEP NAVIGATION EXAMPLES")
print("=" * 70)

# Create test examples: pick node pairs with multi-hop paths
test_examples = []
for source in nodes_list[:10]:  # Limit to first 10 nodes for testing
    for target in nodes_list[:10]:
        if source != target and nx.has_path(G, source, target):
            path = nx.shortest_path(G, source, target)
            if len(path) > 2:  # Multi-hop paths only
                test_examples.append((source, target, path))
                if len(test_examples) >= 5:
                    break
    if len(test_examples) >= 5:
        break

print(f"\nTesting step-by-step navigation using EM for multi-hop paths:")

for i, (source, target, path) in enumerate(test_examples):
    print(f"\n{'='*60}")
    print(f"Example {i+1}: Navigate from {source} to {target}")
    print(f"Expected path: {' -> '.join(map(str, path))}")
    print(f"Path length: {len(path) - 1} steps")

    # Simulate step-by-step navigation
    current = source
    traversed_path = [current]
    success = True

    for step_num in range(len(path) - 1):  # Don't count the start node
        next_step, _ = query_next_step(current, target)
        traversed_path.append(next_step)

        expected_next = path[step_num + 1]
        match = "✓" if next_step == expected_next else "✗"
        print(f"  Step {step_num + 1}: {current} -> {next_step} (expected: {expected_next}) {match}")

        if next_step != expected_next:
            success = False

        current = next_step

        # Stop if we've reached the target
        if current == target:
            break

    traversed_str = " -> ".join(map(str, traversed_path))
    if success and current == target:
        print(f"  ✓ Successfully navigated: {traversed_str}")
    else:
        print(f"  ✗ Navigation failed: {traversed_str}")

print("\n" + "=" * 70)
print("DIRECT EDGE EXAMPLES")
print("=" * 70)

# Test direct edges (single-hop)
edges = list(G.edges())
print(f"\nTesting direct edges (answer should equal target):")

for i, (source, target) in enumerate(edges[:5]):
    next_step, _ = query_next_step(source, target)
    match = "✓" if next_step == target else "✗"
    print(f"{match} {source} -> {target}: retrieved next step = {next_step}")
