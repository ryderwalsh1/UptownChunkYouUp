import numpy as np
import psyneulink as pnl
import warnings
from gen_impgraph import generate_implication_graph, get_implications, enumerate_examples, implications_to_memories
warnings.filterwarnings("ignore", category=UserWarning, module='psyneulink')

# Generate implication graph
NUM_VARS = 5
NUM_CLAUSES = 10

print(f"Generating implication graph with {NUM_VARS} variables and {NUM_CLAUSES} clauses...")
G = generate_implication_graph(NUM_VARS, NUM_CLAUSES)
edges = get_implications(G)
print(f"Generated {len(edges)} implications")

# Convert implication graph to memory format


memories, memory_array, literal_to_idx = implications_to_memories(G, NUM_VARS)
num_literals = 2 * NUM_VARS
memory_capacity = len(memories)

print(f"Generated {len(memories)} memories from implication graph")
print(f"Each memory has {num_literals} literal encodings")

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
    name='Implication_Oracle'
)

print(f"\nEM Oracle initialized")
print(f"Total memories stored: {len(em_oracle.memory)}")

# Helper function to convert index back to literal
def idx_to_literal(idx, num_vars):
    """Convert encoding index back to literal value."""
    if idx < num_vars:
        return idx + 1
    else:
        return -(idx - num_vars + 1)


# Query function
def query_next_step(source_lit, target_lit, num_literals, debug=False):
    """
    Query for the next step to take from source toward target.

    Args:
        source_lit: Source literal (e.g., 1, -1, 5, -5)
        target_lit: Target/goal literal
        num_literals: Total number of literals (2 * num_vars)
        debug: Whether to print debug info

    Returns:
        Retrieved next step (as literal value) and full results
    """
    # Create one-hot encodings
    source_encoding = [0] * num_literals
    source_encoding[literal_to_idx(source_lit)] = 1

    target_encoding = [0] * num_literals
    target_encoding[literal_to_idx(target_lit)] = 1

    # Query the oracle
    query_inputs = {
        em_oracle.query_input_nodes[0]: source_encoding,
        em_oracle.query_input_nodes[1]: target_encoding
    }

    if debug:
        print(f"\nDEBUG query_next_step:")
        print(f"  Source literal: {source_lit}, Target literal: {target_lit}")

    results = em_oracle.run(inputs=query_inputs)

    # Extract the next step from Answer field
    next_step_encoding = results[2]  # Answer field
    next_step_idx = next_step_encoding.argmax()
    next_step_lit = idx_to_literal(next_step_idx, NUM_VARS)

    if debug:
        print(f"  Retrieved next step: {next_step_lit}")
        print(f"  Answer distribution: {next_step_encoding}")

    return next_step_lit, results

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

    source_lit = idx_to_literal(source_idx, NUM_VARS)
    target_lit = idx_to_literal(target_idx, NUM_VARS)
    expected_next_step = idx_to_literal(expected_next_step_idx, NUM_VARS)

    # Query the oracle
    retrieved_next_step, results = query_next_step(source_lit, target_lit, num_literals)

    if retrieved_next_step != expected_next_step:
        errors.append({
            'source': source_lit,
            'target': target_lit,
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

# Test multi-hop path navigation by repeatedly querying EM
dataset = enumerate_examples(G, NUM_VARS)
print(f"\nGenerated {len(dataset)} valid queries from graph traversal")
print("\nTesting step-by-step navigation using EM for multi-hop paths:")

# Show some example multi-hop paths
test_examples = [ex for ex in dataset if len(ex[3]) > 2][:5]  # Get examples with path length > 2

for i, (source, target, expected_answer, path) in enumerate(test_examples):
    print(f"\n{'='*60}")
    print(f"Example {i+1}: Navigate from {source} to {target}")
    print(f"Expected path: {' -> '.join(map(str, path))}")
    print(f"Path length: {len(path) - 1} steps")

    # Simulate step-by-step navigation
    current = source
    traversed_path = [current]
    success = True

    for step_num in range(len(path) - 1):  # Don't count the start node
        next_step, _ = query_next_step(current, target, num_literals)
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
    next_step, _ = query_next_step(source, target, num_literals)
    match = "✓" if next_step == target else "✗"
    print(f"{match} {source} -> {target}: retrieved next step = {next_step}")
