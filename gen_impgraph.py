import random
import statistics
import matplotlib.pyplot as plt
import networkx as nx


def generate_implication_graph(num_vars, num_clauses):
  """Generates a skew-symmetric implication graph using NetworkX.

  Nodes are literals: 1 to num_vars and -1 to -num_vars.
  """
  G = nx.DiGraph()
  literals = [i for i in range(1, num_vars + 1)] + [
      -i for i in range(1, num_vars + 1)
  ]

  # Ensure all literals exist as nodes, even if disconnected
  G.add_nodes_from(literals)

  clauses_added = 0
  while clauses_added < num_clauses:
    u = random.choice(literals)
    v = random.choice(literals)

    # Avoid trivial edges (u -> u), immediate contradictions (u -> -u), and check if edge already exists
    if u == v or u == -v or G.has_edge(u, v):
      continue

    # Add the implication u -> v
    G.add_edge(u, v)

    # Add the required contrapositive: ~v -> ~u
    G.add_edge(-v, -u)

    clauses_added += 1

  return G


def get_implications(G):
  """Returns the list of all implications (edges) in the graph."""
  # NetworkX edges() returns a view, converting to list for standard usage
  return list(G.edges())


def enumerate_examples(G, num_vars):
  """Enumerates all possible examples for the graph.

  Returns: list of tuples (s, t, Answer, Path)
  """
  literals = [i for i in range(1, num_vars + 1)] + [
      -i for i in range(1, num_vars + 1)
  ]
  examples = []

  # NetworkX has a highly optimized all-pairs shortest path function.
  # It returns a dictionary of dictionaries: paths[source][target] = [path_list]
  paths = dict(nx.all_pairs_shortest_path(G))

  for s in literals:
    for t in literals:
      # Skip self-queries and queries about a literal's own negation
      if s == t or s == -t:
        continue

      path_to_t = paths.get(s, {}).get(t)
      path_to_neg_t = paths.get(s, {}).get(-t)

      # Prioritize the positive implication
      if path_to_t is not None:
        examples.append((s, t, True, path_to_t))
      elif path_to_neg_t is not None:
        examples.append((s, t, False, path_to_neg_t))

  return examples


def visualize_graph(G):
  """Utility function to plot the implication graph.

  Red nodes are negative literals, green nodes are positive.
  Automatically adjusts visualization parameters based on graph size.
  """
  num_nodes = G.number_of_nodes()

  # Dynamically scale figure size based on number of nodes
  scale_factor = max(1, num_nodes / 20)
  fig_size = (12 * scale_factor, 10 * scale_factor)
  plt.figure(figsize=fig_size)

  # Use spring layout with stronger spacing for large graphs
  # k controls the optimal distance between nodes (higher = more spread out)
  # iterations helps the layout converge to a better solution
  k_value = 1.0 / (num_nodes ** 0.5) if num_nodes > 10 else None
  pos = nx.spring_layout(G, seed=42, k=k_value, iterations=50)

  # Scale node and font sizes inversely with graph size
  node_size = max(200, 700 - (num_nodes * 5))
  font_size = max(6, 12 - (num_nodes // 10))
  arrow_size = max(8, 15 - (num_nodes // 15))

  # Color coding: positive literals (green), negative literals (red)
  node_colors = ["#8fce00" if node > 0 else "#ff4c4c" for node in G.nodes()]

  nx.draw(
      G,
      pos,
      with_labels=True,
      node_color=node_colors,
      node_size=node_size,
      font_size=font_size,
      font_weight="bold",
      arrowsize=arrow_size,
      edge_color="gray",
      alpha=0.9,  # Slight transparency for edges
      width=0.8,  # Thinner edges for less clutter
  )

  plt.title("Skew-Symmetric Implication Graph", fontsize=16)
  plt.tight_layout()  # Automatically adjust subplot params for better fit

def compute_next_steps(G, num_vars):
    """
    Computes the optimal next step for navigating from any literal to any target literal.

    Uses BFS to find shortest paths, similar to gridworld.compute_optimal_actions().

    Args:
        G: NetworkX DiGraph of implications
        num_vars: Number of variables

    Returns:
        dict: Mapping (source_literal, target_literal) -> next_literal on shortest path
              Returns None if target is unreachable from source
    """
    import networkx as nx
    from collections import deque

    literals = [i for i in range(1, num_vars + 1)] + [-i for i in range(1, num_vars + 1)]
    next_steps = {}

    # For each source literal, run BFS to find shortest paths to all targets
    for source in literals:
        # BFS from source
        queue = deque([source])
        visited = {source}
        parent = {source: None}

        while queue:
            current = queue.popleft()

            # Get neighbors from the graph
            if current in G:
                for neighbor in G.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = current
                        queue.append(neighbor)

        # For each reachable target, determine the first step
        for target in literals:
            if source == target:
                # Already at target, no step needed (could skip or use special encoding)
                next_steps[(source, target)] = None
            elif target not in parent:
                # Target unreachable from source
                next_steps[(source, target)] = None
            else:
                # Reconstruct path from target back to source
                path = []
                current = target
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()

                # The first step is path[1] (path[0] is source)
                if len(path) >= 2:
                    next_steps[(source, target)] = path[1]
                else:
                    next_steps[(source, target)] = None

    return next_steps


def implications_to_memories(G, num_vars):
    """
    Converts an implication graph to memory format compatible with EMComposition.

    For each (source, target) pair, stores the optimal next step to take.
    Similar to gridworld.action_matrix_to_memories().

    Args:
        G: NetworkX DiGraph of implications
        num_vars: Number of variables (literals range from 1 to num_vars and their negations)

    Returns:
        tuple: (memories, memory_array, literal_to_idx) where:
            - memories: List of dicts with 'Source', 'Target', 'Answer' keys
            - memory_array: 3D list format [num_entries, num_fields, field_length]
            - literal_to_idx: Function to convert literal to encoding index
    """
    # Total number of literals: positive (1 to num_vars) + negative (-num_vars to -1)
    num_literals = 2 * num_vars

    # Mapping from literal value to encoding index
    # Positive literals: 1 to num_vars -> indices 0 to (num_vars-1)
    # Negative literals: -1 to -num_vars -> indices num_vars to (2*num_vars-1)
    def literal_to_idx(lit):
        if lit > 0:
            return lit - 1
        else:
            return num_vars + abs(lit) - 1

    # Compute optimal next steps for all (source, target) pairs
    next_steps = compute_next_steps(G, num_vars)

    memories = []
    literals = [i for i in range(1, num_vars + 1)] + [-i for i in range(1, num_vars + 1)]

    for source in literals:
        for target in literals:
            # Skip if same literal or unreachable
            next_step = next_steps.get((source, target))
            if next_step is None:
                continue

            # One-hot encode source literal
            source_encoding = [0] * num_literals
            source_encoding[literal_to_idx(source)] = 1

            # One-hot encode target literal
            target_encoding = [0] * num_literals
            target_encoding[literal_to_idx(target)] = 1

            # Answer is the next step to take (one-hot encoded)
            answer_encoding = [0] * num_literals
            answer_encoding[literal_to_idx(next_step)] = 1

            memories.append({
                'Source': source_encoding,
                'Target': target_encoding,
                'Answer': answer_encoding
            })

    # Convert to 3D array format for pre-loading
    memory_array = []
    for memory in memories:
        entry = [
            memory['Source'],
            memory['Target'],
            memory['Answer']
        ]
        memory_array.append(entry)

    return memories, memory_array, literal_to_idx

def main():
  NUM_VARS = 8
  NUM_CLAUSES = 10

  G = generate_implication_graph(NUM_VARS, NUM_CLAUSES)

  edges = get_implications(G)
  print(f"Total implications: {len(edges)}")

  print("\nSample Implications (Edges):")
  for i, (u, v) in enumerate(edges[:5]):
    print(f"  {u} -> {v}")
  print("...\n")

  dataset = enumerate_examples(G, NUM_VARS)
  print(f"Generated {len(dataset)} valid queries.\n")

  # Calculate the number of edges for each path
  path_lengths = [len(path) - 1 for _, _, _, path in dataset]

  min_len = min(path_lengths)
  max_len = max(path_lengths)
  avg_len = statistics.mean(path_lengths)
  median_len = statistics.median(path_lengths)

  print("--- Path Length Statistics ---")
  print(f"Minimum length: {min_len}")
  print(f"Maximum length: {max_len}")
  print(f"Average length: {avg_len:.2f}")
  print(f"Median length:  {median_len}")
  print("------------------------------\n")

  for i, (s, t, ans, path) in enumerate(dataset[:5]):
    path_str = " -> ".join(map(str, path))
    print(f"Query: {s} |- {t} ? | Answer: {ans} | Path: {path_str}")

  # Moved outside the loop so it only visualizes once
  visualize_graph(G)

if __name__ == "__main__":
  main()