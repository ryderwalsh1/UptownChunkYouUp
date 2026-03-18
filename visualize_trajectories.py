import numpy as np
import random
import pickle
import gen_impgraph
import networkx as nx
from policy_network_pc import PolicyNetworkPC
import os


# ============================================================================
# Utility Functions (from lambda_learner.py)
# ============================================================================

def load_graph(graph_path):
    """Load a saved implication graph from a pickle file."""
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def load_weight_matrices(weights_dir, network_type='policy', epoch=None):
    """Load pre-trained weight matrices from disk."""
    if weights_dir is None:
        return None

    # Try to load from subdirectories first (mlp_basic_learner.py format)
    source_to_hidden_path = os.path.join(weights_dir, 'source_to_hidden')
    target_to_hidden_path = os.path.join(weights_dir, 'target_to_hidden')
    hidden_to_output_path = os.path.join(weights_dir, 'hidden_to_output')

    if os.path.isdir(source_to_hidden_path):
        # Load from subdirectories
        if epoch is not None:
            # Load specific epoch
            source_file = f'epoch_{epoch}.npy'
            target_file = f'epoch_{epoch}.npy'
            output_file = f'epoch_{epoch}.npy'

            source_path = os.path.join(source_to_hidden_path, source_file)
            target_path = os.path.join(target_to_hidden_path, target_file)
            output_path = os.path.join(hidden_to_output_path, output_file)

            if not os.path.exists(source_path):
                raise ValueError(f"Epoch {epoch} not found in {weights_dir}")

            source_matrix = np.load(source_path)
            target_matrix = np.load(target_path)
            output_matrix = np.load(output_path)
            print(f"Loaded weights from epoch {epoch}")
        else:
            # Load latest epoch
            source_files = sorted([f for f in os.listdir(source_to_hidden_path) if f.endswith('.npy')])
            target_files = sorted([f for f in os.listdir(target_to_hidden_path) if f.endswith('.npy')])
            output_files = sorted([f for f in os.listdir(hidden_to_output_path) if f.endswith('.npy')])

            if not source_files or not target_files or not output_files:
                raise ValueError(f"No weight files found in {weights_dir}")

            # Use the last (most recent) epoch
            source_matrix = np.load(os.path.join(source_to_hidden_path, source_files[-1]))
            target_matrix = np.load(os.path.join(target_to_hidden_path, target_files[-1]))
            output_matrix = np.load(os.path.join(hidden_to_output_path, output_files[-1]))

            # Extract epoch number from filename
            latest_epoch = source_files[-1].replace('epoch_', '').replace('.npy', '')
            print(f"Loaded weights from latest epoch ({latest_epoch})")
    else:
        # Load from flat directory (td_lambda_learner.py format)
        if epoch is not None:
            print(f"Warning: epoch parameter ignored for flat directory format")

        source_matrix = np.load(os.path.join(weights_dir, f'{network_type}_source_to_hidden.npy'))
        target_matrix = np.load(os.path.join(weights_dir, f'{network_type}_target_to_hidden.npy'))
        output_matrix = np.load(os.path.join(weights_dir, f'{network_type}_hidden_to_output.npy'))
        print(f"Loaded final weights")

    return (source_matrix, target_matrix, output_matrix)


def literal_to_token_idx(literal, num_vars):
    """Map literal to token index."""
    if literal > 0:
        return literal - 1
    else:
        return num_vars + abs(literal) - 1


def token_idx_to_literal(idx, num_vars):
    """Map token index to literal."""
    if idx < num_vars:
        return idx + 1
    else:
        return -(idx - num_vars + 1)


def get_adjacent_nodes(current_literal, graph):
    """Get valid next nodes (neighbors) from the current literal in the graph."""
    return list(graph.successors(current_literal))


def generate_trajectory_from_graph(source_literal, target_literal, graph):
    """Generate optimal trajectory from source to target using shortest path."""
    try:
        path = nx.shortest_path(graph, source=source_literal, target=target_literal)
        return path
    except nx.NetworkXNoPath:
        return None


def compute_trajectory_chunkability(entropies):
    """Compute the chunkability (corridor-likeness) of a trajectory."""
    if len(entropies) == 0:
        return 0.0
    effective_actions = [np.exp(h) for h in entropies]
    chunkability_values = [1.0 / ea for ea in effective_actions]
    return np.mean(chunkability_values)


# ============================================================================
# Episode Execution with Detailed Logging
# ============================================================================

def run_episode_verbose(policy_net, source_literal, target_literal, graph, num_vars,
                       max_steps=100, entropy_threshold=1.5, oracle_sensitivity=5.0,
                       gamma=0.99, lambda_=0.0, lambda_exponent=2.5):
    """
    Execute one episode with detailed step-by-step information.

    Returns:
        dict with:
            - trajectory: list of states visited
            - reached_goal: bool
            - step_details: list of dicts with per-step information
            - episode_summary: dict with episode-level metrics
    """
    current_literal = source_literal
    trajectory = [source_literal]
    step_details = []
    entropies = []
    policy_weights = []
    chunkability_values = []  # Track individual chunkability values for running average

    # Reset context trace at start of episode
    policy_net.reset_context()
    # Note: decay will be computed each step based on running average chunkability

    target_idx = literal_to_token_idx(target_literal, num_vars)

    for step in range(max_steps):
        # Check if we've reached the goal
        if current_literal == target_literal:
            # Compute episode summary
            avg_entropy = np.mean(entropies) if len(entropies) > 0 else 0.0
            chunkability = compute_trajectory_chunkability(entropies)
            confidence = np.mean(policy_weights) if len(policy_weights) > 0 else 1.0

            episode_summary = {
                'trajectory_length': len(trajectory),
                'reached_goal': True,
                'avg_entropy': avg_entropy,
                'chunkability': chunkability,
                'confidence': confidence,
                'lambda': chunkability ** 2.5  # default lambda_exponent
            }

            return {
                'trajectory': trajectory,
                'reached_goal': True,
                'step_details': step_details,
                'episode_summary': episode_summary
            }

        # Get valid next nodes
        adjacent_literals = get_adjacent_nodes(current_literal, graph)

        # Create one-hot encodings
        current_idx = literal_to_token_idx(current_literal, num_vars)
        source_encoding = np.zeros(2 * num_vars)
        source_encoding[current_idx] = 1.0

        target_encoding = np.zeros(2 * num_vars)
        target_encoding[target_idx] = 1.0

        # Get current context (before updating)
        context_encoding = policy_net.context_trace.copy()

        # Get policy prediction with context
        action_dist = policy_net.predict(source_encoding, target_encoding, context=context_encoding).flatten()

        # Compute entropy
        non_zero_mask = action_dist > 0
        entropy = -np.sum(action_dist[non_zero_mask] * np.log2(action_dist[non_zero_mask]))
        entropies.append(entropy)

        # Get value estimate
        value_estimate = policy_net.compute_value(current_literal, target_literal, context=context_encoding)

        # Compute mixing weight using sigmoid
        if entropy_threshold is not None:
            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))

            policy_weight = sigmoid(oracle_sensitivity * (entropy_threshold - entropy))
            policy_weights.append(policy_weight)

            # Get optimal action
            optimal_trajectory = generate_trajectory_from_graph(current_literal, target_literal, graph)
            if optimal_trajectory is not None and len(optimal_trajectory) > 1:
                optimal_next_literal = optimal_trajectory[1]
                optimal_idx = literal_to_token_idx(optimal_next_literal, num_vars)
                optimal_dist = np.zeros_like(action_dist)
                optimal_dist[optimal_idx] = 1.0
            else:
                optimal_dist = action_dist.copy()

            # Mix distributions
            mixed_dist = policy_weight * action_dist + (1.0 - policy_weight) * optimal_dist
            mixed_dist = mixed_dist / np.sum(mixed_dist)

            # Sample from mixed distribution
            next_idx = np.random.choice(len(mixed_dist), p=mixed_dist)
            oracle_called = (policy_weight < 0.5)  # Oracle dominates if policy_weight < 0.5
        else:
            policy_weights.append(1.0)
            next_idx = np.random.choice(len(action_dist), p=action_dist)
            oracle_called = False

        next_literal = token_idx_to_literal(next_idx, num_vars)

        # Get top 3 actions from policy
        top_3_indices = np.argsort(action_dist)[-3:][::-1]
        top_3_actions = [(token_idx_to_literal(idx, num_vars), action_dist[idx]) for idx in top_3_indices]

        # Compute per-step chunkability and add to running average
        # Chunkability at this step = 1 / exp(H)
        step_chunkability_instant = 1.0 / np.exp(entropy)
        chunkability_values.append(step_chunkability_instant)

        # Running average chunkability (average of all steps so far)
        running_avg_chunkability = np.mean(chunkability_values)

        # Lambda based on running average chunkability
        step_lambda = running_avg_chunkability ** lambda_exponent

        # Record step details
        step_info = {
            'step': step,
            'current_state': current_literal,
            'next_state': next_literal,
            'action_dist': action_dist.copy(),
            'top_3_actions': top_3_actions,
            'entropy': entropy,
            'value_estimate': value_estimate,
            'context_trace': context_encoding.copy(),
            'context_trace_sum': np.sum(context_encoding),
            'context_trace_nonzero': np.count_nonzero(context_encoding),
            'oracle_called': oracle_called,
            'policy_weight': policy_weights[-1] if policy_weights else 1.0,
            'chunkability': running_avg_chunkability,  # Running average
            'lambda': step_lambda
        }
        step_details.append(step_info)

        # Update context trace with current state (for next step)
        # Use the lambda derived from current step's chunkability
        decay = gamma * step_lambda
        policy_net.update_context(current_literal, decay)

        # Update state
        current_literal = next_literal
        trajectory.append(current_literal)

    # Max steps reached without reaching goal
    avg_entropy = np.mean(entropies) if len(entropies) > 0 else 0.0
    chunkability = compute_trajectory_chunkability(entropies)
    confidence = np.mean(policy_weights) if len(policy_weights) > 0 else 1.0

    episode_summary = {
        'trajectory_length': len(trajectory),
        'reached_goal': False,
        'avg_entropy': avg_entropy,
        'chunkability': chunkability,
        'confidence': confidence,
        'lambda': chunkability ** 2.5
    }

    return {
        'trajectory': trajectory,
        'reached_goal': False,
        'step_details': step_details,
        'episode_summary': episode_summary
    }


# ============================================================================
# Display Functions
# ============================================================================

def print_episode_header(source, target, optimal_path_length, network_label):
    """Print header for an episode."""
    print("\n" + "="*100)
    print(f"{network_label}")
    print(f"Episode: {source} → {target} (Optimal path length: {optimal_path_length})")
    print("="*100)


def print_episode_summary(episode_summary):
    """Print episode-level summary statistics."""
    print("\n" + "-"*100)
    print("EPISODE SUMMARY")
    print("-"*100)
    print(f"  Reached Goal:       {'YES' if episode_summary['reached_goal'] else 'NO'}")
    print(f"  Trajectory Length:  {episode_summary['trajectory_length']}")
    print(f"  Average Entropy:    {episode_summary['avg_entropy']:.4f}")
    print(f"  Chunkability:       {episode_summary['chunkability']:.4f}")
    print(f"  Confidence:         {episode_summary['confidence']:.4f}")
    print(f"  Lambda (λ):         {episode_summary['lambda']:.4f}")
    print("-"*100)


def print_step_table(step_details):
    """Print a table of step-by-step details."""
    if len(step_details) == 0:
        print("\nNo steps taken (already at goal)")
        return

    print("\nSTEP-BY-STEP DETAILS")
    print("-"*135)

    # Header
    header = f"{'Step':<6} {'Current':<8} {'Next':<8} {'Top 3 Actions (literal: prob)':<40} {'Entropy':<10} {'Value':<10} {'Chunk':<10} {'Lambda':<10} {'Oracle?':<8} {'Ctx Sum':<10} {'Ctx Nonzero':<12}"
    print(header)
    print("-"*135)

    # Rows
    for step_info in step_details:
        top_3_str = ", ".join([f"{lit}: {prob:.3f}" for lit, prob in step_info['top_3_actions']])
        oracle_str = "YES" if step_info['oracle_called'] else "NO"

        row = (f"{step_info['step']:<6} "
               f"{step_info['current_state']:<8} "
               f"{step_info['next_state']:<8} "
               f"{top_3_str:<40} "
               f"{step_info['entropy']:<10.4f} "
               f"{step_info['value_estimate']:<10.4f} "
               f"{step_info['chunkability']:<10.4f} "
               f"{step_info['lambda']:<10.4f} "
               f"{oracle_str:<8} "
               f"{step_info['context_trace_sum']:<10.4f} "
               f"{step_info['context_trace_nonzero']:<12}")
        print(row)

    print("-"*135)


def print_context_traces(step_details, num_vars):
    """Print full context trace vectors for each step."""
    if len(step_details) == 0:
        return

    print("\nCONTEXT TRACE DETAILS")
    print("-"*100)

    for step_info in step_details:
        print(f"\nStep {step_info['step']} (State {step_info['current_state']} → {step_info['next_state']}):")

        # Show context trace as a compact representation
        ctx = step_info['context_trace']
        nonzero_indices = np.nonzero(ctx)[0]

        if len(nonzero_indices) == 0:
            print("  Context trace: [all zeros]")
        else:
            print(f"  Context trace (showing {len(nonzero_indices)} non-zero entries):")
            for idx in nonzero_indices:
                literal = token_idx_to_literal(idx, num_vars)
                print(f"    Literal {literal:>3} (idx {idx:>3}): {ctx[idx]:.6f}")

    print("-"*100)


# ============================================================================
# Main Visualization Function
# ============================================================================

def visualize_trajectories(graph_path, trained_weights_dir=None, num_examples=5,
                          hidden_size=25, learning_rate=0.225,
                          gamma=0.99, lambda_exponent=2.5, entropy_threshold=0.75,
                          oracle_sensitivity=5.0, max_steps=100, seed=None,
                          fixed_pairs=None, load_value_weights=True):
    """
    Visualize trajectory behavior before and after training.

    Args:
        graph_path: str - path to saved graph .pkl file
        trained_weights_dir: str or None - directory containing trained weights (None = skip trained comparison)
        num_examples: int - number of example trajectories to show (default 5, ignored if fixed_pairs is provided)
        hidden_size: int - hidden layer size for networks
        learning_rate: float - learning rate (for network initialization)
        gamma: float - discount factor
        lambda_exponent: float - exponent for lambda modulation
        entropy_threshold: float - threshold for entropy-based mixing
        oracle_sensitivity: float - sigmoid steepness for oracle mixing
        max_steps: int - maximum steps per episode
        seed: int or None - random seed for reproducibility
        fixed_pairs: list of tuples or None - specific (source, target) pairs to visualize
                     e.g., [(1, 4), (5, -3)]. If provided, num_examples is ignored.
        load_value_weights: bool - whether to load value head weights for trained network (default True)
                           Set to False to see what happens when value head wasn't trained
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Load graph
    print(f"Loading graph from {graph_path}")
    graph = load_graph(graph_path)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    print(f"Graph loaded: {num_nodes} nodes, {num_edges} edges")

    # Infer num_vars from graph
    all_literals = list(graph.nodes())
    num_vars = max([abs(lit) for lit in all_literals])
    print(f"Detected {num_vars} variables")

    # Get valid (source, target) pairs
    next_steps = gen_impgraph.compute_next_steps(graph, num_vars)
    valid_pairs = list(next_steps.keys())
    print(f"Found {len(valid_pairs)} valid (source, target) pairs")

    # Determine which pairs to visualize
    if fixed_pairs is not None:
        # Use fixed pairs
        sampled_pairs = fixed_pairs
        print(f"\nUsing {len(sampled_pairs)} fixed (source, target) pairs:")
        for i, (s, t) in enumerate(sampled_pairs, 1):
            optimal_path = generate_trajectory_from_graph(s, t, graph)
            opt_len = len(optimal_path) if optimal_path else "N/A"
            print(f"  {i}. {s} → {t} (optimal length: {opt_len})")
    else:
        # Sample random pairs
        if num_examples > len(valid_pairs):
            print(f"Warning: requested {num_examples} examples but only {len(valid_pairs)} valid pairs exist")
            num_examples = len(valid_pairs)

        sampled_pairs = random.sample(valid_pairs, num_examples)
        print(f"\nSampled {num_examples} random (source, target) pairs:")
        for i, (s, t) in enumerate(sampled_pairs, 1):
            optimal_path = generate_trajectory_from_graph(s, t, graph)
            opt_len = len(optimal_path) if optimal_path else "N/A"
            print(f"  {i}. {s} → {t} (optimal length: {opt_len})")

    # Create untrained network
    print(f"\nInitializing untrained network...")
    untrained_net = PolicyNetworkPC(
        num_vars=num_vars,
        graph=graph,
        policy_name='Untrained_Policy',
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        source_to_hidden_matrix=None,
        target_to_hidden_matrix=None,
        hidden_to_output_matrix=None,
        hidden_to_value_matrix=None
    )

    # Create trained network if weights provided
    trained_net = None
    if trained_weights_dir is not None:
        print(f"\nLoading trained weights from {trained_weights_dir}...")
        policy_weights = load_weight_matrices(trained_weights_dir, network_type='policy')

        # Load value weights if requested and they exist
        hidden_to_value_matrix = None
        if load_value_weights:
            hidden_to_value_path = os.path.join(trained_weights_dir, 'policy_hidden_to_value.npy')
            if os.path.exists(hidden_to_value_path):
                hidden_to_value_matrix = np.load(hidden_to_value_path)
                print(f"Loaded value head weights")
            else:
                print(f"Warning: Value head weights not found at {hidden_to_value_path}")
        else:
            print(f"Skipping value head weights (load_value_weights=False)")

        trained_net = PolicyNetworkPC(
            num_vars=num_vars,
            graph=graph,
            policy_name='Trained_Policy',
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            source_to_hidden_matrix=policy_weights[0],
            target_to_hidden_matrix=policy_weights[1],
            hidden_to_output_matrix=policy_weights[2],
            hidden_to_value_matrix=hidden_to_value_matrix
        )

    # Run episodes for each sampled pair
    print("\n" + "="*100)
    print("RUNNING EPISODES")
    print("="*100)

    for i, (source, target) in enumerate(sampled_pairs, 1):
        optimal_path = generate_trajectory_from_graph(source, target, graph)
        optimal_length = len(optimal_path) if optimal_path else None

        print("\n\n" + "#"*100)
        print(f"EXAMPLE {i}/{num_examples}")
        print("#"*100)

        # Run untrained episode
        print_episode_header(source, target, optimal_length, "UNTRAINED NETWORK")
        untrained_result = run_episode_verbose(
            untrained_net, source, target, graph, num_vars,
            max_steps=max_steps,
            entropy_threshold=entropy_threshold,
            oracle_sensitivity=oracle_sensitivity,
            gamma=gamma,
            lambda_=0.0,  # Will be computed from chunkability
            lambda_exponent=lambda_exponent
        )

        print_step_table(untrained_result['step_details'])
        print_context_traces(untrained_result['step_details'], num_vars)
        print_episode_summary(untrained_result['episode_summary'])

        # Run trained episode if available
        if trained_net is not None:
            print("\n")
            print_episode_header(source, target, optimal_length, "TRAINED NETWORK")
            trained_result = run_episode_verbose(
                trained_net, source, target, graph, num_vars,
                max_steps=max_steps,
                entropy_threshold=entropy_threshold,
                oracle_sensitivity=oracle_sensitivity,
                gamma=gamma,
                lambda_=0.0,
                lambda_exponent=lambda_exponent
            )

            print_step_table(trained_result['step_details'])
            print_context_traces(trained_result['step_details'], num_vars)
            print_episode_summary(trained_result['episode_summary'])

    print("\n\n" + "="*100)
    print("VISUALIZATION COMPLETE")
    print("="*100)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Configuration
    num_vars = 32
    num_clauses = 75

    # Paths
    graph_path = f'results/graphs/impgraph_{num_vars}v_{num_clauses}c/graph.pkl'
    trained_weights_dir = f'results/td_n/final_weights/impgraph_{num_vars}v_{num_clauses}c_tdlambda'

    # If you want to skip trained network comparison, set to None:
    # trained_weights_dir = None

    # Hyperparameters (should match training configuration)
    hidden_size = 25
    learning_rate = 0.225
    gamma = 0.99
    lambda_exponent = 2.5
    entropy_threshold = 0.75
    oracle_sensitivity = 5.0

    # Option 1: Visualize the specific training pair (if trained on fixed pair)
    # For example, if trained on fixed_pair (1, 4):
    visualize_trajectories(
        graph_path=graph_path,
        trained_weights_dir=trained_weights_dir,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        gamma=gamma,
        lambda_exponent=lambda_exponent,
        entropy_threshold=entropy_threshold,
        oracle_sensitivity=oracle_sensitivity,
        max_steps=20,
        seed=42,  # For reproducibility
        fixed_pairs=[(1, 4)],  # Visualize the training pair
        load_value_weights=True  # Load trained value head weights (default)
    )

    # Option 2: Visualize multiple specific pairs
    # visualize_trajectories(
    #     graph_path=graph_path,
    #     trained_weights_dir=trained_weights_dir,
    #     hidden_size=hidden_size,
    #     learning_rate=learning_rate,
    #     gamma=gamma,
    #     lambda_exponent=lambda_exponent,
    #     entropy_threshold=entropy_threshold,
    #     oracle_sensitivity=oracle_sensitivity,
    #     max_steps=20,
    #     seed=42,
    #     fixed_pairs=[(1, 4), (5, -3), (10, 15)],  # Multiple specific pairs
    #     load_value_weights=True
    # )

    # Option 3: Visualize random pairs
    # visualize_trajectories(
    #     graph_path=graph_path,
    #     trained_weights_dir=trained_weights_dir,
    #     num_examples=5,  # Number of random pairs
    #     hidden_size=hidden_size,
    #     learning_rate=learning_rate,
    #     gamma=gamma,
    #     lambda_exponent=lambda_exponent,
    #     entropy_threshold=entropy_threshold,
    #     oracle_sensitivity=oracle_sensitivity,
    #     max_steps=20,
    #     seed=42,
    #     load_value_weights=True
    # )

    # Option 4: Compare with untrained value head (trained policy only)
    # Useful to see if value head training matters
    # visualize_trajectories(
    #     graph_path=graph_path,
    #     trained_weights_dir=trained_weights_dir,
    #     hidden_size=hidden_size,
    #     learning_rate=learning_rate,
    #     gamma=gamma,
    #     lambda_exponent=lambda_exponent,
    #     entropy_threshold=entropy_threshold,
    #     oracle_sensitivity=oracle_sensitivity,
    #     max_steps=20,
    #     seed=42,
    #     fixed_pairs=[(1, 4)],
    #     load_value_weights=False  # Skip value head weights, use random initialization
    # )
