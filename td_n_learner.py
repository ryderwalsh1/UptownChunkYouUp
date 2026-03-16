import numpy as np
import random
import warnings
import pickle
import gen_impgraph
import networkx as nx
from policy_network import PolicyNetwork
from policy_network_pc import PolicyNetworkPC
from value_network import ValueNetwork
import matplotlib.pyplot as plt
import csv
import os

warnings.filterwarnings("ignore", category=UserWarning, module='psyneulink')


# ============================================================================
# Graph and Weight Loading Functions
# ============================================================================

def load_graph(graph_path):
    """
    Load a saved implication graph from a pickle file.

    Args:
        graph_path: str - path to the .pkl file containing the graph

    Returns:
        networkx.DiGraph - the loaded implication graph
    """
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def load_weight_matrices(weights_dir, network_type='policy', epoch=None):
    """
    Load pre-trained weight matrices from disk.

    Args:
        weights_dir: str or None - directory containing the weight .npy files (None returns None)
        network_type: str - 'policy' or 'value' (used for naming in flat directory format)
        epoch: int or None - specific epoch to load (None = latest epoch)

    Returns:
        tuple or None - (source_matrix, target_matrix, output_matrix) or None if weights_dir is None

    Expected files in weights_dir:
        - source_to_hidden/epoch_{N}.npy (or policy_source_to_hidden.npy)
        - target_to_hidden/epoch_{N}.npy (or policy_target_to_hidden.npy)
        - hidden_to_output/epoch_{N}.npy (or policy_hidden_to_output.npy)
    """
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


# ============================================================================
# Utility Functions
# ============================================================================

def literal_to_token_idx(literal, num_vars):
    """
    Map literal to token index.

    Positive literals (1 to num_vars) → indices 0 to num_vars-1
    Negative literals (-1 to -num_vars) → indices num_vars to 2*num_vars-1

    Args:
        literal: int - literal value (positive or negative)
        num_vars: int - number of variables

    Returns:
        int - token index (0 to 2*num_vars-1)
    """
    if literal > 0:
        return literal - 1
    else:
        return num_vars + abs(literal) - 1


def token_idx_to_literal(idx, num_vars):
    """
    Map token index to literal.

    Args:
        idx: int - token index (0 to 2*num_vars-1)
        num_vars: int - number of variables

    Returns:
        int - literal value
    """
    if idx < num_vars:
        return idx + 1
    else:
        return -(idx - num_vars + 1)


def get_adjacent_nodes(current_literal, graph):
    """
    Get valid next nodes (neighbors) from the current literal in the graph.

    Args:
        current_literal: int - current literal
        graph: nx.DiGraph - implication graph

    Returns:
        list[int] - list of adjacent literals
    """
    # NetworkX successors gives us the outgoing neighbors
    return list(graph.successors(current_literal))


def mask_action_distribution(action_dist, valid_literals, num_vars):
    """
    Apply action masking to policy output: zero out non-adjacent nodes and renormalize.

    Args:
        action_dist: np.array - probability distribution over all literals
        valid_literals: list[int] - list of valid next literals
        num_vars: int - number of variables

    Returns:
        np.array - masked and renormalized probability distribution
    """
    masked_dist = np.zeros_like(action_dist)

    # Set probabilities for valid actions
    for lit in valid_literals:
        idx = literal_to_token_idx(lit, num_vars)
        masked_dist[idx] = action_dist[idx]

    # Renormalize
    total = np.sum(masked_dist)
    if total > 0:
        masked_dist = masked_dist / total
    else:
        # If all valid actions had zero probability, uniform over valid actions
        for lit in valid_literals:
            idx = literal_to_token_idx(lit, num_vars)
            masked_dist[idx] = 1.0 / len(valid_literals)

    return masked_dist


# ============================================================================
# Trajectory Generation from Graph
# ============================================================================

def generate_trajectory_from_graph(source_literal, target_literal, graph):
    """
    Generate a trajectory from source to target using the implication graph.
    Uses shortest path if available, otherwise returns None.

    Args:
        source_literal: int - starting literal
        target_literal: int - goal literal
        graph: nx.DiGraph - implication graph

    Returns:
        list[int] or None - trajectory from source to target (inclusive), or None if no path exists
    """
    try:
        # Find shortest path from source to target
        path = nx.shortest_path(graph, source=source_literal, target=target_literal)
        return path
    except nx.NetworkXNoPath:
        # No path exists
        return None


# ============================================================================
# Evaluation Functions
# ============================================================================

def test_loss_policy(policy_net, graph, num_vars, num_samples=None, fixed_source=None, fixed_target=None):
    """
    Evaluate policy network loss (cross-entropy) on optimal trajectories.

    Evaluates all valid (source, target) pairs, generates optimal trajectories,
    and computes the average cross-entropy loss between the policy's
    predictions and the optimal actions at each step (after action masking).

    Args:
        policy_net: PolicyNetwork - trained policy network
        graph: nx.DiGraph - implication graph
        num_vars: int - number of variables
        num_samples: int or None - ignored (kept for backward compatibility)
        fixed_source: int or None - if provided with fixed_target, evaluate only this specific pair
        fixed_target: int or None - if provided with fixed_source, evaluate only this specific pair

    Returns:
        float - average cross-entropy loss (lower is better)
    """
    # Get all valid (source, target) pairs
    next_steps = gen_impgraph.compute_next_steps(graph, num_vars)
    valid_pairs = list(next_steps.keys())

    if len(valid_pairs) == 0:
        return 0.0

    # Determine which pairs to evaluate
    if fixed_source is not None and fixed_target is not None:
        # Evaluate only the specified fixed pair
        pairs_to_evaluate = [(fixed_source, fixed_target)]
    else:
        # Evaluate all pairs
        pairs_to_evaluate = valid_pairs

    losses = []

    # Evaluate pairs
    for source_literal, target_literal in pairs_to_evaluate:

        # Generate optimal trajectory
        trajectory = generate_trajectory_from_graph(source_literal, target_literal, graph)
        if trajectory is None or len(trajectory) <= 1:
            continue

        target_idx = literal_to_token_idx(target_literal, num_vars)
        target_encoding = np.zeros(2 * num_vars)
        target_encoding[target_idx] = 1.0

        # Evaluate policy at each step in the trajectory
        for t in range(len(trajectory) - 1):
            current_literal = trajectory[t]
            optimal_next_literal = trajectory[t + 1]

            # Get adjacent nodes for action masking
            # adjacent_literals = get_adjacent_nodes(current_literal, graph)
            # if len(adjacent_literals) == 0:
            #     continue

            # Create source encoding
            current_idx = literal_to_token_idx(current_literal, num_vars)
            source_encoding = np.zeros(2 * num_vars)
            source_encoding[current_idx] = 1.0

            # Get policy prediction
            action_dist = policy_net.predict(source_encoding, target_encoding).flatten()

            # Apply action masking
            # masked_dist = mask_action_distribution(action_dist, adjacent_literals, num_vars)

            # Get probability assigned to optimal action
            optimal_idx = literal_to_token_idx(optimal_next_literal, num_vars)
            optimal_prob = action_dist[optimal_idx]

            # Compute cross-entropy loss: -log(p(optimal_action))
            # Add small epsilon to avoid log(0)
            cross_entropy_loss = -np.log(optimal_prob + 1e-10)
            losses.append(cross_entropy_loss)

            # Print predictions and targets
            # if current_literal == 5 and optimal_next_literal == -3:  # Print for a few samples of a specific pair
            #     predicted_literal = token_idx_to_literal(np.argmax(masked_dist), num_vars)
            #     print(f"  Step {t}: current={current_literal}, optimal_next={optimal_next_literal}")
            #     print(f"    Network prediction (masked): {masked_dist}")
            #     print(f"    Predicted literal: {predicted_literal}")
            #     print(f"    Match: {predicted_literal == optimal_next_literal}")

    if len(losses) == 0:
        return 0.0

    return np.mean(losses)


def test_loss_value(value_net, graph, num_vars, num_samples=50):
    """
    Evaluate value network using Bellman error on optimal trajectories.

    Samples random (source, target) pairs, generates optimal trajectories,
    and computes mean squared Bellman error: (V(s) - (r + V(s')))^2
    where r = 1.0 if s' is goal (and V(s') = 0), else r = 0.0.

    Args:
        value_net: ValueNetwork - trained value network
        graph: nx.DiGraph - implication graph
        num_vars: int - number of variables
        num_samples: int - number of (source, target) pairs to sample

    Returns:
        float - mean Bellman error (lower is better)
    """
    # Get all valid (source, target) pairs
    next_steps = gen_impgraph.compute_next_steps(graph, num_vars)
    valid_pairs = list(next_steps.keys())

    if len(valid_pairs) == 0:
        return 0.0

    bellman_errors = []

    # Sample and evaluate
    for _ in range(num_samples):
        source_literal, target_literal = random.choice(valid_pairs)

        # Generate optimal trajectory
        trajectory = generate_trajectory_from_graph(source_literal, target_literal, graph)
        if trajectory is None or len(trajectory) <= 1:
            continue

        target_idx = literal_to_token_idx(target_literal, num_vars)
        target_encoding = np.zeros(2 * num_vars)
        target_encoding[target_idx] = 1.0

        # Evaluate value at each step in the trajectory
        for t in range(len(trajectory)):
            current_literal = trajectory[t]

            # Create source encoding
            current_idx = literal_to_token_idx(current_literal, num_vars)
            source_encoding = np.zeros(2 * num_vars)
            source_encoding[current_idx] = 1.0

            # Get value prediction for current state
            v_current = value_net.predict(source_encoding, target_encoding)

            # Compute Bellman target
            if current_literal == target_literal:
                # At goal: reward = 1.0, V(s') = 0 (episode ends)
                bellman_target = 1.0
            else:
                # Not at goal: reward = 0.0, bootstrap with V(s')
                next_literal = trajectory[t + 1] if t + 1 < len(trajectory) else current_literal
                next_idx = literal_to_token_idx(next_literal, num_vars)
                next_encoding = np.zeros(2 * num_vars)
                next_encoding[next_idx] = 1.0

                # If next state is goal, V(s') = 0
                if next_literal == target_literal:
                    bellman_target = 1.0  # reward for reaching goal
                else:
                    v_next = value_net.predict(next_encoding, target_encoding)
                    bellman_target = 0.0 + v_next

            # Compute Bellman error
            bellman_error = (v_current - bellman_target) ** 2
            bellman_errors.append(bellman_error)

    if len(bellman_errors) == 0:
        return 0.0

    return np.mean(bellman_errors)


# ============================================================================
# Policy Evaluation
# ============================================================================

def compute_trajectory_chunkability(entropies):
    """
    Compute the chunkability (corridor-likeness) of a trajectory.

    At each step, computes exp(H_t) which represents the effective number of actions
    the policy is considering. The chunkability is the average of 1/exp(H_t) across steps:
    - 1.0 for perfectly deterministic policy (one action considered)
    - Approaches 0 for diffuse/uniform policy (many actions considered)

    Args:
        entropies: list[float] - entropy values at each step of the trajectory

    Returns:
        float - chunkability metric (1.0 = corridor-like, 0.0 = diffuse)
    """
    if len(entropies) == 0:
        return 0.0

    # Compute effective number of actions at each step: exp(H_t)
    effective_actions = [np.exp(h) for h in entropies]

    # Chunkability is average of 1 / effective_actions
    chunkability_values = [1.0 / ea for ea in effective_actions]

    return np.mean(chunkability_values)


def run_episode(policy_net, source_literal, target_literal, graph, num_vars, max_steps=100, entropy_threshold=1.5, oracle_sensitivity=5.0):
    """
    Execute one episode from source to target using the policy network with entropy-based mixing.

    Args:
        policy_net: PolicyNetwork - trained policy network
        source_literal: int - starting literal
        target_literal: int - goal literal
        graph: nx.DiGraph - implication graph
        num_vars: int - number of variables
        max_steps: int - maximum steps before termination
        entropy_threshold: float or None - center point for entropy-based mixing; if None, always samples
                                           from policy distribution without mixing
        oracle_sensitivity: float - controls steepness of sigmoid transition (higher = sharper)

    Returns:
        tuple: (trajectory, reached_goal, avg_entropy, chunkability, confidence)
            trajectory: list[int] - sequence of literals visited
            reached_goal: bool - whether target was reached
            avg_entropy: float - average entropy of policy decisions across the episode
            chunkability: float - corridor-likeness metric (1.0 = deterministic, 0.0 = diffuse)
            confidence: float - average policy weight (presence of policy network in action selection)
    """
    current_literal = source_literal
    trajectory = [source_literal]
    entropies = []
    policy_weights = []

    target_idx = literal_to_token_idx(target_literal, num_vars)

    for step in range(max_steps):
        # Check if we've reached the goal
        if current_literal == target_literal:
            avg_entropy = np.mean(entropies) if len(entropies) > 0 else 0.0
            chunkability = compute_trajectory_chunkability(entropies)
            confidence = np.mean(policy_weights) if len(policy_weights) > 0 else 1.0
            return trajectory, True, avg_entropy, chunkability, confidence

        # Get valid next nodes (adjacent in graph)
        adjacent_literals = get_adjacent_nodes(current_literal, graph)

        # If no valid moves, episode ends
        # if len(adjacent_literals) == 0:
        #     avg_entropy = np.mean(entropies) if len(entropies) > 0 else 0.0
        #     return trajectory, False, avg_entropy

        # Create one-hot encodings
        current_idx = literal_to_token_idx(current_literal, num_vars)
        source_encoding = np.zeros(2 * num_vars)
        source_encoding[current_idx] = 1.0

        target_encoding = np.zeros(2 * num_vars)
        target_encoding[target_idx] = 1.0

        # Get policy prediction
        action_dist = policy_net.predict(source_encoding, target_encoding).flatten()

        # if step == 0:
        #     print(f"Initial policy distribution for source {current_literal} to next -6:")
        #     print(action_dist)

        # Compute entropy of the policy's action distribution (before masking)
        entropy = policy_net._decision_entropy(action_dist)
        entropies.append(entropy)

        # Choose action: apply entropy-based mixing if threshold is set
        if entropy_threshold is not None:
            # Compute mixing weight using sigmoid: p(use_policy) = σ(k · (H_threshold - H_current))
            # When H_current > H_threshold: weight approaches 0 (use memory)
            # When H_current < H_threshold: weight approaches 1 (use policy)
            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))

            policy_weight = sigmoid(oracle_sensitivity * (entropy_threshold - entropy))
            policy_weights.append(policy_weight)

            # Get optimal action distribution (one-hot at optimal next literal)
            optimal_trajectory = generate_trajectory_from_graph(current_literal, target_literal, graph)
            if optimal_trajectory is not None and len(optimal_trajectory) > 1:
                optimal_next_literal = optimal_trajectory[1]
                optimal_idx = literal_to_token_idx(optimal_next_literal, num_vars)
                optimal_dist = np.zeros_like(action_dist)
                optimal_dist[optimal_idx] = 1.0
            else:
                # No optimal path found, fallback to policy only
                optimal_dist = action_dist.copy()

            # Mix distributions: mixed_dist = policy_weight * policy + (1 - policy_weight) * optimal
            mixed_dist = policy_weight * action_dist + (1.0 - policy_weight) * optimal_dist

            # Renormalize (should already sum to 1, but ensure numerical stability)
            mixed_dist = mixed_dist / np.sum(mixed_dist)

            # Sample from mixed distribution
            next_idx = np.random.choice(len(mixed_dist), p=mixed_dist)
        else:
            # No threshold set, sample from policy distribution
            policy_weights.append(1.0)  # Full confidence in policy when no mixing
            next_idx = np.random.choice(len(action_dist), p=action_dist)

        next_literal = token_idx_to_literal(next_idx, num_vars)

        # Update state
        current_literal = next_literal
        trajectory.append(current_literal)

    # Max steps reached without reaching goal
    avg_entropy = np.mean(entropies) if len(entropies) > 0 else 0.0
    chunkability = compute_trajectory_chunkability(entropies)
    confidence = np.mean(policy_weights) if len(policy_weights) > 0 else 1.0

    return trajectory, False, avg_entropy, chunkability, confidence


# ============================================================================
# Teacher Forcing Training Functions
# ============================================================================

def train_policy(policy_net, trajectory, target_literal, num_vars):
    """
    Train policy network on a trajectory using episodic teacher forcing.
    Updates are applied at every step during the episode, with the target
    being the next literal in the optimal trajectory (one-hot encoded).
    No action masking, no n-step returns, no value network bootstrapping.

    Args:
        policy_net: PolicyNetwork - policy network to train
        trajectory: list[int] - sequence of literals from source to target (optimal path)
        target_literal: int - goal literal
        num_vars: int - number of variables
    """
    vocab_size = 2 * num_vars
    target_idx = literal_to_token_idx(target_literal, num_vars)
    target_encoding = np.zeros(vocab_size)
    target_encoding[target_idx] = 1.0

    # Prepare batch data for entire trajectory
    source_encodings = []
    target_encodings = []
    policy_targets = []

    # Collect all training examples from trajectory
    for t in range(len(trajectory) - 1):
        # Current state encoding
        current_literal = trajectory[t]
        current_idx = literal_to_token_idx(current_literal, num_vars)
        source_encoding = np.zeros(vocab_size)
        source_encoding[current_idx] = 1.0

        # Create target distribution: one-hot at next literal in optimal path
        next_literal = trajectory[t + 1]
        next_idx = literal_to_token_idx(next_literal, num_vars)
        policy_target = np.zeros(vocab_size)
        policy_target[next_idx] = 1.0

        # Collect for batch update
        source_encodings.append(source_encoding)
        target_encodings.append(target_encoding)
        policy_targets.append(policy_target)

    # Apply batch update for entire trajectory (matching mlpcomposition.py format)
    if len(source_encodings) > 0:
        policy_net.update_batch(
            np.array(source_encodings),
            np.array(target_encodings),
            np.array(policy_targets)
        )


# ============================================================================
# TD(n) Training Functions
# ============================================================================

def train_policy_td_n(policy_net, value_net, trajectory, target_literal, n_steps, num_vars):
    """
    Train policy network on a trajectory using online TD(n) updates.
    Updates are applied at every step during the episode.

    Args:
        policy_net: PolicyNetwork - policy network to train
        trajectory: list[int] - sequence of literals from source to target
        target_literal: int - goal literal
        n_steps: int - number of steps for TD(n)
        num_vars: int - number of variables
    """
    vocab_size = 2 * num_vars
    target_idx = literal_to_token_idx(target_literal, num_vars)
    target_encoding = np.zeros(vocab_size)
    target_encoding[target_idx] = 1.0

    # Apply online updates at every step
    for t in range(len(trajectory)):
        # Compute n-step return
        # G_t = r_t + r_{t+1} + ... + r_{t+n-1} + V(s_{t+n})

        n_step_return = 0.0
        episode_end = len(trajectory) - 1

        # Sum rewards for next n steps (or until episode ends)
        for i in range(n_steps):
            step_idx = t + i
            if step_idx > episode_end:
                break

            # Reward: 1.0 if this step reaches the goal, 0.0 otherwise
            if trajectory[step_idx] == target_literal:
                n_step_return += 1.0
                break  # Episode ends at goal

        # Bootstrap with policy network if episode continues beyond t+n
        bootstrap_idx = t + n_steps
        if bootstrap_idx <= episode_end and trajectory[bootstrap_idx - 1] != target_literal:
            # Get state at t+n
            state_literal = trajectory[bootstrap_idx]
            state_idx = literal_to_token_idx(state_literal, num_vars)
            state_encoding = np.zeros(vocab_size)
            state_encoding[state_idx] = 1.0

            # Get value prediction for bootstrapping
            bootstrap_value = value_net.predict(state_encoding, target_encoding)
            n_step_return += bootstrap_value    
        
        # print(f"Step {t} | n-step return: {n_step_return:.4f} | Next literal: {trajectory[t+1] if t < episode_end else 'N/A'}")

        # Create target distribution: put all mass at next literal in trajectory
        policy_target = np.zeros(vocab_size)
        if t < episode_end:
            next_literal = trajectory[t + 1]
            next_idx = literal_to_token_idx(next_literal, num_vars)
            policy_target[next_idx] = 1.0
        else:
            # At goal, target is goal itself
            policy_target[target_idx] = 1.0

        # Current state encoding
        current_literal = trajectory[t]
        current_idx = literal_to_token_idx(current_literal, num_vars)
        source_encoding = np.zeros(vocab_size)
        source_encoding[current_idx] = 1.0

        # Apply update
        policy_net.update_single(source_encoding, target_encoding, policy_target)


def train_value_td_n(value_net, trajectory, target_literal, n_steps, num_vars):
    """
    Train value network on a trajectory using online TD(n) updates.
    Updates are applied at every step during the episode.

    Args:
        value_net: ValueNetwork - value network to train
        trajectory: list[int] - sequence of literals from source to target
        target_literal: int - goal literal
        n_steps: int - number of steps for TD(n)
        num_vars: int - number of variables
    """
    vocab_size = 2 * num_vars
    target_idx = literal_to_token_idx(target_literal, num_vars)
    target_encoding = np.zeros(vocab_size)
    target_encoding[target_idx] = 1.0

    # Apply online updates at every step
    for t in range(len(trajectory)):
        # Compute n-step return
        # G_t = r_t + r_{t+1} + ... + r_{t+n-1} + V(s_{t+n})

        n_step_return = 0.0
        episode_end = len(trajectory) - 1

        # Sum rewards for next n steps (or until episode ends)
        for i in range(n_steps):
            step_idx = t + i
            if step_idx > episode_end:
                break

            # Reward: 1.0 if this step reaches the goal, 0.0 otherwise
            if trajectory[step_idx] == target_literal:
                n_step_return += 1.0
                break  # Episode ends at goal

        # Bootstrap with value network if episode continues beyond t+n
        bootstrap_idx = t + n_steps
        if bootstrap_idx <= episode_end and trajectory[bootstrap_idx - 1] != target_literal:
            # Get state at t+n
            state_literal = trajectory[bootstrap_idx]
            state_idx = literal_to_token_idx(state_literal, num_vars)
            state_encoding = np.zeros(vocab_size)
            state_encoding[state_idx] = 1.0

            # Get value estimate for bootstrapping
            bootstrap_value = value_net.predict(state_encoding, target_encoding)
            n_step_return += bootstrap_value

        # Current state encoding
        current_literal = trajectory[t]
        current_idx = literal_to_token_idx(current_literal, num_vars)
        source_encoding = np.zeros(vocab_size)
        source_encoding[current_idx] = 1.0

        # Apply update with scalar n-step return
        value_net.update_single(source_encoding, target_encoding, n_step_return)


# ============================================================================
# Teacher Forcing Main Training Loop
# ============================================================================

def train_teacher_forcing(policy_net, graph, num_vars, num_episodes=1000,
                          max_steps=100, capture_interval=50, eval_episodes=10, verbose=True,
                          training_mode='random_sampling', fixed_source=None, fixed_target=None,
                          entropy_threshold=1.5, oracle_sensitivity=10.0, superlinearity=2.0):
    """
    Train policy network using episodic teacher forcing (no value network).

    Args:
        policy_net: PolicyNetwork - policy network to train
        graph: nx.DiGraph - implication graph
        num_vars: int - number of variables
        num_episodes: int - number of training episodes
        max_steps: int - maximum trajectory length to consider (skip if longer)
        capture_interval: int - how often to capture weights, evaluate policy, and log progress
        eval_episodes: int - number of episodes to run for policy evaluation
        verbose: bool - whether to print progress
        training_mode: str - 'random_sampling' (default) or 'fixed_pair'
        fixed_source: int or None - source literal for fixed pair training (required if training_mode='fixed_pair')
        fixed_target: int or None - target literal for fixed pair training (required if training_mode='fixed_pair')
        entropy_threshold: float or None - center point for entropy-based mixing (default 1.5)
        oracle_sensitivity: float - controls steepness of sigmoid transition (default 10.0)
        superlinearity: float - exponent for lambda statistic computation (default 2.0)

    Returns:
        dict - training statistics (policy success_rate, avg_episode_length, policy_loss)
    """
    # Get all valid (source, target) pairs from next_steps
    next_steps = gen_impgraph.compute_next_steps(graph, num_vars)
    valid_pairs = list(next_steps.keys())

    print(f"Found {len(valid_pairs)} valid (source, target) pairs in graph")

    if len(valid_pairs) == 0:
        raise ValueError("No valid (source, target) pairs found in graph")

    # Validate training mode and fixed pair arguments
    if training_mode == 'fixed_pair':
        if fixed_source is None or fixed_target is None:
            raise ValueError("Both fixed_source and fixed_target must be provided when training_mode='fixed_pair'")

        # Check if the fixed pair exists in valid pairs
        if (fixed_source, fixed_target) not in valid_pairs:
            raise ValueError(f"Fixed pair ({fixed_source}, {fixed_target}) is not a valid (source, target) pair in the graph")

        # Verify there's a valid trajectory for the fixed pair
        fixed_trajectory = generate_trajectory_from_graph(fixed_source, fixed_target, graph)
        if fixed_trajectory is None or len(fixed_trajectory) > max_steps:
            raise ValueError(f"No valid trajectory found for fixed pair ({fixed_source}, {fixed_target}) or trajectory exceeds max_steps")

        print(f"Training on fixed pair: ({fixed_source}, {fixed_target}) with trajectory length {len(fixed_trajectory)}")
    elif training_mode == 'random_sampling':
        print(f"Training on random sampling from all valid pairs")
    else:
        raise ValueError(f"Invalid training_mode: {training_mode}. Must be 'random_sampling' or 'fixed_pair'")

    # Tracking metrics for policy evaluation
    policy_success_rates = []
    policy_avg_lengths = []
    policy_avg_entropies = []
    policy_chunkabilities = []
    policy_oracle_call_probs = []
    policy_lambdas = []
    captured_episodes = []
    policy_losses = []

    # Training loop
    for episode in range(num_episodes):
        # Select (source, target) pair based on training mode
        if training_mode == 'fixed_pair':
            # Use the fixed pair for all episodes
            source_literal = fixed_source
            target_literal = fixed_target
            trajectory = fixed_trajectory  # Already validated above
        else:  # random_sampling
            # Sample random (source, target) pair until we get a valid trajectory
            trajectory = None
            while trajectory is None or len(trajectory) > max_steps:
                source_literal, target_literal = random.choice(valid_pairs)
                trajectory = generate_trajectory_from_graph(source_literal, target_literal, graph)

        # Train policy network on optimal trajectory (teacher forcing)
        train_policy(policy_net, trajectory, target_literal, num_vars)

        # Capture metrics and log at intervals
        if (episode + 1) % capture_interval == 0:
            captured_episodes.append(episode + 1)

            # Evaluate policy (use fixed pair if specified)
            eval_successes = 0
            eval_lengths = []
            eval_entropies = []
            eval_chunkabilities = []
            eval_confidences = []

            if training_mode == 'fixed_pair':
                # Evaluate only on the fixed pair
                for _ in range(eval_episodes):
                    eval_trajectory, eval_reached_goal, eval_entropy, eval_chunkability, eval_confidence = run_episode(
                        policy_net, fixed_source, fixed_target, graph, num_vars, max_steps,
                        entropy_threshold, oracle_sensitivity
                    )

                    # print(f"trajectory: {eval_trajectory} | reached_goal: {eval_reached_goal} | entropy: {eval_entropy:.4f}")

                    if eval_reached_goal:
                        eval_successes += 1
                    eval_lengths.append(len(eval_trajectory))
                    eval_entropies.append(eval_entropy)
                    eval_chunkabilities.append(eval_chunkability)
                    eval_confidences.append(eval_confidence)
            else:
                # Evaluate on random episodes
                for _ in range(eval_episodes):
                    eval_source, eval_target = random.choice(valid_pairs)
                    eval_trajectory, eval_reached_goal, eval_entropy, eval_chunkability, eval_confidence = run_episode(
                        policy_net, eval_source, eval_target, graph, num_vars, max_steps,
                        entropy_threshold, oracle_sensitivity
                    )

                    if eval_reached_goal:
                        eval_successes += 1
                    eval_lengths.append(len(eval_trajectory))
                    eval_entropies.append(eval_entropy)
                    eval_chunkabilities.append(eval_chunkability)
                    eval_confidences.append(eval_confidence)

            # Record policy evaluation metrics
            policy_success_rate = eval_successes / eval_episodes
            policy_success_rates.append(policy_success_rate)

            policy_avg_length = np.mean(eval_lengths)
            policy_avg_lengths.append(policy_avg_length)

            policy_avg_entropy = np.mean(eval_entropies)
            policy_avg_entropies.append(policy_avg_entropy)

            policy_avg_chunkability = np.mean(eval_chunkabilities)
            policy_chunkabilities.append(policy_avg_chunkability)

            policy_avg_oracle_call_prob = 1.0 - np.mean(eval_confidences)
            policy_oracle_call_probs.append(policy_avg_oracle_call_prob)

            # Compute lambda statistic: lambda = chunkability^superlinearity
            policy_lambda = policy_avg_chunkability ** superlinearity
            policy_lambdas.append(policy_lambda)

            # Capture network loss (use fixed pair if specified)
            if training_mode == 'fixed_pair':
                policy_loss = test_loss_policy(policy_net, graph, num_vars,
                                               fixed_source=fixed_source, fixed_target=fixed_target)
            else:
                policy_loss = test_loss_policy(policy_net, graph, num_vars)
            policy_losses.append(policy_loss)

            # Log progress
            if verbose:
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Success: {policy_success_rate:.2%} | "
                      f"Avg Len: {policy_avg_length:.1f} | "
                      f"Avg Entropy: {policy_avg_entropy:.4f} | "
                      f"Chunkability: {policy_avg_chunkability:.4f} | "
                      f"Oracle Call Prob: {policy_avg_oracle_call_prob:.4f} | "
                      f"Policy Loss: {policy_loss:.4f}")

    # Capture final weight matrices
    if isinstance(policy_net, PolicyNetworkPC):
        # For PolicyNetworkPC, extract from PyTorch model attributes
        source_to_hidden = policy_net.source_linear.weight.data.T.cpu().numpy()
        target_to_hidden = policy_net.target_linear.weight.data.T.cpu().numpy()
        hidden_to_output = policy_net.output_linear.weight.data.T.cpu().numpy()

        final_policy_matrices = {
            'source_to_hidden': source_to_hidden.copy(),
            'target_to_hidden': target_to_hidden.copy(),
            'hidden_to_output': hidden_to_output.copy()
        }
    else:
        # For PolicyNetwork, extract from PsyNeuLink
        final_policy_matrices = {
            'source_to_hidden': policy_net.source_to_hidden.matrix.base.copy(),
            'target_to_hidden': policy_net.target_to_hidden.matrix.base.copy(),
            'hidden_to_output': policy_net.hidden_to_output.matrix.base.copy()
        }

    # Compute final statistics
    stats = {
        'policy_success_rates': policy_success_rates,
        'policy_avg_lengths': policy_avg_lengths,
        'policy_avg_entropies': policy_avg_entropies,
        'policy_chunkabilities': policy_chunkabilities,
        'policy_oracle_call_probs': policy_oracle_call_probs,
        'policy_lambdas': policy_lambdas,
        'captured_episodes': captured_episodes,
        'policy_losses': policy_losses,
        'final_policy_matrices': final_policy_matrices,
        'entropy_threshold': entropy_threshold,
        'superlinearity': superlinearity
    }

    return stats


# ============================================================================
# Main Training Loop
# ============================================================================

def train_td_n(policy_net, value_net, graph, num_vars, num_episodes=1000, n_steps=3,
               max_steps=100, capture_interval=50, eval_episodes=10, verbose=True,
               training_mode='random_sampling', fixed_source=None, fixed_target=None,
               entropy_threshold=1.5, oracle_sensitivity=5.0, superlinearity=2.0):
    """
    Train policy and value networks using online TD(n).

    Args:
        policy_net: PolicyNetwork - policy network to train
        value_net: ValueNetwork - value network to train
        graph: nx.DiGraph - implication graph
        num_vars: int - number of variables
        num_episodes: int - number of training episodes
        n_steps: int - number of steps for TD(n) (1 = TD(0), higher = more Monte Carlo)
        max_steps: int - maximum trajectory length to consider (skip if longer)
        capture_interval: int - how often to capture weights, evaluate policy, and log progress
        eval_episodes: int - number of episodes to run for policy evaluation
        verbose: bool - whether to print progress
        training_mode: str - 'random_sampling' (default) or 'fixed_pair'
        fixed_source: int or None - source literal for fixed pair training (required if training_mode='fixed_pair')
        fixed_target: int or None - target literal for fixed pair training (required if training_mode='fixed_pair')
        entropy_threshold: float or None - center point for entropy-based mixing (default 1.5)
        oracle_sensitivity: float - controls steepness of sigmoid transition (default 5.0)
        superlinearity: float - exponent for lambda statistic computation (default 2.0)

    Returns:
        dict - training statistics (policy success_rate, avg_episode_length, etc.)
    """
    # Get all valid (source, target) pairs from next_steps
    next_steps = gen_impgraph.compute_next_steps(graph, num_vars)
    valid_pairs = list(next_steps.keys())

    print(f"Found {len(valid_pairs)} valid (source, target) pairs in graph")

    if len(valid_pairs) == 0:
        raise ValueError("No valid (source, target) pairs found in graph")

    # Validate training mode and fixed pair arguments
    if training_mode == 'fixed_pair':
        if fixed_source is None or fixed_target is None:
            raise ValueError("Both fixed_source and fixed_target must be provided when training_mode='fixed_pair'")

        # Check if the fixed pair exists in valid pairs
        if (fixed_source, fixed_target) not in valid_pairs:
            raise ValueError(f"Fixed pair ({fixed_source}, {fixed_target}) is not a valid (source, target) pair in the graph")

        # Verify there's a valid trajectory for the fixed pair
        fixed_trajectory = generate_trajectory_from_graph(fixed_source, fixed_target, graph)
        if fixed_trajectory is None or len(fixed_trajectory) > max_steps:
            raise ValueError(f"No valid trajectory found for fixed pair ({fixed_source}, {fixed_target}) or trajectory exceeds max_steps")

        print(f"Training on fixed pair: ({fixed_source}, {fixed_target}) with trajectory length {len(fixed_trajectory)}")
    elif training_mode == 'random_sampling':
        print(f"Training on random sampling from all valid pairs")
    else:
        raise ValueError(f"Invalid training_mode: {training_mode}. Must be 'random_sampling' or 'fixed_pair'")

    # Tracking metrics for policy evaluation
    policy_success_rates = []
    policy_avg_lengths = []
    policy_avg_entropies = []
    policy_chunkabilities = []
    policy_oracle_call_probs = []
    policy_lambdas = []
    captured_episodes = []
    policy_losses = []
    value_losses = []

    # Training loop
    for episode in range(num_episodes):
        # Select (source, target) pair based on training mode
        if training_mode == 'fixed_pair':
            # Use the fixed pair for all episodes
            source_literal = fixed_source
            target_literal = fixed_target
            trajectory = fixed_trajectory  # Already validated above
        else:  # random_sampling
            # Sample random (source, target) pair until we get a valid trajectory
            trajectory = None
            while trajectory is None or len(trajectory) > max_steps:
                source_literal, target_literal = random.choice(valid_pairs)
                trajectory = generate_trajectory_from_graph(source_literal, target_literal, graph)

        # Train both networks on trajectory
        train_policy_td_n(policy_net, value_net, trajectory, target_literal, n_steps, num_vars)
        train_value_td_n(value_net, trajectory, target_literal, n_steps, num_vars)

        # Capture metrics and log at intervals
        if (episode + 1) % capture_interval == 0:
            captured_episodes.append(episode + 1)

            # Evaluate policy (use fixed pair if specified)
            eval_successes = 0
            eval_lengths = []
            eval_entropies = []
            eval_chunkabilities = []
            eval_confidences = []

            if training_mode == 'fixed_pair':
                # Evaluate only on the fixed pair
                for _ in range(eval_episodes):
                    eval_trajectory, eval_reached_goal, eval_entropy, eval_chunkability, eval_confidence = run_episode(
                        policy_net, fixed_source, fixed_target, graph, num_vars, max_steps,
                        entropy_threshold, oracle_sensitivity
                    )

                    if eval_reached_goal:
                        eval_successes += 1
                    eval_lengths.append(len(eval_trajectory))
                    eval_entropies.append(eval_entropy)
                    eval_chunkabilities.append(eval_chunkability)
                    eval_confidences.append(eval_confidence)
            else:
                # Evaluate on random episodes
                for _ in range(eval_episodes):
                    eval_source, eval_target = random.choice(valid_pairs)
                    eval_trajectory, eval_reached_goal, eval_entropy, eval_chunkability, eval_confidence = run_episode(
                        policy_net, eval_source, eval_target, graph, num_vars, max_steps,
                        entropy_threshold, oracle_sensitivity
                    )

                    if eval_reached_goal:
                        eval_successes += 1
                    eval_lengths.append(len(eval_trajectory))
                    eval_entropies.append(eval_entropy)
                    eval_chunkabilities.append(eval_chunkability)
                    eval_confidences.append(eval_confidence)

            # Record policy evaluation metrics
            policy_success_rate = eval_successes / eval_episodes
            policy_success_rates.append(policy_success_rate)

            policy_avg_length = np.mean(eval_lengths)
            policy_avg_lengths.append(policy_avg_length)

            policy_avg_entropy = np.mean(eval_entropies)
            policy_avg_entropies.append(policy_avg_entropy)

            policy_avg_chunkability = np.mean(eval_chunkabilities)
            policy_chunkabilities.append(policy_avg_chunkability)

            policy_avg_oracle_call_prob = 1.0 - np.mean(eval_confidences)
            policy_oracle_call_probs.append(policy_avg_oracle_call_prob)

            # Compute lambda statistic: lambda = chunkability^superlinearity
            policy_lambda = policy_avg_chunkability ** superlinearity
            policy_lambdas.append(policy_lambda)

            # Capture network losses (use fixed pair if specified)
            if training_mode == 'fixed_pair':
                policy_loss = test_loss_policy(policy_net, graph, num_vars,
                                               fixed_source=fixed_source, fixed_target=fixed_target)
            else:
                policy_loss = test_loss_policy(policy_net, graph, num_vars)
            policy_losses.append(policy_loss)

            value_loss = test_loss_value(value_net, graph, num_vars)
            value_losses.append(value_loss)

            # Log progress
            if verbose:
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Success: {policy_success_rate:.2%} | "
                      f"Avg Len: {policy_avg_length:.1f} | "
                      f"Avg Entropy: {policy_avg_entropy:.4f} | "
                      f"Chunkability: {policy_avg_chunkability:.4f} | "
                      f"Oracle Call Prob: {policy_avg_oracle_call_prob:.4f} | "
                      f"Policy Loss: {policy_loss:.4f} | "
                      f"Value Loss: {value_loss:.4f}")

    # Capture final weight matrices
    if isinstance(policy_net, PolicyNetworkPC):
        # For PolicyNetworkPC, extract from PyTorch model attributes
        source_to_hidden = policy_net.source_linear.weight.data.T.cpu().numpy()
        target_to_hidden = policy_net.target_linear.weight.data.T.cpu().numpy()
        hidden_to_output = policy_net.output_linear.weight.data.T.cpu().numpy()

        final_policy_matrices = {
            'source_to_hidden': source_to_hidden.copy(),
            'target_to_hidden': target_to_hidden.copy(),
            'hidden_to_output': hidden_to_output.copy()
        }
    else:
        # For PolicyNetwork, extract from PsyNeuLink
        final_policy_matrices = {
            'source_to_hidden': policy_net.source_to_hidden.matrix.base.copy(),
            'target_to_hidden': policy_net.target_to_hidden.matrix.base.copy(),
            'hidden_to_output': policy_net.hidden_to_output.matrix.base.copy()
        }

    final_value_matrices = {
        'source_to_hidden': value_net.source_to_hidden.matrix.base.copy(),
        'target_to_hidden': value_net.target_to_hidden.matrix.base.copy(),
        'hidden_to_output': value_net.hidden_to_output.matrix.base.copy()
    }

    # Compute final statistics
    stats = {
        'policy_success_rates': policy_success_rates,
        'policy_avg_lengths': policy_avg_lengths,
        'policy_avg_entropies': policy_avg_entropies,
        'policy_chunkabilities': policy_chunkabilities,
        'policy_oracle_call_probs': policy_oracle_call_probs,
        'policy_lambdas': policy_lambdas,
        'captured_episodes': captured_episodes,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'final_policy_matrices': final_policy_matrices,
        'final_value_matrices': final_value_matrices,
        'entropy_threshold': entropy_threshold,
        'superlinearity': superlinearity
    }

    return stats


# ============================================================================
# Results Saving and Plotting
# ============================================================================

def save_results(stats, experiment_name, save_dir='results/td_n'):
    """
    Save training statistics and final weight matrices to disk.

    Args:
        stats: dict - statistics returned from train_td_n
        experiment_name: str - name for this experiment (e.g., 'impgraph_10v_15c_n3')
        save_dir: str - directory to save results
    """
    # Create directories
    os.makedirs(f'{save_dir}/metrics', exist_ok=True)
    os.makedirs(f'{save_dir}/final_weights/{experiment_name}', exist_ok=True)
    os.makedirs(f'{save_dir}/plots', exist_ok=True)

    # Save policy success rates
    with open(f'{save_dir}/metrics/{experiment_name}_policy_success_rates.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'policy_success_rate'])
        for episode, rate in zip(stats['captured_episodes'], stats['policy_success_rates']):
            writer.writerow([episode, rate])

    # Save policy average episode lengths
    with open(f'{save_dir}/metrics/{experiment_name}_policy_avg_lengths.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'policy_avg_length'])
        for episode, length in zip(stats['captured_episodes'], stats['policy_avg_lengths']):
            writer.writerow([episode, length])

    # Save policy average entropies
    with open(f'{save_dir}/metrics/{experiment_name}_policy_avg_entropies.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'policy_avg_entropy'])
        for episode, entropy in zip(stats['captured_episodes'], stats['policy_avg_entropies']):
            writer.writerow([episode, entropy])

    # Save policy chunkabilities
    with open(f'{save_dir}/metrics/{experiment_name}_policy_chunkabilities.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'policy_chunkability'])
        for episode, chunkability in zip(stats['captured_episodes'], stats['policy_chunkabilities']):
            writer.writerow([episode, chunkability])

    # Save policy oracle call probabilities
    with open(f'{save_dir}/metrics/{experiment_name}_policy_oracle_call_probs.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'policy_oracle_call_prob'])
        for episode, oracle_call_prob in zip(stats['captured_episodes'], stats['policy_oracle_call_probs']):
            writer.writerow([episode, oracle_call_prob])

    # Save policy lambdas
    with open(f'{save_dir}/metrics/{experiment_name}_policy_lambdas.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'policy_lambda'])
        for episode, lambda_val in zip(stats['captured_episodes'], stats['policy_lambdas']):
            writer.writerow([episode, lambda_val])

    # Save policy losses
    with open(f'{save_dir}/metrics/{experiment_name}_policy_losses.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'loss'])
        for episode, loss in zip(stats['captured_episodes'], stats['policy_losses']):
            writer.writerow([episode, loss])

    # Save value losses (only if present, i.e., TD(n) mode)
    if 'value_losses' in stats:
        with open(f'{save_dir}/metrics/{experiment_name}_value_losses.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'loss'])
            for episode, loss in zip(stats['captured_episodes'], stats['value_losses']):
                writer.writerow([episode, loss])

    # Save final policy weight matrices (no target_to_hidden for fixed-goal policy)
    np.save(f'{save_dir}/final_weights/{experiment_name}/policy_source_to_hidden.npy',
            stats['final_policy_matrices']['source_to_hidden'])
    np.save(f'{save_dir}/final_weights/{experiment_name}/policy_target_to_hidden.npy',
            stats['final_policy_matrices']['target_to_hidden'])
    np.save(f'{save_dir}/final_weights/{experiment_name}/policy_hidden_to_output.npy',
            stats['final_policy_matrices']['hidden_to_output'])

    # Save final value weight matrices (only if present, i.e., TD(n) mode)
    if 'final_value_matrices' in stats:
        np.save(f'{save_dir}/final_weights/{experiment_name}/value_source_to_hidden.npy',
                stats['final_value_matrices']['source_to_hidden'])
        np.save(f'{save_dir}/final_weights/{experiment_name}/value_target_to_hidden.npy',
                stats['final_value_matrices']['target_to_hidden'])
        np.save(f'{save_dir}/final_weights/{experiment_name}/value_hidden_to_output.npy',
                stats['final_value_matrices']['hidden_to_output'])

    print(f"\nResults saved to {save_dir}/")
    print(f"  - Metrics: {save_dir}/metrics/{experiment_name}_*.csv")
    print(f"  - Final weights: {save_dir}/final_weights/{experiment_name}/")


def plot_training_curves(stats, experiment_name, save_dir='results/td_n', show=False):
    """
    Plot and save training curves (policy success rate, policy episode length, losses, chunkability, oracle_call_prob).

    Args:
        stats: dict - statistics returned from train_td_n or train_teacher_forcing
        experiment_name: str - name for this experiment
        save_dir: str - directory to save plots
        show: bool - whether to display plots (default False, just save)
    """
    # Determine layout based on whether we have value network data
    has_value_loss = 'value_losses' in stats

    if has_value_loss:
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    else:
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    # Assign axes based on layout
    if has_value_loss:
        ax_success = axes[0, 0]
        ax_length = axes[1, 0]
        ax_entropy = axes[0, 1]
        ax_policy_loss = axes[1, 1]
        ax_chunkability = axes[0, 2]
        ax_oracle_call_prob = axes[1, 2]
        ax_value_loss = axes[2, 1]
        ax_lambda = axes[2, 2]
        # Hide unused subplot
        axes[2, 0].axis('off')
    else:
        ax_success = axes[0, 0]
        ax_length = axes[1, 0]
        ax_entropy = axes[0, 1]
        ax_policy_loss = axes[1, 1]
        ax_chunkability = axes[0, 2]
        ax_oracle_call_prob = axes[1, 2]
        ax_lambda = axes[2, 2]
        # Hide unused subplots for teacher forcing mode
        axes[2, 0].axis('off')
        axes[2, 1].axis('off')

    # Policy success rate
    ax_success.plot(stats['captured_episodes'], stats['policy_success_rates'], 'b-', linewidth=2)
    ax_success.set_xlabel('Training Episode')
    ax_success.set_ylabel('Policy Success Rate')
    ax_success.set_title('Policy Success Rate over Training')
    ax_success.grid(True, alpha=0.3)
    ax_success.set_ylim([0, 1.05])

    # Policy average episode length
    ax_length.plot(stats['captured_episodes'], stats['policy_avg_lengths'], 'g-', linewidth=2)
    ax_length.set_xlabel('Training Episode')
    ax_length.set_ylabel('Policy Average Episode Length')
    ax_length.set_title('Policy Average Episode Length over Training')
    ax_length.grid(True, alpha=0.3)

    # Policy average entropy
    ax_entropy.plot(stats['captured_episodes'], stats['policy_avg_entropies'], 'orange', linewidth=2)

    # Add dashed line for entropy threshold if it exists
    if 'entropy_threshold' in stats and stats['entropy_threshold'] is not None:
        ax_entropy.axhline(y=stats['entropy_threshold'], color='red', linestyle='--', linewidth=2, label=f'Threshold = {stats["entropy_threshold"]:.2f}')
        ax_entropy.legend()

    ax_entropy.set_xlabel('Training Episode')
    ax_entropy.set_ylabel('Policy Average Entropy')
    ax_entropy.set_title('Policy Average Entropy over Training')
    ax_entropy.grid(True, alpha=0.3)

    # Policy loss
    ax_policy_loss.plot(stats['captured_episodes'], stats['policy_losses'], 'r-', linewidth=2)
    ax_policy_loss.set_xlabel('Training Episode')
    ax_policy_loss.set_ylabel('Policy Loss (Cross-Entropy)')
    ax_policy_loss.set_title('Policy Network Loss over Training')
    ax_policy_loss.grid(True, alpha=0.3)

    # Chunkability
    ax_chunkability.plot(stats['captured_episodes'], stats['policy_chunkabilities'], 'purple', linewidth=2)
    ax_chunkability.set_xlabel('Training Episode')
    ax_chunkability.set_ylabel('Chunkability (Corridor-likeness)')
    ax_chunkability.set_title('Policy Chunkability over Training')
    ax_chunkability.grid(True, alpha=0.3)
    ax_chunkability.set_ylim([0, 1.05])

    # Oracle Call Probability (1 - policy weight)
    ax_oracle_call_prob.plot(stats['captured_episodes'], stats['policy_oracle_call_probs'], 'brown', linewidth=2)
    ax_oracle_call_prob.set_xlabel('Training Episode')
    ax_oracle_call_prob.set_ylabel('Oracle Call Probability')
    ax_oracle_call_prob.set_title('Oracle Call Probability over Training')
    ax_oracle_call_prob.grid(True, alpha=0.3)
    ax_oracle_call_prob.set_ylim([0, 1.05])

    # Value loss (only if present)
    if has_value_loss:
        ax_value_loss.plot(stats['captured_episodes'], stats['value_losses'], 'm-', linewidth=2)
        ax_value_loss.set_xlabel('Training Episode')
        ax_value_loss.set_ylabel('Value Loss (MSE)')
        ax_value_loss.set_title('Value Network Loss over Training')
        ax_value_loss.grid(True, alpha=0.3)

    # Lambda statistic (chunkability^superlinearity)
    superlinearity = stats.get('superlinearity', 2.0)
    ax_lambda.plot(stats['captured_episodes'], stats['policy_lambdas'], 'teal', linewidth=2)
    ax_lambda.set_xlabel('Training Episode')
    ax_lambda.set_ylabel(f'Lambda (Chunkability^{superlinearity:.1f})')
    ax_lambda.set_title(f'Lambda Statistic over Training')
    ax_lambda.grid(True, alpha=0.3)
    ax_lambda.set_ylim([0, 1.05])

    plt.tight_layout()

    # Save plot
    os.makedirs(f'{save_dir}/plots', exist_ok=True)
    plot_path = f'{save_dir}/plots/{experiment_name}_training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  - Training curves plot: {plot_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # ========================================================================
    # Configuration
    # ========================================================================

    # Choose training mode: 'teacher_forcing' or 'td_n'
    training_mode = 'teacher_forcing'

    # Implication graph parameters
    num_vars = 16
    num_clauses = 25

    # Option 1: Load an existing graph
    # Uncomment the following lines to load a saved graph:
    graph_path = f'results/graphs/impgraph_{num_vars}v_{num_clauses}c_{training_mode}/graph.pkl'
    graph = load_graph(graph_path)
    print(f"Loaded graph from {graph_path}")

    # Option 2: Generate a new graph
    # graph = gen_impgraph.generate_implication_graph(num_vars, num_clauses)
    # print(f"Generated new graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Network hyperparameters
    hidden_size = 25
    policy_learning_rate = 0.32
    value_learning_rate = 0.5

    # Number of training episodes
    num_episodes = 3000

    # ========================================================================
    # Teacher Forcing Mode
    # ========================================================================
    if training_mode == 'teacher_forcing':
        # Create experiment name
        experiment_name = f'impgraph_{num_vars}v_{num_clauses}c_teacher_forcing'

        # Save graph for reference
        # os.makedirs(f'results/graphs/{experiment_name}', exist_ok=True)
        # with open(f'results/graphs/{experiment_name}/graph.pkl', 'wb') as f:
        #     pickle.dump(graph, f)
        # print(f"Saved graph to results/graphs/{experiment_name}/graph.pkl")
        # gen_impgraph.visualize_graph(graph)
        # plt.savefig(f'results/graphs/{experiment_name}/graph_visualization.png', dpi=300, bbox_inches='tight')

        # Option: Load pre-trained weights
        # policy_weights_dir = 'results/off_policy/learned_matrices/impgraph_8v_10c'
        # policy_weights = load_weight_matrices(policy_weights_dir, network_type='policy', epoch=450)
        policy_weights = None  # Set to None for fresh start

        # Create policy network (no value network needed for teacher forcing)
        policy_net = PolicyNetworkPC(
            num_vars=num_vars,
            graph=graph,
            policy_name='TeacherForcing_Policy',
            hidden_size=hidden_size,
            learning_rate=policy_learning_rate,
            source_to_hidden_matrix=policy_weights[0] if policy_weights else None,
            target_to_hidden_matrix=policy_weights[1] if policy_weights else None,
            hidden_to_output_matrix=policy_weights[2] if policy_weights else None
        )

        print(f"Training with Teacher Forcing (Episodic)")
        print(f"Graph: {num_vars} vars, {num_clauses} clauses")
        print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        print()

        # Train using teacher forcing
        stats = train_teacher_forcing(
            policy_net=policy_net,
            graph=graph,
            num_vars=num_vars,
            num_episodes=num_episodes,
            max_steps=5,
            capture_interval=10,
            eval_episodes=50,
            training_mode="fixed_pair",
            fixed_source=15,
            fixed_target=14,
            entropy_threshold=0.5,
            superlinearity=2.5,
            verbose=True
        )

        print("\n" + "="*60)
        print("Training Complete!")
        if len(stats['policy_success_rates']) > 0:
            print(f"Final Policy Success Rate: {stats['policy_success_rates'][-1]:.2%}")
            print(f"Final Policy Avg Episode Length: {stats['policy_avg_lengths'][-1]:.2f}")
        print(f"Training Episodes: {num_episodes}")
        print("="*60)

        # Save results
        save_results(stats, experiment_name)

        # Plot training curves
        plot_training_curves(stats, experiment_name)

    # ========================================================================
    # TD(n) Mode
    # ========================================================================
    elif training_mode == 'td_n':
        # TD(n) hyperparameters
        n_steps = 1  # Number of steps for TD(n): 1=TD(0), higher=more Monte Carlo

        # Create experiment name
        experiment_name = f'impgraph_{num_vars}v_{num_clauses}c_n{n_steps}'

        # Save graph for reference
        # os.makedirs(f'results/graphs/{experiment_name}', exist_ok=True)
        # with open(f'results/graphs/{experiment_name}/graph.pkl', 'wb') as f:
        #     pickle.dump(graph, f)
        # print(f"Saved graph to results/graphs/{experiment_name}/graph.pkl")
        # gen_impgraph.visualize_graph(graph)
        # plt.savefig(f'results/graphs/{experiment_name}/graph_visualization.png', dpi=300, bbox_inches='tight')

        # Option: Load pre-trained weights (e.g., from mlp_basic_learner.py)
        # Uncomment to load weights:
        # policy_weights_dir = 'results/off_policy/learned_matrices/impgraph_8v_10c'
        # policy_weights = load_weight_matrices(policy_weights_dir, network_type='policy', epoch=450)
        # value_weights_dir = 'results/off_policy/learned_matrices/impgraph_8v_10c'
        # value_weights = load_weight_matrices(value_weights_dir, network_type='value')
        policy_weights = None  # Set to None for fresh start
        value_weights = None  # Set to None if not loading pretrained weights

        # Create networks with optional pretrained weights
        policy_net = PolicyNetwork(
            num_vars=num_vars,
            graph=graph,
            policy_name='TDN_Policy',
            hidden_size=hidden_size,
            learning_rate=policy_learning_rate,
            source_to_hidden_matrix=policy_weights[0] if policy_weights else None,
            target_to_hidden_matrix=policy_weights[1] if policy_weights else None,
            hidden_to_output_matrix=policy_weights[2] if policy_weights else None
        )

        value_net = ValueNetwork(
            num_vars=num_vars,
            graph=graph,
            value_name='TDN_Value',
            hidden_size=hidden_size,
            learning_rate=value_learning_rate,
            source_to_hidden_matrix=value_weights[0] if value_weights else None,
            target_to_hidden_matrix=value_weights[1] if value_weights else None,
            hidden_to_output_matrix=value_weights[2] if value_weights else None
        )

        print(f"Training TD(n) with n={n_steps}")
        print(f"Graph: {num_vars} vars, {num_clauses} clauses")
        print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        print()

        # Train
        stats = train_td_n(
            policy_net=policy_net,
            value_net=value_net,
            graph=graph,
            num_vars=num_vars,
            num_episodes=num_episodes,
            n_steps=n_steps,
            max_steps=50,
            capture_interval=10,
            verbose=True,
            training_mode='fixed_pair',
            fixed_source=-2,  # uncomment for fixed_pair mode
            fixed_target=4    # uncomment for fixed_pair mode
        )

        print("\n" + "="*60)
        print("Training Complete!")
        if len(stats['policy_success_rates']) > 0:
            print(f"Final Policy Success Rate: {stats['policy_success_rates'][-1]:.2%}")
            print(f"Final Policy Avg Episode Length: {stats['policy_avg_lengths'][-1]:.2f}")
        print(f"Training Episodes: {num_episodes}")
        print("="*60)

        # Save results
        save_results(stats, experiment_name)

        # Plot training curves
        plot_training_curves(stats, experiment_name)
