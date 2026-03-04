import numpy as np
import random
import warnings
import pickle
import gen_impgraph
import networkx as nx
from policy_network import PolicyNetwork
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


def load_weights_into_network(network, weights_dir, network_type='policy', epoch=None):
    """
    Load pre-trained weights into a PolicyNetwork or ValueNetwork.

    Args:
        network: PolicyNetwork or ValueNetwork - the network to load weights into
        weights_dir: str - directory containing the weight .npy files
        network_type: str - 'policy' or 'value' (used for naming)
        epoch: int or None - specific epoch to load (None = latest epoch)

    Expected files in weights_dir:
        - source_to_hidden/epoch_{N}.npy (or policy_source_to_hidden.npy)
        - target_to_hidden/epoch_{N}.npy (or policy_target_to_hidden.npy)
        - hidden_to_output/epoch_{N}.npy (or policy_hidden_to_output.npy)
    """
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

    # Load weights into the network
    network.source_to_hidden.matrix.base[:] = source_matrix
    network.target_to_hidden.matrix.base[:] = target_matrix
    network.hidden_to_output.matrix.base[:] = output_matrix

    print(f"Loaded weights into {network_type} network from {weights_dir}")


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

def test_loss_policy(policy_net, graph, num_vars, num_samples=50):
    """
    Evaluate policy network accuracy on optimal trajectories.

    Samples random (source, target) pairs, generates optimal trajectories,
    and checks if policy's argmax action matches the optimal next literal
    at each step (after action masking).

    Args:
        policy_net: PolicyNetwork - trained policy network
        graph: nx.DiGraph - implication graph
        num_vars: int - number of variables
        num_samples: int - number of (source, target) pairs to sample

    Returns:
        float - accuracy (correct predictions / total steps), higher is better
    """
    # Get all valid (source, target) pairs
    next_steps = gen_impgraph.compute_next_steps(graph, num_vars)
    valid_pairs = list(next_steps.keys())

    if len(valid_pairs) == 0:
        return 0.0

    total_steps = 0
    correct_predictions = 0

    # Sample and evaluate
    for sample in range(num_samples):
        source_literal, target_literal = random.choice(valid_pairs)

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
            adjacent_literals = get_adjacent_nodes(current_literal, graph)
            if len(adjacent_literals) == 0:
                continue

            # Create source encoding
            current_idx = literal_to_token_idx(current_literal, num_vars)
            source_encoding = np.zeros(2 * num_vars)
            source_encoding[current_idx] = 1.0

            # Get policy prediction
            action_dist = policy_net.predict(source_encoding, target_encoding).flatten()

            # Apply action masking
            masked_dist = mask_action_distribution(action_dist, adjacent_literals, num_vars)

            # Get argmax action
            predicted_idx = np.argmax(masked_dist)
            predicted_literal = token_idx_to_literal(predicted_idx, num_vars)

            # Print predictions and targets
            if sample == 0:  # Only print for the first sample to avoid clutter
                print(f"  Step {t}: current={current_literal}, optimal_next={optimal_next_literal}")
                print(f"    Network prediction (masked): {masked_dist}")
                print(f"    Predicted literal: {predicted_literal}")
                print(f"    Match: {predicted_literal == optimal_next_literal}")

            # Check if prediction matches optimal action
            total_steps += 1
            if predicted_literal == optimal_next_literal:
                correct_predictions += 1

    if total_steps == 0:
        return 0.0

    return correct_predictions / total_steps


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

def run_episode(policy_net, source_literal, target_literal, graph, num_vars, max_steps=100):
    """
    Execute one episode from source to target using the policy network with action masking.

    Args:
        policy_net: PolicyNetwork - trained policy network
        source_literal: int - starting literal
        target_literal: int - goal literal
        graph: nx.DiGraph - implication graph
        num_vars: int - number of variables
        max_steps: int - maximum steps before termination

    Returns:
        tuple: (trajectory, reached_goal)
            trajectory: list[int] - sequence of literals visited
            reached_goal: bool - whether target was reached
    """
    current_literal = source_literal
    trajectory = [source_literal]

    target_idx = literal_to_token_idx(target_literal, num_vars)

    for step in range(max_steps):
        # Check if we've reached the goal
        if current_literal == target_literal:
            return trajectory, True

        # Get valid next nodes (adjacent in graph)
        adjacent_literals = get_adjacent_nodes(current_literal, graph)

        # If no valid moves, episode ends
        if len(adjacent_literals) == 0:
            return trajectory, False

        # Create one-hot encodings
        current_idx = literal_to_token_idx(current_literal, num_vars)
        source_encoding = np.zeros(2 * num_vars)
        source_encoding[current_idx] = 1.0

        target_encoding = np.zeros(2 * num_vars)
        target_encoding[target_idx] = 1.0

        # Get policy prediction
        action_dist = policy_net.predict(source_encoding, target_encoding).flatten()

        # Apply action masking
        masked_dist = mask_action_distribution(action_dist, adjacent_literals, num_vars)

        # Sample action from masked distribution
        next_idx = np.random.choice(len(masked_dist), p=masked_dist)
        next_literal = token_idx_to_literal(next_idx, num_vars)

        # Update state
        current_literal = next_literal
        trajectory.append(current_literal)

    # Max steps reached without reaching goal
    return trajectory, False


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
# Main Training Loop
# ============================================================================

def train_td_n(policy_net, value_net, graph, num_vars, num_episodes=1000, n_steps=3,
               max_steps=100, capture_interval=50, eval_episodes=10, verbose=True):
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

    Returns:
        dict - training statistics (policy success_rate, avg_episode_length, etc.)
    """
    # Get all valid (source, target) pairs from next_steps
    next_steps = gen_impgraph.compute_next_steps(graph, num_vars)
    valid_pairs = list(next_steps.keys())

    if len(valid_pairs) == 0:
        raise ValueError("No valid (source, target) pairs found in graph")

    # Tracking metrics for policy evaluation
    policy_success_rates = []
    policy_avg_lengths = []
    captured_episodes = []
    policy_losses = []
    value_losses = []

    # Training loop
    for episode in range(num_episodes):
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

            # Evaluate policy on random episodes
            eval_successes = 0
            eval_lengths = []

            for _ in range(eval_episodes):
                eval_source, eval_target = random.choice(valid_pairs)
                eval_trajectory, eval_reached_goal = run_episode(
                    policy_net, eval_source, eval_target, graph, num_vars, max_steps
                )

                if eval_reached_goal:
                    eval_successes += 1
                eval_lengths.append(len(eval_trajectory))

            # Record policy evaluation metrics
            policy_success_rate = eval_successes / eval_episodes
            policy_success_rates.append(policy_success_rate)

            policy_avg_length = np.mean(eval_lengths)
            policy_avg_lengths.append(policy_avg_length)

            # Capture network losses
            policy_loss = test_loss_policy(policy_net, graph, num_vars)
            policy_losses.append(policy_loss)

            value_loss = test_loss_value(value_net, graph, num_vars)
            value_losses.append(value_loss)

            # Log progress
            if verbose:
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Success: {policy_success_rate:.2%} | "
                      f"Avg Len: {policy_avg_length:.1f} | "
                      f"Policy Acc: {policy_loss:.4f} | "
                      f"Value Loss: {value_loss:.4f}")

    # Capture final weight matrices
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
        'captured_episodes': captured_episodes,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'final_policy_matrices': final_policy_matrices,
        'final_value_matrices': final_value_matrices
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

    # Save policy losses
    with open(f'{save_dir}/metrics/{experiment_name}_policy_losses.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'loss'])
        for episode, loss in zip(stats['captured_episodes'], stats['policy_losses']):
            writer.writerow([episode, loss])

    # Save value losses
    with open(f'{save_dir}/metrics/{experiment_name}_value_losses.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'loss'])
        for episode, loss in zip(stats['captured_episodes'], stats['value_losses']):
            writer.writerow([episode, loss])

    # Save final policy weight matrices
    np.save(f'{save_dir}/final_weights/{experiment_name}/policy_source_to_hidden.npy',
            stats['final_policy_matrices']['source_to_hidden'])
    np.save(f'{save_dir}/final_weights/{experiment_name}/policy_target_to_hidden.npy',
            stats['final_policy_matrices']['target_to_hidden'])
    np.save(f'{save_dir}/final_weights/{experiment_name}/policy_hidden_to_output.npy',
            stats['final_policy_matrices']['hidden_to_output'])

    # Save final value weight matrices
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
    Plot and save training curves (policy success rate, policy episode length, losses).

    Args:
        stats: dict - statistics returned from train_td_n
        experiment_name: str - name for this experiment
        save_dir: str - directory to save plots
        show: bool - whether to display plots (default False, just save)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Policy success rate
    axes[0, 0].plot(stats['captured_episodes'], stats['policy_success_rates'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Training Episode')
    axes[0, 0].set_ylabel('Policy Success Rate')
    axes[0, 0].set_title('Policy Success Rate over Training')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])

    # Policy average episode length
    axes[0, 1].plot(stats['captured_episodes'], stats['policy_avg_lengths'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Training Episode')
    axes[0, 1].set_ylabel('Policy Average Episode Length')
    axes[0, 1].set_title('Policy Average Episode Length over Training')
    axes[0, 1].grid(True, alpha=0.3)

    # Policy accuracy
    axes[1, 0].plot(stats['captured_episodes'], stats['policy_losses'], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Training Episode')
    axes[1, 0].set_ylabel('Policy Accuracy')
    axes[1, 0].set_title('Policy Network Accuracy over Training')
    axes[1, 0].grid(True, alpha=0.3)

    # Value loss
    axes[1, 1].plot(stats['captured_episodes'], stats['value_losses'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Training Episode')
    axes[1, 1].set_ylabel('Value Loss (MSE)')
    axes[1, 1].set_title('Value Network Loss over Training')
    axes[1, 1].grid(True, alpha=0.3)

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

    # Implication graph parameters
    num_vars = 8
    num_clauses = 16

    # Option 1: Load an existing graph
    # Uncomment the following lines to load a saved graph:
    # graph_path = 'results/graphs/impgraph_8v_10c/graph.pkl'
    # graph = load_graph(graph_path)
    # print(f"Loaded graph from {graph_path}")

    # Option 2: Generate a new graph
    graph = gen_impgraph.generate_implication_graph(num_vars, num_clauses)
    print(f"Generated new graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Network hyperparameters
    hidden_size = 25
    policy_learning_rate = 0.5
    value_learning_rate = 0.5


    # TD(n) hyperparameters
    n_steps = 1  # Number of steps for TD(n): 1=TD(0), higher=more Monte Carlo

    # Number of training episodes
    num_episodes = 500

    # Create experiment name
    experiment_name = f'impgraph_{num_vars}v_{num_clauses}c_n{n_steps}'

    # Save graph for reference
    os.makedirs(f'results/graphs/{experiment_name}', exist_ok=True)
    with open(f'results/graphs/{experiment_name}/graph.pkl', 'wb') as f:
        pickle.dump(graph, f)
    print(f"Saved graph to results/graphs/{experiment_name}/graph.pkl")
    gen_impgraph.visualize_graph(graph)
    plt.savefig(f'results/graphs/{experiment_name}/graph_visualization.png', dpi=300, bbox_inches='tight')

    # Create networks
    policy_net = PolicyNetwork(
        num_vars=num_vars,
        graph=graph,
        policy_name='TDN_Policy',
        hidden_size=hidden_size,
        learning_rate=policy_learning_rate
    )

    value_net = ValueNetwork(
        num_vars=num_vars,
        graph=graph,
        value_name='TDN_Value',
        hidden_size=hidden_size,
        learning_rate=value_learning_rate
    )

    # Option: Load pre-trained weights (e.g., from mlp_basic_learner.py)
    # Uncomment to load weights:
    # policy_weights_dir = 'results/off_policy/learned_matrices/impgraph_8v_10c'
    # load_weights_into_network(policy_net, policy_weights_dir, network_type='policy', epoch=495)
    # value_weights_dir = 'results/off_policy/learned_matrices/impgraph_8v_10c'
    # load_weights_into_network(value_net, value_weights_dir, network_type='value')

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
