import numpy as np
import random
import warnings
import pickle
import gen_impgraph
from lambda_labels import lambda_labels
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
# Value Target Extraction Functions
# ============================================================================

def extract_value_target_goal_prob(labels, goal_idx):
    """
    Extract value target as probability mass at goal node.

    Args:
        labels: np.array - soft label distribution (shape: vocab_size)
        goal_idx: int - index of goal literal in vocabulary

    Returns:
        float - probability at goal index
    """
    return labels[goal_idx]


def extract_value_target_entropy(labels):
    """
    Extract value target based on entropy of the distribution.
    Lower entropy = more certain = higher value (closer to goal).

    Args:
        labels: np.array - soft label distribution

    Returns:
        float - normalized inverse entropy (high when certain, low when uncertain)
    """
    # Avoid log(0) by filtering out zeros
    non_zero_mask = labels > 1e-10
    if not np.any(non_zero_mask):
        return 0.0

    # Calculate entropy
    entropy = -np.sum(labels[non_zero_mask] * np.log2(labels[non_zero_mask]))

    # Max entropy for a distribution over vocab_size elements is log2(vocab_size)
    max_entropy = np.log2(len(labels))

    # Normalize and invert: high certainty (low entropy) = high value
    normalized_inverse_entropy = 1.0 - (entropy / max_entropy)

    return normalized_inverse_entropy


# ============================================================================
# Episode Execution
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
# Training Functions
# ============================================================================

def train_policy_on_trajectory(policy_net, trajectory, target_literal, lambda_decay, num_vars):
    """
    Train policy network on a completed trajectory using lambda-weighted soft labels.

    Args:
        policy_net: PolicyNetwork - policy network to train
        trajectory: list[int] - sequence of literals visited
        target_literal: int - goal literal
        lambda_decay: float - lambda parameter for TD(lambda)
        num_vars: int - number of variables
    """
    # Convert trajectory to token indices
    token_trajectory = [literal_to_token_idx(lit, num_vars) for lit in trajectory]

    # Compute lambda labels
    vocab_size = 2 * num_vars
    soft_labels = lambda_labels(token_trajectory, vocab_size, lambda_=lambda_decay)

    # Target encoding (constant throughout trajectory)
    target_idx = literal_to_token_idx(target_literal, num_vars)
    target_encoding = np.zeros(vocab_size)
    target_encoding[target_idx] = 1.0

    # Apply updates for each timestep
    for t, literal in enumerate(trajectory):
        # Source encoding (current state)
        source_idx = literal_to_token_idx(literal, num_vars)
        source_encoding = np.zeros(vocab_size)
        source_encoding[source_idx] = 1.0

        # Soft target from lambda labels
        policy_target = soft_labels[t]

        # Single update
        policy_net.update_single(source_encoding, target_encoding, policy_target)


def train_value_on_trajectory(value_net, trajectory, target_literal, lambda_decay, num_vars,
                                value_method='goal_prob'):
    """
    Train value network on a completed trajectory using lambda-weighted soft labels.

    Args:
        value_net: ValueNetwork - value network to train
        trajectory: list[int] - sequence of literals visited
        target_literal: int - goal literal
        lambda_decay: float - lambda parameter for TD(lambda)
        num_vars: int - number of variables
        value_method: str - 'goal_prob' or 'entropy' for extracting scalar value from distribution
    """
    # Convert trajectory to token indices
    token_trajectory = [literal_to_token_idx(lit, num_vars) for lit in trajectory]

    # Compute lambda labels
    vocab_size = 2 * num_vars
    soft_labels = lambda_labels(token_trajectory, vocab_size, lambda_=lambda_decay)

    # Target encoding (constant throughout trajectory)
    target_idx = literal_to_token_idx(target_literal, num_vars)
    target_encoding = np.zeros(vocab_size)
    target_encoding[target_idx] = 1.0

    # Apply updates for each timestep
    for t, literal in enumerate(trajectory):
        # Source encoding (current state)
        source_idx = literal_to_token_idx(literal, num_vars)
        source_encoding = np.zeros(vocab_size)
        source_encoding[source_idx] = 1.0

        # Extract scalar value target from soft labels
        if value_method == 'goal_prob':
            value_target = extract_value_target_goal_prob(soft_labels[t], target_idx)
        elif value_method == 'entropy':
            value_target = extract_value_target_entropy(soft_labels[t])
        else:
            raise ValueError(f"Unknown value_method: {value_method}")

        # Single update
        value_net.update_single(source_encoding, target_encoding, value_target)


# ============================================================================
# Main Training Loop
# ============================================================================

def train_td_lambda(policy_net, value_net, graph, num_vars, num_episodes=1000, lambda_decay=0.9,
                     value_method='goal_prob', max_steps=100, log_interval=10, capture_interval=50, verbose=True):
    """
    Train policy and value networks using online TD(lambda) with soft labels.

    Args:
        policy_net: PolicyNetwork - policy network to train
        value_net: ValueNetwork - value network to train
        graph: nx.DiGraph - implication graph
        num_vars: int - number of variables
        num_episodes: int - number of training episodes
        lambda_decay: float - lambda parameter for TD(lambda) (0.0 to 1.0)
        value_method: str - 'goal_prob' or 'entropy'
        max_steps: int - maximum steps per episode
        log_interval: int - how often to log statistics
        capture_interval: int - how often to capture weights and compute test metrics
        verbose: bool - whether to print progress

    Returns:
        dict - training statistics (success_rate, avg_episode_length, etc.)
    """
    # Get all literals
    literals = [i for i in range(1, num_vars + 1)] + [-i for i in range(1, num_vars + 1)]

    # Get all valid (source, target) pairs from next_steps
    next_steps = gen_impgraph.compute_next_steps(graph, num_vars)
    valid_pairs = list(next_steps.keys())

    if len(valid_pairs) == 0:
        raise ValueError("No valid (source, target) pairs found in graph")

    # Tracking metrics
    episode_lengths = []
    success_rates = []
    avg_lengths = []
    captured_episodes = []
    policy_losses = []
    value_losses = []

    success_count = 0
    total_episodes = 0

    # Training loop
    for episode in range(num_episodes):
        # Sample random (source, target) pair
        source_literal, target_literal = random.choice(valid_pairs)

        # Run episode
        trajectory, reached_goal = run_episode(
            policy_net, source_literal, target_literal, graph, num_vars, max_steps
        )

        # Update metrics
        episode_lengths.append(len(trajectory))
        if reached_goal:
            success_count += 1
        total_episodes += 1

        # Train both networks on trajectory
        train_policy_on_trajectory(policy_net, trajectory, target_literal, lambda_decay, num_vars)
        train_value_on_trajectory(value_net, trajectory, target_literal, lambda_decay, num_vars, value_method)

        # Capture metrics at intervals
        if (episode + 1) % capture_interval == 0:
            captured_episodes.append(episode + 1)

            # Compute success rate over all episodes so far
            current_success_rate = success_count / total_episodes
            success_rates.append(current_success_rate)

            # Compute average length over recent episodes
            recent_window = min(capture_interval, len(episode_lengths))
            current_avg_length = np.mean(episode_lengths[-recent_window:])
            avg_lengths.append(current_avg_length)

            # Capture policy network loss
            policy_loss = policy_net.test_loss()
            policy_losses.append(policy_loss)

            # Capture value network loss
            value_loss = value_net.test_loss()
            value_losses.append(value_loss)

        # Log progress
        if verbose and (episode + 1) % log_interval == 0:
            recent_successes = success_count
            recent_episodes = total_episodes
            success_rate = recent_successes / recent_episodes if recent_episodes > 0 else 0.0
            avg_length = np.mean(episode_lengths[-log_interval:]) if len(episode_lengths) >= log_interval else np.mean(episode_lengths)

            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Success Rate: {success_rate:.2%} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Last: {len(trajectory)} steps, {'SUCCESS' if reached_goal else 'FAILED'}")

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
        'episode_lengths': episode_lengths,
        'success_rate': success_count / total_episodes,
        'avg_episode_length': np.mean(episode_lengths),
        'total_episodes': total_episodes,
        'successful_episodes': success_count,
        'captured_episodes': captured_episodes,
        'success_rates': success_rates,
        'avg_lengths': avg_lengths,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'final_policy_matrices': final_policy_matrices,
        'final_value_matrices': final_value_matrices
    }

    return stats


# ============================================================================
# Results Saving and Plotting
# ============================================================================

def save_results(stats, experiment_name, save_dir='results/on_policy'):
    """
    Save training statistics and final weight matrices to disk.

    Args:
        stats: dict - statistics returned from train_td_lambda
        experiment_name: str - name for this experiment (e.g., 'impgraph_10v_15c_lambda0.9')
        save_dir: str - directory to save results
    """
    # Create directories
    os.makedirs(f'{save_dir}/metrics', exist_ok=True)
    os.makedirs(f'{save_dir}/final_weights/{experiment_name}', exist_ok=True)
    os.makedirs(f'{save_dir}/plots', exist_ok=True)

    # Save success rates
    with open(f'{save_dir}/metrics/{experiment_name}_success_rates.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'success_rate'])
        for episode, rate in zip(stats['captured_episodes'], stats['success_rates']):
            writer.writerow([episode, rate])

    # Save average episode lengths
    with open(f'{save_dir}/metrics/{experiment_name}_avg_lengths.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'avg_length'])
        for episode, length in zip(stats['captured_episodes'], stats['avg_lengths']):
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

    # Save all episode lengths
    with open(f'{save_dir}/metrics/{experiment_name}_all_episode_lengths.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'length'])
        for episode, length in enumerate(stats['episode_lengths'], 1):
            writer.writerow([episode, length])

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


def plot_training_curves(stats, experiment_name, save_dir='results/on_policy', show=False):
    """
    Plot and save training curves (success rate, episode length, losses).

    Args:
        stats: dict - statistics returned from train_td_lambda
        experiment_name: str - name for this experiment
        save_dir: str - directory to save plots
        show: bool - whether to display plots (default False, just save)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Success rate
    axes[0, 0].plot(stats['captured_episodes'], stats['success_rates'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_title('Success Rate over Training')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])

    # Average episode length
    axes[0, 1].plot(stats['captured_episodes'], stats['avg_lengths'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Episode Length')
    axes[0, 1].set_title('Average Episode Length over Training')
    axes[0, 1].grid(True, alpha=0.3)

    # Policy loss
    axes[1, 0].plot(stats['captured_episodes'], stats['policy_losses'], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Policy Loss (MSE)')
    axes[1, 0].set_title('Policy Network Loss over Training')
    axes[1, 0].grid(True, alpha=0.3)

    # Value loss
    axes[1, 1].plot(stats['captured_episodes'], stats['value_losses'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
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
    num_clauses = 10

    # Option 1: Load an existing graph
    # Uncomment the following lines to load a saved graph:
    graph_path = 'results/graphs/impgraph_8v_10c/graph.pkl'
    graph = load_graph(graph_path)
    print(f"Loaded graph from {graph_path}")

    # Option 2: Generate a new graph
    # graph = gen_impgraph.generate_implication_graph(num_vars, num_clauses)
    # print(f"Generated new graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Network hyperparameters
    hidden_size = 25
    learning_rate = 0.5

    # TD(lambda) hyperparameters
    lambda_decay = 0.9
    value_method = 'goal_prob'  # or 'entropy'

    # Create experiment name
    experiment_name = f'impgraph_{num_vars}v_{num_clauses}c_lambda{lambda_decay}'

    # Create networks
    policy_net = PolicyNetwork(
        num_vars=num_vars,
        graph=graph,
        policy_name='TDLambda_Policy',
        hidden_size=hidden_size,
        learning_rate=learning_rate
    )

    value_net = ValueNetwork(
        num_vars=num_vars,
        graph=graph,
        value_name='TDLambda_Value',
        hidden_size=hidden_size,
        learning_rate=learning_rate
    )

    # Option: Load pre-trained weights (e.g., from mlp_basic_learner.py)
    # Uncomment to load weights:
    policy_weights_dir = 'results/off_policy/learned_matrices/impgraph_8v_10c'
    load_weights_into_network(policy_net, policy_weights_dir, network_type='policy')
    # value_weights_dir = 'results/off_policy/learned_matrices/impgraph_8v_10c'
    # load_weights_into_network(value_net, value_weights_dir, network_type='value')

    print(f"Training TD(λ) with λ={lambda_decay}, value_method={value_method}")
    print(f"Graph: {num_vars} vars, {num_clauses} clauses")
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    print()

    # Train
    stats = train_td_lambda(
        policy_net=policy_net,
        value_net=value_net,
        graph=graph,
        num_vars=num_vars,
        num_episodes=100,
        lambda_decay=lambda_decay,
        value_method=value_method,
        max_steps=50,
        log_interval=10,
        capture_interval=50,
        verbose=True
    )

    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Final Success Rate: {stats['success_rate']:.2%}")
    print(f"Average Episode Length: {stats['avg_episode_length']:.2f}")
    print(f"Total Episodes: {stats['total_episodes']}")
    print(f"Successful Episodes: {stats['successful_episodes']}")
    print("="*60)

    # Save results
    save_results(stats, experiment_name)

    # Plot training curves
    plot_training_curves(stats, experiment_name)
