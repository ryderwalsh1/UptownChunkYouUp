import numpy as np
import random
import warnings
import gen_impgraph
from lambda_labels import lambda_labels
from policy_network import PolicyNetwork
from value_network import ValueNetwork
import matplotlib.pyplot as plt
import csv
import os

warnings.filterwarnings("ignore", category=UserWarning, module='psyneulink')


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

def train_policy_on_trajectory(policy_net, trajectory, target_literal, lambda_decay, num_vars, pad_id=0):
    """
    Train policy network on a completed trajectory using lambda-weighted soft labels.

    Args:
        policy_net: PolicyNetwork - policy network to train
        trajectory: list[int] - sequence of literals visited
        target_literal: int - goal literal
        lambda_decay: float - lambda parameter for TD(lambda)
        num_vars: int - number of variables
        pad_id: int - padding token ID (default 0, will map to literal 1)
    """
    # Convert trajectory to token indices
    token_trajectory = [literal_to_token_idx(lit, num_vars) for lit in trajectory]

    # Compute lambda labels
    vocab_size = 2 * num_vars
    soft_labels = lambda_labels(token_trajectory, vocab_size, lambda_=lambda_decay, pad_id=pad_id)

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
                                value_method='goal_prob', pad_id=0):
    """
    Train value network on a completed trajectory using lambda-weighted soft labels.

    Args:
        value_net: ValueNetwork - value network to train
        trajectory: list[int] - sequence of literals visited
        target_literal: int - goal literal
        lambda_decay: float - lambda parameter for TD(lambda)
        num_vars: int - number of variables
        value_method: str - 'goal_prob' or 'entropy' for extracting scalar value from distribution
        pad_id: int - padding token ID
    """
    # Convert trajectory to token indices
    token_trajectory = [literal_to_token_idx(lit, num_vars) for lit in trajectory]

    # Compute lambda labels
    vocab_size = 2 * num_vars
    soft_labels = lambda_labels(token_trajectory, vocab_size, lambda_=lambda_decay, pad_id=pad_id)

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
                     value_method='goal_prob', max_steps=100, log_interval=10, verbose=True):
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

    # Compute final statistics
    stats = {
        'episode_lengths': episode_lengths,
        'success_rate': success_count / total_episodes,
        'avg_episode_length': np.mean(episode_lengths),
        'total_episodes': total_episodes,
        'successful_episodes': success_count
    }

    return stats


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Implication graph parameters
    num_vars = 10
    num_clauses = 15

    # Generate the implication graph
    graph = gen_impgraph.generate_implication_graph(num_vars, num_clauses)

    # Network hyperparameters
    hidden_size = 20
    learning_rate = 0.5

    # TD(lambda) hyperparameters
    lambda_decay = 0.9
    value_method = 'goal_prob'  # or 'entropy'

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
        num_episodes=500,
        lambda_decay=lambda_decay,
        value_method=value_method,
        max_steps=50,
        log_interval=10,
        verbose=True
    )

    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Final Success Rate: {stats['success_rate']:.2%}")
    print(f"Average Episode Length: {stats['avg_episode_length']:.2f}")
    print(f"Total Episodes: {stats['total_episodes']}")
    print(f"Successful Episodes: {stats['successful_episodes']}")
    print("="*60)
