"""
Evaluation Metrics for Lambda Experiment

Computes learning efficiency, decision quality, and policy performance metrics
from training trajectories and results.

Based on specification section 11: Core evaluation metrics
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats


def compute_auc(results_df, metric='success', x_axis='episode', per_condition=True):
    """
    Compute Area Under Curve for learning curves.

    This is the single best summary of learning speed + stability (§11.1c).

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results with columns [topology, lambda, seed, episode, metric]
    metric : str
        Column name to compute AUC over (default: 'success')
    x_axis : str
        X-axis column (default: 'episode')
    per_condition : bool
        If True, compute per (topology, lambda) averaged over seeds
        If False, compute per individual run

    Returns:
    --------
    auc_results : pd.DataFrame or float
        AUC values, either per condition or single value
    """
    if per_condition:
        # Group by topology, lambda, seed
        grouped = results_df.groupby(['topology', 'lambda', 'seed'])

        auc_values = []
        for (topology, lambda_val, seed), group in grouped:
            # Sort by x_axis
            group = group.sort_values(x_axis)
            x = group[x_axis].values
            y = group[metric].values

            # Trapezoidal integration
            auc = np.trapz(y, x)

            auc_values.append({
                'topology': topology,
                'lambda': lambda_val,
                'seed': seed,
                'auc': auc
            })

        return pd.DataFrame(auc_values)
    else:
        # Single AUC over entire dataframe
        results_sorted = results_df.sort_values(x_axis)
        x = results_sorted[x_axis].values
        y = results_sorted[metric].values
        return np.trapz(y, x)


def compute_episodes_to_threshold(results_df, thresholds=[0.5, 0.8, 0.95]):
    """
    Compute episodes needed to reach success rate thresholds.

    Uses rolling window to smooth success rate (§11.1a).

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results with columns [topology, lambda, seed, episode, success]
    thresholds : list of float
        Success rate thresholds (default: [0.5, 0.8, 0.95])

    Returns:
    --------
    threshold_results : pd.DataFrame
        Episodes to reach each threshold per condition
    """
    window_size = 20  # Rolling window for smoothing

    grouped = results_df.groupby(['topology', 'lambda', 'seed'])

    results = []
    for (topology, lambda_val, seed), group in grouped:
        # Sort by episode and reset index to avoid indexing issues
        group = group.sort_values('episode').reset_index(drop=True)

        # Compute rolling success rate
        success_rate = group['success'].rolling(window=window_size, min_periods=1).mean()

        # Find first episode reaching each threshold
        episodes_dict = {
            'topology': topology,
            'lambda': lambda_val,
            'seed': seed
        }

        for thresh in thresholds:
            thresh_name = f'episodes_to_{int(thresh*100)}pct'
            reached = success_rate >= thresh
            if reached.any():
                # Find first index where threshold is reached
                first_idx = reached.idxmax()
                episodes_dict[thresh_name] = group['episode'].iloc[first_idx]
            else:
                episodes_dict[thresh_name] = np.nan  # Never reached

        results.append(episodes_dict)

    return pd.DataFrame(results)


def compute_steps_to_threshold(results_df, thresholds=[0.5, 0.8, 0.95]):
    """
    Compute environment steps needed to reach success rate thresholds.

    Similar to episodes_to_threshold but counts environment steps (§11.1b).

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results with columns [topology, lambda, seed, episode, success, episode_length]
    thresholds : list of float
        Success rate thresholds

    Returns:
    --------
    threshold_results : pd.DataFrame
        Steps to reach each threshold per condition
    """
    window_size = 20

    grouped = results_df.groupby(['topology', 'lambda', 'seed'])

    results = []
    for (topology, lambda_val, seed), group in grouped:
        # Sort by episode and reset index to avoid indexing issues
        group = group.sort_values('episode').reset_index(drop=True)

        # Compute cumulative steps
        group['cumulative_steps'] = group['episode_length'].cumsum()

        # Compute rolling success rate
        success_rate = group['success'].rolling(window=window_size, min_periods=1).mean()

        steps_dict = {
            'topology': topology,
            'lambda': lambda_val,
            'seed': seed
        }

        for thresh in thresholds:
            thresh_name = f'steps_to_{int(thresh*100)}pct'
            reached = success_rate >= thresh
            if reached.any():
                # Find first index where threshold is reached
                first_idx = reached.idxmax()
                steps_dict[thresh_name] = group['cumulative_steps'].iloc[first_idx]
            else:
                steps_dict[thresh_name] = np.nan

        results.append(steps_dict)

    return pd.DataFrame(results)


def compute_timeout_rate(results_df):
    """
    Compute timeout rate: fraction of episodes hitting max_steps limit.

    From §11.2: Final policy quality metrics.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results with columns [topology, lambda, seed, success, episode_length, optimal_path_length]

    Returns:
    --------
    timeout_results : pd.DataFrame
        Timeout rate per condition
    """
    # Estimate max_steps from data: episodes with very long lengths relative to optimal
    # A timeout is when: not successful AND episode_length >> optimal_path_length

    grouped = results_df.groupby(['topology', 'lambda', 'seed'])

    results = []
    for (topology, lambda_val, seed), group in grouped:
        # Heuristic: timeout if failed and length > 3 * optimal_path_length
        timeouts = (~group['success']) & (group['episode_length'] > 3 * group['optimal_path_length'])

        timeout_rate = timeouts.mean()

        results.append({
            'topology': topology,
            'lambda': lambda_val,
            'seed': seed,
            'timeout_rate': timeout_rate
        })

    return pd.DataFrame(results)


def compute_junction_decision_accuracy(trajectory, graph, shortest_path):
    """
    Compute accuracy of decisions at junction nodes.

    From §11.3: Decision quality at junctions.

    Parameters:
    -----------
    trajectory : dict
        Trajectory with 'node_sequence' (list of visited nodes)
    graph : nx.Graph
        Maze graph
    shortest_path : list
        Shortest path from start to goal (node sequence)

    Returns:
    --------
    metrics : dict
        - num_junctions_visited: number of junction nodes visited
        - num_correct_junction_choices: number where action was on shortest path
        - junction_accuracy: fraction of correct choices
    """
    if 'node_sequence' not in trajectory:
        return {
            'num_junctions_visited': 0,
            'num_correct_junction_choices': 0,
            'junction_accuracy': np.nan
        }

    node_sequence = trajectory['node_sequence']
    shortest_path_set = set(shortest_path)

    num_junctions = 0
    num_correct = 0

    for i, node in enumerate(node_sequence[:-1]):  # Exclude last node (no action taken)
        # Check if junction (degree >= 3)
        if graph.degree(node) >= 3:
            num_junctions += 1

            # Check if next node is on shortest path
            next_node = node_sequence[i + 1]

            # Correct if next_node is on shortest path and follows from current node
            if next_node in shortest_path_set:
                # More refined: check if this edge is on THE shortest path
                # (not just any shortest path if multiple exist)
                if i < len(shortest_path) - 1:
                    # Are we on the shortest path at this point?
                    if node in shortest_path_set:
                        idx = shortest_path.index(node) if node in shortest_path else -1
                        if idx >= 0 and idx < len(shortest_path) - 1:
                            if next_node == shortest_path[idx + 1]:
                                num_correct += 1

    junction_accuracy = num_correct / num_junctions if num_junctions > 0 else np.nan

    return {
        'num_junctions_visited': num_junctions,
        'num_correct_junction_choices': num_correct,
        'junction_accuracy': junction_accuracy
    }


def compute_junction_action_entropy(trajectory, graph, action_probs):
    """
    Compute action entropy at junction nodes.

    From §11.3: Decision quality at junctions.

    Parameters:
    -----------
    trajectory : dict
        Trajectory with 'node_sequence'
    graph : nx.Graph
        Maze graph
    action_probs : list of np.ndarray
        Action probability distributions at each timestep [T, num_actions]

    Returns:
    --------
    metrics : dict
        - mean_junction_entropy: average entropy at junctions
        - mean_corridor_entropy: average entropy at corridors (for comparison)
    """
    if 'node_sequence' not in trajectory or len(action_probs) == 0:
        return {
            'mean_junction_entropy': np.nan,
            'mean_corridor_entropy': np.nan
        }

    node_sequence = trajectory['node_sequence']

    junction_entropies = []
    corridor_entropies = []

    for i, node in enumerate(node_sequence[:-1]):
        if i >= len(action_probs):
            break

        # Compute entropy: -Σ p log p
        probs = action_probs[i]
        # Avoid log(0)
        probs = probs + 1e-10
        entropy = -np.sum(probs * np.log(probs))

        # Classify node
        degree = graph.degree(node)
        if degree >= 3:
            junction_entropies.append(entropy)
        elif degree == 2:
            corridor_entropies.append(entropy)

    return {
        'mean_junction_entropy': np.mean(junction_entropies) if len(junction_entropies) > 0 else np.nan,
        'mean_corridor_entropy': np.mean(corridor_entropies) if len(corridor_entropies) > 0 else np.nan
    }


def compute_junction_policy_margin(trajectory, graph, action_probs):
    """
    Compute policy margin (gap between best and 2nd-best action) at junctions.

    From §11.3: Decision quality at junctions.

    Parameters:
    -----------
    trajectory : dict
        Trajectory with 'node_sequence'
    graph : nx.Graph
        Maze graph
    action_probs : list of np.ndarray
        Action probability distributions [T, num_actions]

    Returns:
    --------
    metrics : dict
        - mean_junction_margin: average margin at junctions
        - mean_corridor_margin: average margin at corridors
    """
    if 'node_sequence' not in trajectory or len(action_probs) == 0:
        return {
            'mean_junction_margin': np.nan,
            'mean_corridor_margin': np.nan
        }

    node_sequence = trajectory['node_sequence']

    junction_margins = []
    corridor_margins = []

    for i, node in enumerate(node_sequence[:-1]):
        if i >= len(action_probs):
            break

        probs = action_probs[i]
        # Sort to get top 2
        sorted_probs = np.sort(probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else sorted_probs[0]

        degree = graph.degree(node)
        if degree >= 3:
            junction_margins.append(margin)
        elif degree == 2:
            corridor_margins.append(margin)

    return {
        'mean_junction_margin': np.mean(junction_margins) if len(junction_margins) > 0 else np.nan,
        'mean_corridor_margin': np.mean(corridor_margins) if len(corridor_margins) > 0 else np.nan
    }


def compute_wrong_turn_rate(trajectory, graph, shortest_path):
    """
    Compute rate of wrong turns at junction nodes.

    From §11.3: Decision quality at junctions.

    Parameters:
    -----------
    trajectory : dict
        Trajectory with 'node_sequence'
    graph : nx.Graph
        Maze graph
    shortest_path : list
        Shortest path nodes

    Returns:
    --------
    wrong_turn_rate : float
        Fraction of junctions where agent chose off-path action
    """
    if 'node_sequence' not in trajectory:
        return np.nan

    node_sequence = trajectory['node_sequence']
    shortest_path_set = set(shortest_path)

    num_junctions = 0
    num_wrong_turns = 0

    for i, node in enumerate(node_sequence[:-1]):
        if graph.degree(node) >= 3:
            num_junctions += 1
            next_node = node_sequence[i + 1]

            # Wrong turn if next_node is NOT on shortest path
            # or if it violates the shortest path sequence
            if node in shortest_path:
                idx = shortest_path.index(node)
                if idx < len(shortest_path) - 1:
                    if next_node != shortest_path[idx + 1]:
                        num_wrong_turns += 1
                else:
                    # Already at goal, shouldn't happen
                    pass
            else:
                # Already off path, any junction choice is a "wrong turn"
                num_wrong_turns += 1

    return num_wrong_turns / num_junctions if num_junctions > 0 else np.nan


if __name__ == "__main__":
    print("Testing evaluation metrics...")

    # Create sample data for testing
    np.random.seed(42)

    # Test AUC computation
    print("\n1. Testing AUC computation:")
    sample_data = []
    for topology in ['corridor_short', 'proc_junction_heavy']:
        for lambda_val in [0.0, 0.5, 0.95]:
            for seed in [42, 43]:
                for episode in range(100):
                    # Simulate learning curve: success rate increases
                    base_success = min(1.0, episode / 50.0 + 0.1 * lambda_val)
                    success = np.random.rand() < base_success

                    sample_data.append({
                        'topology': topology,
                        'lambda': lambda_val,
                        'seed': seed,
                        'episode': episode,
                        'success': int(success),
                        'episode_length': 10 + np.random.randint(5),
                        'optimal_path_length': 10
                    })

    df = pd.DataFrame(sample_data)
    auc_results = compute_auc(df)
    print(f"  Computed AUC for {len(auc_results)} conditions")
    print(f"  Sample AUC values: {auc_results['auc'].describe()}")

    # Test episodes to threshold
    print("\n2. Testing episodes-to-threshold:")
    threshold_results = compute_episodes_to_threshold(df)
    print(f"  Computed thresholds for {len(threshold_results)} conditions")
    print(f"  Sample episodes to 80%: {threshold_results['episodes_to_80pct'].describe()}")

    # Test steps to threshold
    print("\n3. Testing steps-to-threshold:")
    steps_results = compute_steps_to_threshold(df)
    print(f"  Computed steps for {len(steps_results)} conditions")

    # Test timeout rate
    print("\n4. Testing timeout rate:")
    timeout_results = compute_timeout_rate(df)
    print(f"  Computed timeout rates for {len(timeout_results)} conditions")

    # Test junction metrics
    print("\n5. Testing junction decision metrics:")
    # Create simple test graph and trajectory
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4)])  # Junction at node 1

    trajectory = {
        'node_sequence': [0, 1, 2, 4]  # Correct path
    }
    shortest_path = [0, 1, 2, 4]

    junction_acc = compute_junction_decision_accuracy(trajectory, G, shortest_path)
    print(f"  Junction accuracy: {junction_acc}")

    action_probs = [
        np.array([0.1, 0.9, 0.0, 0.0, 0.0]),  # At node 0
        np.array([0.0, 0.1, 0.8, 0.1, 0.0]),  # At node 1 (junction)
        np.array([0.0, 0.0, 0.1, 0.0, 0.9]),  # At node 2
    ]

    entropy_metrics = compute_junction_action_entropy(trajectory, G, action_probs)
    print(f"  Junction entropy metrics: {entropy_metrics}")

    margin_metrics = compute_junction_policy_margin(trajectory, G, action_probs)
    print(f"  Junction margin metrics: {margin_metrics}")

    wrong_turn_rate = compute_wrong_turn_rate(trajectory, G, shortest_path)
    print(f"  Wrong turn rate: {wrong_turn_rate}")

    print("\n✓ Evaluation metrics tests passed!")
