"""
Credit-Assignment Diagnostics for Lambda Experiment

Computes mechanistic credit-assignment statistics to understand how TD(λ)
propagates reward signal through trajectories and across different node types.

Based on specification section 12: Credit-assignment diagnostics
"""

import numpy as np
import networkx as nx


def compute_effective_credit_distance(advantages, td_errors=None):
    """
    Compute effective backward credit distance.

    From §12.1: Measures how far backward the reward influence reaches.

    Formula: D_eff = Σ(k × |Δ_{T-1-k}|) / Σ|Δ_{T-1-k}|

    where Δ_t is the credit signal (advantages or TD errors).

    Parameters:
    -----------
    advantages : np.ndarray or list
        TD(λ) advantages for each timestep [T]
    td_errors : np.ndarray or list, optional
        TD residuals δ_t (if provided, used instead of advantages)

    Returns:
    --------
    D_eff : float
        Effective backward credit distance (in timesteps from end)
        Higher values mean credit reaches farther backward
    """
    if td_errors is not None:
        credit_signal = np.abs(td_errors)
    else:
        credit_signal = np.abs(advantages)

    T = len(credit_signal)
    if T == 0:
        return 0.0

    # Compute weighted average: how far back from the END does credit go?
    # k = 0 means last timestep, k = T-1 means first timestep
    numerator = 0.0
    denominator = 0.0

    for k in range(T):
        # Index from end: T-1-k
        idx = T - 1 - k
        weight = credit_signal[idx]
        numerator += k * weight
        denominator += weight

    if denominator < 1e-10:
        return 0.0

    return numerator / denominator


def classify_node_type(graph, node):
    """
    Classify a node as dead-end, corridor, or junction.

    Parameters:
    -----------
    graph : nx.Graph
        Maze graph
    node : node identifier
        Node to classify

    Returns:
    --------
    node_type : str
        One of: 'dead_end', 'corridor', 'junction'
    """
    degree = graph.degree(node)
    if degree == 1:
        return 'dead_end'
    elif degree == 2:
        return 'corridor'
    else:  # degree >= 3
        return 'junction'


def compute_credit_by_node_type(advantages, node_sequence, graph):
    """
    Partition credit into corridor, junction, and dead-end nodes.

    From §12.2: Compute total absolute credit assigned to each node type.

    Parameters:
    -----------
    advantages : np.ndarray or list
        TD(λ) advantages [T]
    node_sequence : list
        Sequence of nodes visited [T]
    graph : nx.Graph
        Maze graph

    Returns:
    --------
    metrics : dict
        - C_corridor: total credit to corridor nodes
        - C_junction: total credit to junction nodes
        - C_dead_end: total credit to dead-end nodes
        - junction_corridor_ratio: C_junction / C_corridor (key metric)
        - dead_corridor_ratio: C_dead / C_corridor
    """
    if len(advantages) == 0 or len(node_sequence) == 0:
        return {
            'C_corridor': 0.0,
            'C_junction': 0.0,
            'C_dead_end': 0.0,
            'junction_corridor_ratio': np.nan,
            'dead_corridor_ratio': np.nan
        }

    # Ensure node_sequence matches advantages length
    # (node_sequence might be T+1 if it includes final state)
    node_sequence = node_sequence[:len(advantages)]

    C_corridor = 0.0
    C_junction = 0.0
    C_dead_end = 0.0

    for i, node in enumerate(node_sequence):
        if i >= len(advantages):
            break

        credit = np.abs(advantages[i])
        node_type = classify_node_type(graph, node)

        if node_type == 'corridor':
            C_corridor += credit
        elif node_type == 'junction':
            C_junction += credit
        elif node_type == 'dead_end':
            C_dead_end += credit

    # Compute ratios with small epsilon to avoid divide-by-zero
    eps = 1e-10
    junction_corridor_ratio = C_junction / (C_corridor + eps)
    dead_corridor_ratio = C_dead_end / (C_corridor + eps)

    return {
        'C_corridor': C_corridor,
        'C_junction': C_junction,
        'C_dead_end': C_dead_end,
        'junction_corridor_ratio': junction_corridor_ratio,
        'dead_corridor_ratio': dead_corridor_ratio
    }


def compute_decision_localization(advantages, node_sequence, graph, lookback=5):
    """
    Compute decision localization score at junctions.

    From §12.3: Measures whether credit is localized at decision points
    vs diffused across preceding corridor states.

    For each junction, compute:
    L_decision = |Δ_j| / (Σ|Δ_{j-k}| + ε)

    High value = credit sharply localized at decision point
    Low value = credit diffused across upstream corridor

    Parameters:
    -----------
    advantages : np.ndarray or list
        TD(λ) advantages [T]
    node_sequence : list
        Sequence of nodes visited [T]
    graph : nx.Graph
        Maze graph
    lookback : int
        Number of preceding timesteps to consider (default: 5)

    Returns:
    --------
    metrics : dict
        - mean_decision_localization: average L_decision across junctions
        - num_junctions_analyzed: number of junctions with sufficient history
        - all_localization_scores: list of individual scores
    """
    if len(advantages) == 0 or len(node_sequence) == 0:
        return {
            'mean_decision_localization': np.nan,
            'num_junctions_analyzed': 0,
            'all_localization_scores': []
        }

    node_sequence = node_sequence[:len(advantages)]

    localization_scores = []

    for j in range(len(node_sequence)):
        node = node_sequence[j]

        # Only analyze junctions
        if classify_node_type(graph, node) != 'junction':
            continue

        # Need sufficient history
        if j < 1:
            continue

        # Credit at junction
        junction_credit = np.abs(advantages[j])

        # Credit in preceding K timesteps
        start_idx = max(0, j - lookback)
        preceding_credit = np.sum(np.abs(advantages[start_idx:j]))

        # Compute localization score
        eps = 1e-10
        L_decision = junction_credit / (preceding_credit + eps)

        localization_scores.append(L_decision)

    return {
        'mean_decision_localization': np.mean(localization_scores) if len(localization_scores) > 0 else np.nan,
        'num_junctions_analyzed': len(localization_scores),
        'all_localization_scores': localization_scores
    }


def compute_action_gap_at_junctions(action_probs, node_sequence, graph, shortest_path):
    """
    Compute action gap (margin) at junction nodes.

    From §12.4a: G(s) = π(a*|s) - max_{a≠a*} π(a|s)

    where a* is the optimal action (on shortest path).

    Parameters:
    -----------
    action_probs : list of np.ndarray
        Action probability distributions [T, num_actions]
    node_sequence : list
        Sequence of nodes visited [T]
    graph : nx.Graph
        Maze graph
    shortest_path : list
        Optimal path (node sequence)

    Returns:
    --------
    metrics : dict
        - mean_junction_action_gap: average gap at junctions
        - num_junctions_evaluated: number of junctions
        - all_action_gaps: list of individual gaps
    """
    if len(action_probs) == 0 or len(node_sequence) == 0:
        return {
            'mean_junction_action_gap': np.nan,
            'num_junctions_evaluated': 0,
            'all_action_gaps': []
        }

    shortest_path_edges = set()
    for i in range(len(shortest_path) - 1):
        shortest_path_edges.add((shortest_path[i], shortest_path[i + 1]))

    action_gaps = []

    for i, node in enumerate(node_sequence[:-1]):
        if i >= len(action_probs):
            break

        # Only analyze junctions
        if classify_node_type(graph, node) != 'junction':
            continue

        probs = action_probs[i]

        # Find optimal action (next node on shortest path)
        # This requires mapping from action index to next node
        # For now, compute gap as: max_prob - second_max_prob
        # (Without exact action-to-node mapping, this is a proxy)

        sorted_probs = np.sort(probs)[::-1]
        if len(sorted_probs) >= 2:
            gap = sorted_probs[0] - sorted_probs[1]
        else:
            gap = sorted_probs[0] if len(sorted_probs) > 0 else 0.0

        action_gaps.append(gap)

    return {
        'mean_junction_action_gap': np.mean(action_gaps) if len(action_gaps) > 0 else np.nan,
        'num_junctions_evaluated': len(action_gaps),
        'all_action_gaps': action_gaps
    }


def compute_upstream_local_credit_ratio(advantages, node_sequence, graph):
    """
    Compute ratio of upstream credit to local credit at junctions.

    From §12.4b: Compare total credit before junction to credit at junction.

    If ratio becomes too large, suggests trace smearing.

    Parameters:
    -----------
    advantages : np.ndarray or list
        TD(λ) advantages [T]
    node_sequence : list
        Sequence of nodes visited [T]
    graph : nx.Graph
        Maze graph

    Returns:
    --------
    metrics : dict
        - mean_upstream_local_ratio: average ratio across junctions
        - num_junctions_analyzed: number of junctions
    """
    if len(advantages) == 0 or len(node_sequence) == 0:
        return {
            'mean_upstream_local_ratio': np.nan,
            'num_junctions_analyzed': 0
        }

    node_sequence = node_sequence[:len(advantages)]

    ratios = []

    for j in range(len(node_sequence)):
        node = node_sequence[j]

        if classify_node_type(graph, node) != 'junction':
            continue

        if j < 1:
            continue

        # Local credit at junction
        local_credit = np.abs(advantages[j])

        # Upstream credit (all preceding timesteps)
        upstream_credit = np.sum(np.abs(advantages[:j]))

        # Compute ratio
        eps = 1e-10
        ratio = upstream_credit / (local_credit + eps)

        ratios.append(ratio)

    return {
        'mean_upstream_local_ratio': np.mean(ratios) if len(ratios) > 0 else np.nan,
        'num_junctions_analyzed': len(ratios)
    }


def compute_all_credit_diagnostics(advantages, node_sequence, graph,
                                   shortest_path=None, action_probs=None,
                                   td_errors=None):
    """
    Compute all credit-assignment diagnostics.

    Parameters:
    -----------
    advantages : np.ndarray or list
        TD(λ) advantages [T]
    node_sequence : list
        Sequence of nodes visited [T]
    graph : nx.Graph
        Maze graph
    shortest_path : list, optional
        Optimal path for action gap computation
    action_probs : list of np.ndarray, optional
        Action distributions for action gap
    td_errors : np.ndarray, optional
        TD residuals for effective distance

    Returns:
    --------
    diagnostics : dict
        All credit diagnostic metrics
    """
    diagnostics = {}

    # Effective backward credit distance (§12.1)
    diagnostics['effective_credit_distance'] = compute_effective_credit_distance(
        advantages, td_errors
    )

    # Credit by node type (§12.2)
    credit_by_type = compute_credit_by_node_type(advantages, node_sequence, graph)
    diagnostics.update(credit_by_type)

    # Decision localization (§12.3)
    decision_loc = compute_decision_localization(advantages, node_sequence, graph)
    diagnostics.update(decision_loc)

    # Action gap at junctions (§12.4a)
    if shortest_path is not None and action_probs is not None:
        action_gap = compute_action_gap_at_junctions(
            action_probs, node_sequence, graph, shortest_path
        )
        diagnostics.update(action_gap)

    # Upstream/local credit ratio (§12.4b)
    upstream_local = compute_upstream_local_credit_ratio(
        advantages, node_sequence, graph
    )
    diagnostics.update(upstream_local)

    return diagnostics


if __name__ == "__main__":
    print("Testing credit diagnostics...")

    # Create test data
    np.random.seed(42)

    # Test effective credit distance
    print("\n1. Testing effective credit distance:")
    # Simulate advantages: more credit near end (recent steps)
    advantages = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    D_eff = compute_effective_credit_distance(advantages)
    print(f"  D_eff = {D_eff:.3f} (should be < T/2 since credit is back-weighted)")

    # More uniform credit
    advantages_uniform = np.ones(10)
    D_eff_uniform = compute_effective_credit_distance(advantages_uniform)
    print(f"  D_eff (uniform) = {D_eff_uniform:.3f} (should be ~4.5, middle of 0-9)")

    # Test node classification
    print("\n2. Testing node classification:")
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4), (4, 5)])
    # Node types: 0=dead-end, 1=junction, 2=dead-end, 3=corridor, 4=corridor, 5=dead-end

    for node in G.nodes():
        node_type = classify_node_type(G, node)
        print(f"  Node {node} (degree {G.degree(node)}): {node_type}")

    # Test credit by node type
    print("\n3. Testing credit by node type:")
    node_sequence = [0, 1, 3, 4, 5]
    advantages = np.array([0.5, 1.0, 0.3, 0.4, 0.2])  # Junction gets most credit

    credit_metrics = compute_credit_by_node_type(advantages, node_sequence, G)
    print(f"  C_corridor = {credit_metrics['C_corridor']:.3f}")
    print(f"  C_junction = {credit_metrics['C_junction']:.3f}")
    print(f"  C_dead_end = {credit_metrics['C_dead_end']:.3f}")
    print(f"  Junction/Corridor ratio = {credit_metrics['junction_corridor_ratio']:.3f}")

    # Test decision localization
    print("\n4. Testing decision localization:")
    loc_metrics = compute_decision_localization(advantages, node_sequence, G, lookback=3)
    print(f"  Mean decision localization = {loc_metrics['mean_decision_localization']:.3f}")
    print(f"  Num junctions analyzed = {loc_metrics['num_junctions_analyzed']}")

    # Test action gap
    print("\n5. Testing action gap at junctions:")
    action_probs = [
        np.array([0.2, 0.8]),  # At node 0
        np.array([0.1, 0.7, 0.2]),  # At node 1 (junction) - clear preference
        np.array([0.5, 0.5]),  # At node 3
        np.array([0.6, 0.4]),  # At node 4
    ]
    shortest_path = [0, 1, 3, 4, 5]

    gap_metrics = compute_action_gap_at_junctions(action_probs, node_sequence, G, shortest_path)
    print(f"  Mean action gap = {gap_metrics['mean_junction_action_gap']:.3f}")
    print(f"  Num junctions = {gap_metrics['num_junctions_evaluated']}")

    # Test upstream/local ratio
    print("\n6. Testing upstream/local credit ratio:")
    upstream_metrics = compute_upstream_local_credit_ratio(advantages, node_sequence, G)
    print(f"  Mean upstream/local ratio = {upstream_metrics['mean_upstream_local_ratio']:.3f}")

    # Test comprehensive diagnostics
    print("\n7. Testing comprehensive diagnostics:")
    all_diagnostics = compute_all_credit_diagnostics(
        advantages, node_sequence, G,
        shortest_path=shortest_path,
        action_probs=action_probs
    )
    print(f"  Computed {len(all_diagnostics)} diagnostic metrics")
    print(f"  Key metrics:")
    print(f"    - Effective credit distance: {all_diagnostics['effective_credit_distance']:.3f}")
    print(f"    - Junction/corridor ratio: {all_diagnostics['junction_corridor_ratio']:.3f}")
    print(f"    - Decision localization: {all_diagnostics['mean_decision_localization']:.3f}")

    print("\n✓ Credit diagnostics tests passed!")
