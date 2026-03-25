"""
Conflict Map

Per-state long-term storage of fast-slow policy disagreement.
Tracks where cognitive control is historically needed.
"""

import numpy as np
import torch
import torch.nn.functional as F


class ConflictMap:
    def __init__(self, num_states, alpha=0.01, initial_value=0.0):
        """
        Initialize conflict map.

        Parameters:
        -----------
        num_states : int
            Number of states in the environment
        alpha : float
            EMA update rate (small value for long-term averaging)
        initial_value : float
            Initial conflict value for all states
        """
        self.num_states = num_states
        self.alpha = alpha
        self.initial_value = initial_value

        # Per-state conflict values (EMA of KL divergence)
        self.conflict_values = np.full(num_states, initial_value, dtype=np.float32)

        # Track update counts per state (for diagnostics)
        self.update_counts = np.zeros(num_states, dtype=np.int32)

    def get(self, state_idx):
        """
        Get conflict value for a state.

        Parameters:
        -----------
        state_idx : int or np.ndarray
            State index or indices

        Returns:
        --------
        conflict : float or np.ndarray
            Conflict value(s)
        """
        return self.conflict_values[state_idx]

    def update(self, state_idx, kl_divergence):
        """
        Update conflict value for a state using exponential moving average.

        Parameters:
        -----------
        state_idx : int
            State index
        kl_divergence : float
            Current KL divergence between fast and slow policies
        """
        # EMA update: C(s) <- (1 - alpha) * C(s) + alpha * KL
        current_value = self.conflict_values[state_idx]
        new_value = (1 - self.alpha) * current_value + self.alpha * kl_divergence
        self.conflict_values[state_idx] = new_value
        self.update_counts[state_idx] += 1

    def batch_update(self, state_indices, kl_divergences):
        """
        Update conflict values for multiple states.

        Parameters:
        -----------
        state_indices : np.ndarray or list
            Array of state indices [batch_size]
        kl_divergences : np.ndarray or list
            Array of KL divergences [batch_size]
        """
        state_indices = np.array(state_indices)
        kl_divergences = np.array(kl_divergences)

        for idx, kl in zip(state_indices, kl_divergences):
            self.update(idx, kl)

    def get_statistics(self):
        """
        Get statistics about conflict map.

        Returns:
        --------
        stats : dict
            Dictionary with statistics
        """
        return {
            'mean_conflict': float(np.mean(self.conflict_values)),
            'std_conflict': float(np.std(self.conflict_values)),
            'min_conflict': float(np.min(self.conflict_values)),
            'max_conflict': float(np.max(self.conflict_values)),
            'median_conflict': float(np.median(self.conflict_values)),
            'total_updates': int(np.sum(self.update_counts)),
            'states_never_updated': int(np.sum(self.update_counts == 0)),
            'states_updated': int(np.sum(self.update_counts > 0))
        }

    def get_top_conflict_states(self, k=10):
        """
        Get states with highest conflict values.

        Parameters:
        -----------
        k : int
            Number of top states to return

        Returns:
        --------
        top_states : np.ndarray
            Indices of top-k states with highest conflict
        top_values : np.ndarray
            Conflict values for top-k states
        """
        top_indices = np.argsort(self.conflict_values)[-k:][::-1]
        top_values = self.conflict_values[top_indices]
        return top_indices, top_values

    def get_low_conflict_states(self, k=10):
        """
        Get states with lowest conflict values (chunk candidates).

        Parameters:
        -----------
        k : int
            Number of bottom states to return

        Returns:
        --------
        low_states : np.ndarray
            Indices of bottom-k states with lowest conflict
        low_values : np.ndarray
            Conflict values for bottom-k states
        """
        low_indices = np.argsort(self.conflict_values)[:k]
        low_values = self.conflict_values[low_indices]
        return low_indices, low_values

    def reset(self):
        """Reset all conflict values to initial value."""
        self.conflict_values = np.full(self.num_states, self.initial_value, dtype=np.float32)
        self.update_counts = np.zeros(self.num_states, dtype=np.int32)

    def save(self, filepath):
        """Save conflict map to file."""
        np.savez(filepath,
                 conflict_values=self.conflict_values,
                 update_counts=self.update_counts,
                 alpha=self.alpha,
                 initial_value=self.initial_value)

    def load(self, filepath):
        """Load conflict map from file."""
        data = np.load(filepath)
        self.conflict_values = data['conflict_values']
        self.update_counts = data['update_counts']
        self.alpha = float(data['alpha'])
        self.initial_value = float(data['initial_value'])


def compute_kl_divergence(logits_p, logits_q):
    """
    Compute KL divergence between two categorical distributions.

    KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))

    Parameters:
    -----------
    logits_p : torch.Tensor
        Logits for distribution P [batch_size, num_actions]
    logits_q : torch.Tensor
        Logits for distribution Q [batch_size, num_actions]

    Returns:
    --------
    kl : torch.Tensor
        KL divergence [batch_size]
    """
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = torch.exp(log_p)

    kl = (p * (log_p - log_q)).sum(dim=-1)
    return kl


def compute_js_divergence(logits_p, logits_q):
    """
    Compute Jensen-Shannon divergence between two distributions.

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    JS divergence is symmetric and bounded [0, 1].

    Parameters:
    -----------
    logits_p : torch.Tensor
        Logits for distribution P [batch_size, num_actions]
    logits_q : torch.Tensor
        Logits for distribution Q [batch_size, num_actions]

    Returns:
    --------
    js : torch.Tensor
        JS divergence [batch_size]
    """
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)

    m = 0.5 * (p + q)
    log_m = torch.log(m + 1e-10)

    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)

    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)

    js = 0.5 * kl_pm + 0.5 * kl_qm
    return js


if __name__ == "__main__":
    print("Testing ConflictMap...")

    num_states = 64
    conflict_map = ConflictMap(num_states=num_states, alpha=0.1)

    print(f"Created ConflictMap with {num_states} states")
    print(f"Initial statistics:")
    stats = conflict_map.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Simulate some updates
    print("\nSimulating updates...")
    np.random.seed(42)

    for episode in range(100):
        # Random walk through states
        for _ in range(20):
            state_idx = np.random.randint(num_states)
            kl_value = np.random.exponential(0.5)  # Random KL divergence
            conflict_map.update(state_idx, kl_value)

    print("\nAfter 100 episodes:")
    stats = conflict_map.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get top conflict states
    print("\nTop 5 conflict states:")
    top_states, top_values = conflict_map.get_top_conflict_states(k=5)
    for state, value in zip(top_states, top_values):
        print(f"  State {state}: {value:.4f} (updated {conflict_map.update_counts[state]} times)")

    # Get low conflict states
    print("\nBottom 5 conflict states (chunk candidates):")
    low_states, low_values = conflict_map.get_low_conflict_states(k=5)
    for state, value in zip(low_states, low_values):
        print(f"  State {state}: {value:.4f} (updated {conflict_map.update_counts[state]} times)")

    # Test KL divergence computation
    print("\nTesting KL divergence computation...")
    batch_size = 4
    num_actions = 4

    logits_fast = torch.randn(batch_size, num_actions)
    logits_slow = torch.randn(batch_size, num_actions)

    kl = compute_kl_divergence(logits_fast, logits_slow)
    js = compute_js_divergence(logits_fast, logits_slow)

    print(f"  KL divergence: {kl}")
    print(f"  JS divergence: {js}")

    # Test with identical distributions
    kl_same = compute_kl_divergence(logits_fast, logits_fast)
    js_same = compute_js_divergence(logits_fast, logits_fast)
    print(f"  KL (same distributions): {kl_same} (should be ~0)")
    print(f"  JS (same distributions): {js_same} (should be ~0)")

    # Save and load
    print("\nTesting save/load...")
    conflict_map.save('/tmp/conflict_map_test.npz')
    conflict_map_loaded = ConflictMap(num_states=num_states)
    conflict_map_loaded.load('/tmp/conflict_map_test.npz')

    print(f"  Original mean: {conflict_map.get_statistics()['mean_conflict']:.4f}")
    print(f"  Loaded mean: {conflict_map_loaded.get_statistics()['mean_conflict']:.4f}")
    print(f"  Match: {np.allclose(conflict_map.conflict_values, conflict_map_loaded.conflict_values)}")

    print("\n✓ ConflictMap tests passed!")
