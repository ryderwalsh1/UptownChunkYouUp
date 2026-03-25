"""
Lambda Modulator

Maps control demand to eligibility trace parameter λ.
High control demand → low λ (local credit assignment)
Low control demand → high λ (broad credit assignment, chunk-like learning)
"""

import numpy as np
import torch


class LambdaModulator:
    def __init__(self, beta=2.0, w_long=0.8, w_short=0.2, lambda_min=0.1, lambda_max=0.99):
        """
        Initialize lambda modulator.

        Parameters:
        -----------
        beta : float
            Exponent for superlinear power-law mapping (beta > 1)
            Higher beta = more aggressive modulation
        w_long : float
            Weight for long-term signal (conflict map)
        w_short : float
            Weight for short-term signal (controller output)
        lambda_min : float
            Minimum lambda value
        lambda_max : float
            Maximum lambda value
        """
        if beta <= 1.0:
            raise ValueError("beta must be > 1 for superlinear mapping")

        if w_long + w_short != 1.0:
            # Normalize weights
            total = w_long + w_short
            w_long = w_long / total
            w_short = w_short / total

        self.beta = beta
        self.w_long = w_long
        self.w_short = w_short
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def compute_lambda(self, conflict_map_value, p_slow):
        """
        Compute lambda from control demand.

        Formula:
        d_t = w_long * C_map(s_t) + w_short * p_slow
        lambda_t = (1 - d_t)^beta

        where d_t is control demand in [0, 1].

        Parameters:
        -----------
        conflict_map_value : float or torch.Tensor or np.ndarray
            Long-term conflict value from map
        p_slow : float or torch.Tensor or np.ndarray
            Short-term probability of using slow processing

        Returns:
        --------
        lambda_val : float or torch.Tensor or np.ndarray
            Lambda value for eligibility traces
        """
        # Handle different input types
        is_tensor = isinstance(conflict_map_value, torch.Tensor) or isinstance(p_slow, torch.Tensor)
        is_array = isinstance(conflict_map_value, np.ndarray) or isinstance(p_slow, np.ndarray)

        # Convert to appropriate type
        if is_tensor:
            if not isinstance(conflict_map_value, torch.Tensor):
                conflict_map_value = torch.tensor(conflict_map_value, dtype=torch.float32)
            if not isinstance(p_slow, torch.Tensor):
                p_slow = torch.tensor(p_slow, dtype=torch.float32)

            # Compute control demand (weighted combination)
            d_t = self.w_long * conflict_map_value + self.w_short * p_slow

            # sigmoid to ensure d_t is in [0, 1]
            d_t = torch.sigmoid(d_t)

            # Apply superlinear power-law mapping
            lambda_val = (1.0 - d_t) ** self.beta

            # Clamp to valid range
            lambda_val = torch.clamp(lambda_val, self.lambda_min, self.lambda_max)

        elif is_array:
            if not isinstance(conflict_map_value, np.ndarray):
                conflict_map_value = np.array(conflict_map_value, dtype=np.float32)
            if not isinstance(p_slow, np.ndarray):
                p_slow = np.array(p_slow, dtype=np.float32)

            d_t = self.w_long * conflict_map_value + self.w_short * p_slow
            d_t = np.clip(d_t, 0.0, 1.0)
            lambda_val = (1.0 - d_t) ** self.beta
            lambda_val = np.clip(lambda_val, self.lambda_min, self.lambda_max)

        else:
            # Scalar
            d_t = self.w_long * conflict_map_value + self.w_short * p_slow
            d_t = max(0.0, min(1.0, d_t))
            lambda_val = (1.0 - d_t) ** self.beta
            lambda_val = max(self.lambda_min, min(self.lambda_max, lambda_val))

        return lambda_val

    def is_chunk_eligible(self, lambda_val, threshold=0.8):
        """
        Check if lambda value indicates chunk-eligible region.

        High lambda (> threshold) suggests low control demand,
        indicating a potential chunk candidate.

        Parameters:
        -----------
        lambda_val : float or torch.Tensor or np.ndarray
            Lambda value
        threshold : float
            Threshold for chunk eligibility

        Returns:
        --------
        eligible : bool or torch.Tensor or np.ndarray
            Whether region is chunk-eligible
        """
        if isinstance(lambda_val, torch.Tensor):
            return lambda_val > threshold
        elif isinstance(lambda_val, np.ndarray):
            return lambda_val > threshold
        else:
            return lambda_val > threshold

    def get_control_demand(self, conflict_map_value, p_slow):
        """
        Get the control demand value (before power-law transformation).

        Returns:
        --------
        d_t : float or torch.Tensor or np.ndarray
            Control demand in [0, 1]
        """
        is_tensor = isinstance(conflict_map_value, torch.Tensor) or isinstance(p_slow, torch.Tensor)
        is_array = isinstance(conflict_map_value, np.ndarray) or isinstance(p_slow, np.ndarray)

        if is_tensor:
            if not isinstance(conflict_map_value, torch.Tensor):
                conflict_map_value = torch.tensor(conflict_map_value, dtype=torch.float32)
            if not isinstance(p_slow, torch.Tensor):
                p_slow = torch.tensor(p_slow, dtype=torch.float32)

            d_t = self.w_long * conflict_map_value + self.w_short * p_slow
            d_t = torch.clamp(d_t, 0.0, 1.0)

        elif is_array:
            if not isinstance(conflict_map_value, np.ndarray):
                conflict_map_value = np.array(conflict_map_value, dtype=np.float32)
            if not isinstance(p_slow, np.ndarray):
                p_slow = np.array(p_slow, dtype=np.float32)

            d_t = self.w_long * conflict_map_value + self.w_short * p_slow
            d_t = np.clip(d_t, 0.0, 1.0)

        else:
            d_t = self.w_long * conflict_map_value + self.w_short * p_slow
            d_t = max(0.0, min(1.0, d_t))

        return d_t


def visualize_lambda_modulation(beta_values=[1.5, 2.0, 3.0], w_long=0.8, w_short=0.2):
    """
    Visualize lambda modulation curves for different beta values.

    Parameters:
    -----------
    beta_values : list of float
        Different beta values to compare
    w_long : float
        Weight for long-term signal
    w_short : float
        Weight for short-term signal
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Range of control demand values
    d_values = np.linspace(0, 1, 100)

    # Plot 1: Lambda vs control demand for different beta
    ax = axes[0]
    for beta in beta_values:
        modulator = LambdaModulator(beta=beta, w_long=w_long, w_short=w_short)
        lambda_values = [(1 - d) ** beta for d in d_values]
        ax.plot(d_values, lambda_values, label=f'β={beta}', linewidth=2)

    ax.set_xlabel('Control Demand', fontsize=12)
    ax.set_ylabel('Lambda (λ)', fontsize=12)
    ax.set_title('Lambda vs Control Demand', fontsize=14, fontweight='500')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Plot 2: Lambda vs conflict map (with p_slow fixed)
    ax = axes[1]
    p_slow_fixed = 0.2
    conflict_values = np.linspace(0, 1, 100)

    for beta in beta_values:
        modulator = LambdaModulator(beta=beta, w_long=w_long, w_short=w_short)
        lambda_values = [modulator.compute_lambda(c, p_slow_fixed) for c in conflict_values]
        ax.plot(conflict_values, lambda_values, label=f'β={beta}', linewidth=2)

    ax.set_xlabel('Conflict Map Value', fontsize=12)
    ax.set_ylabel('Lambda (λ)', fontsize=12)
    ax.set_title(f'Lambda vs Conflict (p_slow={p_slow_fixed})', fontsize=14, fontweight='500')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Plot 3: Lambda vs p_slow (with conflict fixed)
    ax = axes[2]
    conflict_fixed = 0.2
    p_slow_values = np.linspace(0, 1, 100)

    for beta in beta_values:
        modulator = LambdaModulator(beta=beta, w_long=w_long, w_short=w_short)
        lambda_values = [modulator.compute_lambda(conflict_fixed, p) for p in p_slow_values]
        ax.plot(p_slow_values, lambda_values, label=f'β={beta}', linewidth=2)

    ax.set_xlabel('p(slow)', fontsize=12)
    ax.set_ylabel('Lambda (λ)', fontsize=12)
    ax.set_title(f'Lambda vs p(slow) (conflict={conflict_fixed})', fontsize=14, fontweight='500')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('lambda_modulation_curves.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to lambda_modulation_curves.png")


if __name__ == "__main__":
    print("Testing LambdaModulator...")

    # Create modulator
    modulator = LambdaModulator(beta=2.0, w_long=0.8, w_short=0.2)
    print(f"Created LambdaModulator:")
    print(f"  beta: {modulator.beta}")
    print(f"  w_long: {modulator.w_long}, w_short: {modulator.w_short}")

    # Test with scalar inputs
    print("\nScalar inputs:")
    conflict = 0.3
    p_slow = 0.4

    lambda_val = modulator.compute_lambda(conflict, p_slow)
    control_demand = modulator.get_control_demand(conflict, p_slow)

    print(f"  Conflict map: {conflict}")
    print(f"  p_slow: {p_slow}")
    print(f"  Control demand: {control_demand:.3f}")
    print(f"  Lambda: {lambda_val:.3f}")
    print(f"  Chunk eligible: {modulator.is_chunk_eligible(lambda_val)}")

    # Test edge cases
    print("\nEdge cases:")
    print(f"  Low conflict (0.0), low p_slow (0.0):")
    print(f"    Lambda: {modulator.compute_lambda(0.0, 0.0):.3f} (should be high)")

    print(f"  High conflict (1.0), high p_slow (1.0):")
    print(f"    Lambda: {modulator.compute_lambda(1.0, 1.0):.3f} (should be low)")

    print(f"  Medium conflict (0.5), medium p_slow (0.5):")
    print(f"    Lambda: {modulator.compute_lambda(0.5, 0.5):.3f}")

    # Test with arrays
    print("\nArray inputs:")
    conflict_array = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    p_slow_array = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    lambda_array = modulator.compute_lambda(conflict_array, p_slow_array)
    print(f"  Conflicts: {conflict_array}")
    print(f"  Lambdas: {lambda_array}")

    # Test with tensors
    print("\nTensor inputs:")
    conflict_tensor = torch.tensor([0.1, 0.5, 0.9])
    p_slow_tensor = torch.tensor([0.3, 0.3, 0.3])

    lambda_tensor = modulator.compute_lambda(conflict_tensor, p_slow_tensor)
    print(f"  Conflicts: {conflict_tensor}")
    print(f"  Lambdas: {lambda_tensor}")

    # Compare different beta values
    print("\nComparing different beta values (conflict=0.3, p_slow=0.2):")
    for beta in [1.5, 2.0, 3.0, 5.0]:
        mod = LambdaModulator(beta=beta, w_long=0.8, w_short=0.2)
        lam = mod.compute_lambda(0.3, 0.2)
        print(f"  beta={beta}: lambda={lam:.3f}")

    # Demonstrate weighting effect
    print("\nDemonstrating long-term vs short-term weighting:")
    print("  High conflict (0.8), low p_slow (0.1):")
    print(f"    w_long=0.8, w_short=0.2: lambda={modulator.compute_lambda(0.8, 0.1):.3f}")

    mod_balanced = LambdaModulator(beta=2.0, w_long=0.5, w_short=0.5)
    print(f"    w_long=0.5, w_short=0.5: lambda={mod_balanced.compute_lambda(0.8, 0.1):.3f}")

    print("\n  Low conflict (0.1), high p_slow (0.8):")
    print(f"    w_long=0.8, w_short=0.2: lambda={modulator.compute_lambda(0.1, 0.8):.3f}")
    print(f"    w_long=0.5, w_short=0.5: lambda={mod_balanced.compute_lambda(0.1, 0.8):.3f}")

    # Generate visualization
    print("\nGenerating visualization...")
    visualize_lambda_modulation(beta_values=[1.5, 2.0, 3.0], w_long=0.8, w_short=0.2)

    print("\n✓ LambdaModulator tests passed!")
