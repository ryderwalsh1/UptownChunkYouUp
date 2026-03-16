import numpy as np
import matplotlib.pyplot as plt

def lambda_labels(
    tokens: np.ndarray | list[int],
    vocab_size: int,
    lambda_: float = 1.0
) -> np.ndarray:
    """
    Computes lambda-decayed soft labels for trajectory compression / step-skipping.

    Given a generated completion trajectory, this function computes targets that
    encourage the model to skip intermediate reasoning/generation steps.

    Conceptual mapping to TD(lambda):
      - lambda_ = 0.0 [TD(0)]: The target at step t is the token from t+1.
        The model learns to skip 1 step incrementally along its own trajectory.
      - lambda_ = 1.0 [TD(1)]: The target at every step is the final valid token.
        The model learns to skip the entire remaining sequence and jump straight
        to the conclusion.
      - 0 < lambda_ < 1 [TD(lambda)]: The target is an exponentially decaying
        mixture of the next token and all subsequent future tokens. It teaches
        the model a soft look-ahead over its own future reasoning path.
    """
    tokens = np.asarray(tokens)
    seq_len = len(tokens)
    labels = np.zeros((seq_len, vocab_size), dtype=np.float32)

    if seq_len == 0:
        return labels

    # Find the terminal target (the last token in the sequence)
    last_idx = seq_len - 1
    end_token = tokens[last_idx]

    # Initialize the backward carry
    # The carry represents the label distribution propagating backward from the end
    carry = np.zeros(vocab_size, dtype=np.float32)
    carry[end_token] = 1.0

    # Iterate backward to propagate future information
    for t in range(seq_len - 1, -1, -1):
        # The immediate target is the next token (or itself if it's the last token)
        if t == last_idx:
            next_token = end_token
        else:
            next_token = tokens[t + 1]

        next_one_hot = np.zeros(vocab_size, dtype=np.float32)
        next_one_hot[next_token] = 1.0

        # Interpolate:
        # (1 - lambda_) focuses on the 1-step skip (the immediate next token).
        # lambda_ focuses on the long-term goal (the carry from the end).
        soft_label = (1.0 - lambda_) * next_one_hot + lambda_ * carry

        labels[t] = soft_label
        carry = soft_label

    return labels

def lambda_values(
    rewards: np.ndarray,       # reward at each step (e.g., [0, 0, 0, 1])
    values: np.ndarray,        # current value estimates V(s_t) from the network
    gamma: float = 0.99,       # discount factor
    lambda_: float = 1.0
) -> np.ndarray:
    """
    Computes TD(lambda) scalar value targets via backward accumulation.
    
    - lambda_ = 0: target_t = r_{t+1} + gamma * V(s_{t+1})        [TD(0)]
    - lambda_ = 1: target_t = r_{t+1} + gamma*r_{t+2} + ... + gamma^n * R_T  [Monte Carlo]
    - 0 < lambda_ < 1: exponential blend of n-step returns
    """
    seq_len = len(rewards)
    targets = np.zeros(seq_len, dtype=np.float32)
    
    # Bootstrap from the terminal value (0 if episode ends)
    carry = 0.0
    
    for t in range(seq_len - 1, -1, -1):
        if t == seq_len - 1:
            # Terminal step: target is just the reward
            targets[t] = rewards[t]
            carry = rewards[t]
        else:
            # TD(0) target: one-step bootstrap
            td0_target = rewards[t] + gamma * values[t + 1]
            
            # Monte Carlo direction: use the carry
            mc_target = rewards[t] + gamma * carry
            
            # Blend
            targets[t] = (1.0 - lambda_) * td0_target + lambda_ * mc_target
            carry = targets[t]
    
    return targets