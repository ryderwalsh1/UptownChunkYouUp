import numpy as np
import matplotlib.pyplot as plt

def lambda_labels(
    tokens: np.ndarray | list[int],
    vocab_size: int,
    lambda_: float = 1.0,
    pad_id: int = 0
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

    # 1. Identify valid tokens
    mask = tokens != pad_id
    if not np.any(mask):
        return labels

    # 2. Find the terminal target (the last valid token in the sequence)
    last_idx = int(np.sum(mask)) - 1
    end_token = tokens[last_idx]

    # 3. Initialize the backward carry
    # The carry represents the label distribution propagating backward from the end
    carry = np.zeros(vocab_size, dtype=np.float32)
    carry[end_token] = 1.0

    # 4. Iterate backward to propagate future information
    for t in range(seq_len - 1, -1, -1):
        if not mask[t]:
            # Padding tokens don't generate predictions; output is zero.
            labels[t] = 0.0
            # (The carry remains unchanged and passes through)
        else:
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

