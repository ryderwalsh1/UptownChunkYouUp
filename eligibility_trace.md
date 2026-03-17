# Task: Add Eligibility Trace Context Input to PolicyNetwork

## Context

The PolicyNetwork currently takes (source_node, goal_node) as input — two one-hot vectors. Each forward pass is memoryless. We're adding a third input: a decaying accumulator that functions as an eligibility trace over the state space, providing sequential context about recently visited states.

## Cognitive and theoretical grounding

This trace input is mathematically identical to the accumulating eligibility trace from TD(λ):

```
E(S) ← E(S) + 1          (bump current state)
E(s) ← γλ · E(s)  ∀s     (decay all states)
```

The decay rate is γλ, where γ is the discount factor and λ is the same chunkability-modulated lambda already used for the value head's TD(λ) targets. This means the effective memory horizon grows as the policy automatizes — early in training (low λ), the trace decays quickly and decisions are nearly memoryless; late in training (high λ), the trace persists across multiple steps, enabling the network to detect it's mid-sequence.

## What to implement

### 1. Context trace management

Add a context trace vector and methods to manage it. The trace lives OUTSIDE the PsyNeuLink composition — it's maintained by the PolicyNetwork object during trajectory traversal.

```python
# In __init__:
self.context_trace = np.zeros(self.num_literals)

def reset_context(self):
    """Reset the context trace to zeros (start of a new trajectory)."""
    self.context_trace = np.zeros(self.num_literals)

def update_context(self, current_literal, decay):
    """
    Update the context trace with the current state using eligibility trace dynamics.
    
    This is mathematically identical to accumulating eligibility traces:
      E(s) ← decay * E(s)  for all s    (decay all)
      E(S) ← E(S) + 1                   (bump current)
    
    Args:
        current_literal: int - the state just visited
        decay: float - decay rate, should be gamma * lambda where lambda 
               is the chunkability-modulated trace parameter
    
    Returns:
        np.array - the updated context trace (also stored in self.context_trace)
    """
    self.context_trace = decay * self.context_trace
    current_idx = self.literal_to_idx(current_literal)
    self.context_trace[current_idx] += 1.0
    return self.context_trace.copy()
```

### 2. Third input mechanism in PsyNeuLink

Add a new ProcessingMechanism and MappingProjection for the context input, following the exact same pattern as the existing source and target inputs. In __init__:

```python
# Context trace input
self.policy_context_input = pnl.ProcessingMechanism(
    name=f'{policy_name}_Context_Input',
    input_shapes=self.num_literals
)

# Context to hidden projection
context_to_hidden_matrix = kwargs.get("context_to_hidden_matrix", None)
if context_to_hidden_matrix is not None:
    self.context_to_hidden = pnl.MappingProjection(
        matrix=context_to_hidden_matrix
    )
else:
    self.context_to_hidden = pnl.MappingProjection(
        matrix=(0.2 * np.random.rand(self.num_literals, hidden_size) - 0.1)
    )
```

Add this to the composition's processing pathways:
```python
policy_comp.add_linear_processing_pathway(
    [self.policy_context_input, self.context_to_hidden, 
     self.policy_hidden, self.hidden_to_output, self.policy_output]
)
```

And for the value head pathway too:
```python
policy_comp.add_linear_processing_pathway(
    [self.policy_context_input, self.context_to_hidden,
     self.policy_hidden, self.hidden_to_value, self.policy_value_output]
)
```

Add backpropagation learning pathways:
```python
policy_comp.add_backpropagation_learning_pathway(
    pathway=[self.policy_context_input, self.policy_hidden, self.policy_output],
    learning_rate=learning_rate
)
policy_comp.add_backpropagation_learning_pathway(
    pathway=[self.policy_context_input, self.policy_hidden, self.policy_value_output],
    learning_rate=learning_rate
)
```

The hidden layer now computes: σ(W_s·x_s + W_g·x_g + W_c·x_c + b)

### 3. Update ALL call sites

Every method that calls `self.policy.run()` or `self.policy.learn()` must now include the context input. Search the entire class for these calls. The context encoding to pass is `self.context_trace` (or a zero vector if context is not available).

Key methods to update (non-exhaustive — search for ALL occurrences):

- `predict(source, target)` — add optional `context` parameter, default to self.context_trace
- `traverse_path(...)` — reset context at start, call update_context at each step BEFORE the forward pass, pass context to run()
- `update_single(...)` — add context_encoding parameter
- `update_batch(...)` — add context_encodings parameter  
- `learn_combined_step(...)` — build context traces for each step in the trajectory, pass them during both policy and value updates
- `compute_value(...)` — add optional context parameter
- `compute_trajectory_values(...)` — build and pass context traces
- `per_step_entropy(...)` — build and pass context traces
- `chunkability_from_trajectory(...)` — build and pass context traces
- `chunking_index(...)` — uses per_step_entropy, should work if that's updated
- `test_accuracy()` — pass zero context (these are single-step evals, no trajectory)
- `test_loss()` — pass zero context
- `_test_trajectory_accuracy(...)` — pass zero context or build traces
- `_learn_off_policy(...)` — pass zero context (no trajectory structure)
- `_learn_on_policy(...)` — reset and update context at each step

The pattern for run() calls becomes:
```python
self.policy.run(
    inputs={
        self.policy_source_input: source_encoding,
        self.policy_target_input: target_encoding,
        self.policy_context_input: context_encoding  # NEW
    }
)
```

And for learn() calls:
```python
self.policy.learn(
    inputs={
        self.policy_source_input: [source_encoding],
        self.policy_target_input: [target_encoding],
        self.policy_context_input: [context_encoding],  # NEW
        self.target_node: [answer_encoding]
    }
)
```

### 4. Context trace construction during trajectory training

In `learn_combined_step` and `traverse_path`, the context trace must be built incrementally as the trajectory unfolds. The pattern is:

```python
self.reset_context()
for i, literal in enumerate(path[:-1]):
    # Update context BEFORE the forward pass at this step
    # (the agent "remembers" arriving at this state)
    if i > 0:
        # First step has zero context (trajectory initiation)
        self.update_context(path[i-1], decay=gamma * lambda_)
    
    context_encoding = self.context_trace.copy()
    
    # Now do forward pass / learning with this context
    ...
```

Wait — there's a subtlety about WHEN to update the context. The trace should reflect states visited BEFORE the current decision point. So at step 0 (state 15), context is zeros. At step 1 (state -10), context has decayed activation from state 15. At step 2 (state 3), context has activation from 15 (decayed twice) and -10 (decayed once). This means you update the context with the PREVIOUS state, not the current one, before each forward pass.

For batch training where you need all context vectors at once:
```python
def build_trajectory_contexts(self, path, gamma, lambda_):
    """
    Build context trace vectors for each step in a trajectory.
    
    Args:
        path: list[int] - full trajectory
        gamma: float - discount factor
        lambda_: float - trace decay parameter (chunkability-modulated)
    
    Returns:
        list[np.array] - context vector for each step in path
    """
    contexts = []
    trace = np.zeros(self.num_literals)
    decay = gamma * lambda_
    
    for i, literal in enumerate(path):
        # Store current context BEFORE updating with this state
        contexts.append(trace.copy())
        # Update trace with current state
        idx = self.literal_to_idx(literal)
        trace = decay * trace
        trace[idx] += 1.0
    
    return contexts
```

### 5. Decay rate

The decay rate should be `gamma * lambda_` where:
- `gamma` is the discount factor (same one used for TD(λ) value targets, default 0.99)
- `lambda_` is the chunkability-modulated parameter (`chunkability ^ lambda_exponent`)

This means you need to compute chunkability and lambda BEFORE building the context traces for training. The sequence in learn_combined_step becomes:

1. Compute chunkability from the trajectory (using previous step's context traces, or zero context for the first call)
2. Derive lambda = chunkability ^ exponent
3. Build context traces with decay = gamma * lambda
4. Train policy head with teacher targets + context traces
5. Compute value estimates with context traces
6. Compute TD(λ) value targets
7. Train value head with value targets + context traces

For the FIRST episode (and early episodes where chunkability ≈ 0), lambda ≈ 0, so decay ≈ 0, and the context is effectively zeros everywhere — the network behaves as if memoryless. This is correct: before the policy has learned anything, there's no sequence to chunk.

## Important notes

- The context trace is NOT a PsyNeuLink mechanism's internal state. It's a numpy array maintained by the PolicyNetwork class and passed as input data. PsyNeuLink doesn't need to know it's a trace.
- Make sure context_trace is RESET at the start of each trajectory (each episode). Don't let traces leak across episodes.
- The context input adds num_literals parameters to the hidden layer (the W_c matrix). This is the same cost as each existing input.
- For evaluation methods (per_step_entropy, chunking_index, etc.), you need to reconstruct the context traces for the trajectory being evaluated. Use build_trajectory_contexts with the CURRENT lambda value.
- Test that with decay=0, the network behaves identically to the version without context input (all context vectors are zeros).