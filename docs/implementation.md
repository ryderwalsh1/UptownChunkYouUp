# Cognitive Control, Episodic Retrieval, and Lambda-Modulated Chunking

## Purpose

This document specifies the implementation of a cognitive architecture with:

* a fast impulsive policy network,
* a slower episodic-memory-informed policy network,
* a meta-controller that decides whether to defer to slow processing,
* a conflict map storing long-term estimates of control demand,
* a lambda modulation mechanism that determines the backup horizon for eligibility traces,
* a chunking mechanism defined by sustained low control demand and expressed through longer-horizon backup rather than explicit macro-actions.

The design is intended to be cognitively motivated while remaining implementable in a neural-network-based RL setting.

---

## 1. High-Level Architecture

### 1.1 Modules

1. **Fast network**

   * GRU policy network.
   * Produces a fast action distribution from current and goal states.
   * Represents intuitive / habitual / low-cost processing.

2. **Slow network**

   * Episodic-memory-informed policy network.
   * Uses a PsyNeuLink episodic memory module.
   * Represents slower, more flexible, memory-guided processing.

3. **Meta-controller**

   * A control-value network.
   * Chooses between control actions such as trusting fast processing vs deferring to slow processing.
   * Operates over meta-actions, not environment actions.

4. **Conflict map**

   * Per-state long-term map storing EMA of divergence between network output distributions.
   * Used as a long-timescale estimate of where control is typically needed.

5. **Lambda modulator**

   * Maps control demand to an eligibility-trace parameter.
   * High control demand -> low lambda.
   * Low control demand -> high lambda.

---

## 2. Final Action Arbitration

### Choice

**Sample-based arbitration** between fast and slow systems.

Let the meta-controller output values over control actions:

* `use_fast`
* `use_slow`

These values define a policy over controller choices. The selected control action determines which policy supplies the final environment action.

### Formalization

Let:

* `z_f(s)` = fast logits
* `z_s(s, m)` = slow logits conditioned on memory retrieval
* `Q_ctrl(x, use_fast)` and `Q_ctrl(x, use_slow)` = meta-controller values
* `x` = controller input features

Then define:

* `pi_ctrl(u | x) = softmax(Q_ctrl(x, ·))`
* sample `u_t ~ pi_ctrl(u | x_t)`

If `u_t = use_fast`, action is sampled from `softmax(z_f)`.
If `u_t = use_slow`, action is sampled from `softmax(z_s)`.

This preserves exploratory arbitration rather than imposing a deterministic gate.

---

## 3. Fast Network

### 3.1 Role

The fast network is the impulsive / habitual policy.
It should:

* act from compact recent context,
* produce low-latency action proposals,
* gradually absorb practiced structure,
* become confident in familiar regions.

### 3.2 Architecture

**Gated Recurrent Unit** 

Inputs  include:

* current state embedding
* goal state embedding

Outputs:

* action logits `z_f`
* optional state value `V_f`

### 3.3 Training

Train first, before adding slow and control components. Use standard RL training (e.g. actor-critic / TD(𝜆)-based objective). This pretraining stage gives the system an initial habitual policy before control arbitration is introduced to add stability to training.

---

## 4. Slow Network

### 4.1 Role

The slow network is the flexible, memory-based policy.
It does not simply represent “a larger policy”; it represents deliberate access to episodic support.

### 4.2 Memory Representation

Stored memory item type:

* **state -> action**

The episodic system stores state-action associations and retrieves relevant prior items using the PsyNeuLink episodic memory module.

### 4.3 Retrieval

Retrieval is always available, but whether its resulting policy is used is determined by the controller through sampled arbitration.

Because memory access is allowed continuously but carries control cost through use of the slow branch, the architecture supports graded familiarity while preserving explicit meta-control.

### 4.4 Outputs

The slow network produces:

* action logits `z_s`
* optional slow value estimate `V_s`

These logits are conditioned on:

* current state embedding,
* retrieved episodic content.

---

## 5. Meta-Controller

### 5.1 Role

The meta-controller estimates the value of allocating additional computation.
It is a **meta-level value network**, not a policy over environment actions.

### 5.2 Control Actions

At minimum:

* `use_fast`
* `use_slow`

Optional future extensions:

* `blend`
* `retrieve_only`
* `deliberate_longer`

### 5.3 Inputs

Selected controller inputs:

* fast entropy,
* KL divergence between fast and slow policies,
* conflict map value for current state,
* state embedding.

So the controller input can be written as:

`x_ctrl = [embed(s_t), H_fast, KL(fast || slow), C_map(s_t)]`

### 5.4 Output

The controller outputs meta-values:

* `Q_ctrl(x_ctrl, use_fast)`
* `Q_ctrl(x_ctrl, use_slow)`

These values define the control policy via softmax.

### 5.5 Interpretation

The controller estimates the expected utility of deferring to slow processing under the objective:

**reward + efficiency - control cost**

So slow processing should be selected only when its expected downstream benefit outweighs its added control cost.

### 5.6 Training

Training method: **reinforcement learning at the meta-level**.

The controller is trained from returns induced by its control choices. Conceptually, it learns:

* where slow processing improves reward,
* where slow processing reduces path length / increases efficiency,
* where slow processing is not worth its cost.

---

## 6. Conflict, Confidence, and Their Separation

### 6.1 Confidence

Confidence is tied primarily to the fast network’s decisiveness.
Operational proxy:

* low entropy of `softmax(z_f)`

### 6.2 Conflict

Conflict is not identical to uncertainty.
Here, conflict is defined primarily by **disagreement between systems**.
Operational proxy:

* KL divergence between fast and slow policy distributions

### 6.3 Why separate them

A state can be:

* high confidence, low conflict: fast system decisive and slow agrees,
* low confidence, low conflict: fast uncertain but slow not substantially different,
* high confidence, high conflict: fast decisive but slow strongly disagrees,
* low confidence, high conflict: both ambiguity and disagreement.

The architecture therefore treats:

* **confidence** as a local property of the fast system,
* **conflict** as a cross-system property relevant for control.

---

## 7. Conflict Map

### 7.1 Stored Quantity

**The conflict map stores an exponential moving average of fast-slow disagreement**.

Per-state value:

* `C_map(s)`

This is a long-timescale representation of how often a state tends to require flexible processing.

### 7.2 Granularity

Granularity choice:

* **per-state**

### 7.3 Update Rule

Use an exponential moving average:

`C_map(s_t) <- (1 - alpha_c) * C_map(s_t) + alpha_c * KL_t`

where:

* `KL_t = KL(pi_f(.|s_t) || pi_s(.|s_t, m_t))`
* `alpha_c` is a small update rate

A small `alpha_c` ensures that the map reflects long-term structure rather than episode-level noise.

### 7.4 Function

The conflict map is the main long-term proxy for learned control demand.
It stabilizes the controller and the lambda modulator by separating enduring structure from transient fluctuations.

---

## 8. Lambda Modulation

### 8.1 Principle

Lambda should be modulated by the **controller output**, not directly by raw entropy.

Selected design:

* lambda is a function of control demand
* control demand comes from the meta-controller policy/value state

### 8.2 Functional Form

Use a **superlinear power-law mapping**.

Let `p_slow = pi_ctrl(use_slow | x_ctrl)`.
Then define:

`lambda_t = (1 - p_slow)^beta`

with `beta > 1`, so higher expected need for slow processing implies lower lambda, and the mapping is superlinear over the unit interval.

Equivalent monotone variants are acceptable so long as:

* low control demand -> high lambda,
* high control demand -> low lambda.

### 8.3 Interpretation

* High lambda: broad temporal credit assignment, eligible for chunk-like learning.
* Low lambda: local credit assignment, granular control near difficult points.

### 8.4 Stability Mechanism

Selected stabilization choice:

* **separate short-term vs long-term signals**

Implement this explicitly:

* short-term signal: current controller output or current disagreement
* long-term signal: conflict map value

Recommended usage:

* controller may depend on both,
* lambda should be dominated by the long-term signal, with only mild short-term modulation.

For example:

`d_t = w_long * C_map(s_t) + w_short * p_slow`

with `w_long > w_short` and `d_t` constrained to `[0, 1]`, then:

`lambda_t = (1 - d_t)^beta`

with `beta > 1`.

This avoids lambda collapsing because of temporary spikes caused by ongoing target shifts while preserving the same superlinear control-to-lambda relationship.

---

## 9. Chunking

### 9.1 Definition

Chunking is defined by **high-lambda segments**.

A segment is chunk-eligible when control demand remains low enough that lambda stays elevated across successive states.

### 9.2 Implementation Choice

Selected chunking effect:

* **increase value backup horizon only**

That is, chunking is not implemented as explicit macro-actions or state skipping.
Instead, chunking is expressed through longer-horizon eligibility traces and temporal compression in credit assignment.

### 9.3 Criterion

A practical criterion:

* if `lambda_t > tau_lambda` for a sufficient consecutive span, that span behaves as a chunk candidate.

### 9.4 Interpretation

This treats chunking as a continuous learning phenomenon:

* repeated low-control traversal increases effective temporal compression,
* the system backs up information farther through time,
* the path becomes functionally more “unitized” without requiring explicit discrete chunk creation.

---

## 10. Objective Function

### 10.1 Global Objective

Selected objective:

**maximize reward + efficiency - control cost**

### 10.2 Terms

1. **Reward term**

   * task success / return

2. **Efficiency term**

   * shorter path length,
   * lower deliberation time,
   * lower decision latency,
   * fewer unnecessary control interventions

3. **Control cost term**

   * penalizes use of slow processing / meta-control allocation

### 10.3 Consequence

The controller should not invoke slow processing whenever it is merely available.
It should invoke it only when doing so improves expected objective enough to justify the cost.

---

## 11. Training Schedule

### 11.1 Chosen Training Strategy

**Pretrain fast -> add slow & control**

### 11.2 Stage 1: Fast pretraining

Train fast network alone until it develops a usable baseline policy.
This creates an impulsive system worth arbitrating against.

### 11.3 Stage 2: Add slow network

Enable episodic retrieval and train the slow branch to leverage memory-conditioned information. The slow system will be pre-populated with correct actions, as if recalling a teacher's instructions. Possible future implementation includes populating episodic memory on the fly and decaying memory traces.

Train the meta-controller over `use_fast` vs `use_slow` choices using RL.
The controller now learns the expected value of extra processing under the full objective.

### 11.4 Conflict map updates

The conflict map can begin updating once both fast and slow policies exist, since disagreement is undefined before then.

---

## 12. Exploration

### 12.1 Chosen Exploration Strategy

**Softmax sampling**

Applies at two levels:

1. control action sampling from `pi_ctrl`
2. environment action sampling from selected policy branch

This preserves graded exploration in both cognitive allocation and action selection.

---

## 13. Suggested Per-Step Computation

At each timestep `t`:

1. Encode state/history with fast GRU -> `z_f`
2. Retrieve episodic content through PsyNeuLink memory
3. Compute slow policy logits `z_s`
4. Compute controller features:

   * state embedding
   * fast entropy
   * KL(fast || slow)
   * conflict map value
5. Compute controller meta-values and control policy
6. Sample control action `u_t`
7. Use selected branch to sample environment action `a_t`
8. Step environment
9. Update RL signals
10. Update conflict map with current KL disagreement
11. Compute control-demand summary and resulting `lambda_t`
12. Apply lambda-modulated eligibility-trace update

---

## 14. Core Equations

### 14.1 Fast and slow policies

`pi_f(a|s_t) = softmax(z_f(s_t))`

`pi_s(a|s_t, m_t) = softmax(z_s(s_t, m_t))`

### 14.2 Fast entropy

`H_t = - sum_a pi_f(a|s_t) log pi_f(a|s_t)`

### 14.3 Conflict signal

`KL_t = KL(pi_f(.|s_t) || pi_s(.|s_t, m_t))`

### 14.4 Conflict map update

`C_map(s_t) <- (1 - alpha_c) C_map(s_t) + alpha_c KL_t`

### 14.5 Meta-control policy

`Q_ctrl = f_ctrl(embed(s_t), H_t, KL_t, C_map(s_t))`

`pi_ctrl(u|x_t) = softmax(Q_ctrl(x_t, ·))`

### 14.6 Lambda modulation

`d_t = w_long C_map(s_t) + w_short pi_ctrl(use_slow | x_t)`

`lambda_t = (1 - d_t)^beta`

with `w_long > w_short`.

---

## 15. Design Intent Summary

### Fast network

Habitual, intuitive, context-driven policy.

### Slow network

Memory-guided flexible policy using episodic retrieval.

### Meta-controller

Connectionist value network over control allocation.
Learns whether slow processing is worth its cost.

### Conflict map

Long-term per-state estimate of fast-slow disagreement.
Tracks where control is historically needed.

### Lambda

Derived from control demand, mainly long-term conflict structure.
High lambda corresponds to chunk-eligible low-control regions.

### Chunking

Not explicit macro-action creation.
Chunking is expressed as increased backup horizon through sustained high lambda.

---

## 16. Immediate Implementation Priorities

1. Implement fast GRU policy and pretrain it.
2. Implement slow policy with PsyNeuLink episodic retrieval.
3. Compute per-step fast entropy and fast-slow KL.
4. Add per-state conflict map with EMA updates.
5. Implement controller as a meta-Q/value network over `{use_fast, use_slow}`.
6. Train controller with RL under objective `reward + efficiency - control cost`.
7. Drive lambda from controller-derived control demand, dominated by long-term conflict map structure.
8. Evaluate whether high-lambda regions align with intuitively chunkable paths.

---

## 17. Open Future Extensions

Not part of the current design, but natural next steps:

* learned conflict map generalization across similar states,
* explicit control cost scheduling,
* On the fly episodic memory population
* explicit retrieval-quality feature into the controller,
* multi-action control space beyond `use_fast` / `use_slow`,
* chunk boundary inference as a learned latent variable.
