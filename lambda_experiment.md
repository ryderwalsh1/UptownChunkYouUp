# Comprehensive experiment specification:

## Topology-dependent optimal (\lambda) in goal-conditioned GRU actor-critic TD((\lambda))

## 1. Objective

Determine how the optimal trace parameter (\lambda) depends on maze topology for a **goal-conditioned recurrent agent** trained with **actor-critic TD((\lambda))**.

The central claim being tested is:

> The best (\lambda) increases with the temporal separation between consequential decisions, and decreases with branching ambiguity.

In your setting, the agent receives:

* current state input
* goal state input

and uses a **GRU** to maintain latent state over time. Because the policy and value are recurrent, the study is really about how the external eligibility trace parameter (\lambda) should interact with:

* delayed reward,
* topology-induced decision spacing,
* memory demands,
* and branch ambiguity.

---

# 2. Main hypotheses

## H1. Corridor hypothesis

In mazes with long deterministic corridors and sparse decision points, larger (\lambda) will perform best.

Reason:

* successful outcomes should influence many upstream states,
* there is little ambiguity about which preceding states belong to the same successful behavioral unit,
* long-range temporal credit assignment is useful and relatively safe.

Prediction:

* best (\lambda) near the upper range, often (0.8) to (1.0).

---

## H2. Junction-density hypothesis

In mazes with frequent branch points / intersections, smaller (\lambda) will perform best.

Reason:

* the most important causal events are local branch choices,
* long traces will over-credit pre-junction states and smear credit across states that did not uniquely determine success,
* sharper local bootstrapping helps isolate decisive actions.

Prediction:

* best (\lambda) shifts downward as junction density and branch ambiguity increase.

---

## H3. Mixed-topology hypothesis

In mazes containing long corridors between sparse but consequential decisions, intermediate-to-high (\lambda) will perform best.

Reason:

* credit must travel far enough to link sparse decisions,
* but not so diffusely that branch-responsibility becomes blurred.

Prediction:

* best (\lambda) tracks the average corridor length between critical decisions.

---

## H4. Topology law

The optimal (\lambda^*) is predictable from graph statistics, especially:

* mean corridor length
* junction density
* branch ambiguity
* number of decision points along shortest path
* non-decision steps per decision step

A particularly important candidate predictor is:

[
R_{\text{corr/dec}} = \frac{\text{expected non-decision steps on optimal path}}{\text{expected decision steps on optimal path}}
]

Prediction:

* (\lambda^*) increases monotonically with (R_{\text{corr/dec}}).

---

## H5. Recurrent-memory interaction

Because the policy/value network is a GRU, part of the temporal burden is already handled internally. Therefore:

* the absolute value of optimal (\lambda) may be lower than in a feedforward agent,
* but the topology trend should remain.

So the key result is not “(\lambda=1) always wins in corridors,” but whether the **ordering across topologies** is stable.

---

# 3. RL formulation

## 3.1 Task type

Goal-conditioned episodic navigation.

Each episode consists of:

* a start state (s_0),
* a goal state (g),
* a graph-structured maze (M),
* an episode terminates on reaching goal or time limit.

The agent observes:

* current state (s_t),
* goal state (g),
* optionally action mask for legal moves.

The GRU hidden state (h_t) evolves over time.

---

## 3.2 Policy and value functions

Actor-critic with shared recurrent trunk:

[
h_t = \mathrm{GRU}(h_{t-1}, x_t)
]

where (x_t) is the observation encoding of current state and goal.

Outputs:

* policy logits (\pi(a_t \mid h_t))
* state-value (V(h_t))

The FastNetwork class in fast.py is the GRU you will use, with parameter prospection_head=False.

---

## 3.3 TD((\lambda)) target

The FastNetworkTrainer class in fast.py contains all of the training algorithm you need.

---

# 4. Environment families

You should include both **elementary motifs** and **procedurally generated full mazes**.

## 4.1 Elementary topology families

These provide interpretable anchor cases.

### Family A: Pure corridor

A linear chain:
[
A \to B \to C \to \dots \to G
]

Parameters:

* corridor length (L \in {3, 5, 7, 10, 15, 20})

Use:

* fixed start at one end
* goal at the other end

Purpose:

* isolate long-range credit assignment without branch ambiguity.

Prediction:

* performance should improve with larger (\lambda) as (L) increases.

---

### Family B: Single branch

A corridor leading to one branch:
[
A \to B \to C \to {D,F}
]
then one branch reaches goal, one does not.

Parameters:

* pre-branch corridor length
* post-branch branch lengths
* one correct branch vs one wrong branch

Purpose:

* isolate one decisive choice following passive traversal.

Prediction:

* moderate to small (\lambda) may outperform very large (\lambda) when branch credit must remain localized.

---

### Family C: Intersection / crossing

A central intersection with multiple approach directions and exits.

Parameters:

* degree of central node
* number of arms
* arm lengths
* goal placement

Purpose:

* high local ambiguity and high branching centrality.

Prediction:

* lower (\lambda) preferred relative to long corridors.

---

### Family D: Corridor-chain-with-periodic-junctions

A sequence of corridor segments separated by junctions.

Parameters:

* number of junctions
* average corridor length between junctions
* branching factor per junction
* fraction of dead branches

Purpose:

* tests the mixed-topology regime directly.

Prediction:

* (\lambda^*) rises with corridor segment length.

---

### Family E: Trees

Directed or undirected tree mazes.

Parameters:

* depth
* branching factor
* goal depth
* dead-end count

Purpose:

* frequent hierarchical decisions without loops.

Prediction:

* lower (\lambda) than corridor-heavy environments.

---

## 4.2 Procedural grid-based families

You also want full spatial mazes that vary continuously in corridor-likeness.

Represent them using MazeGraph in corridors.py, varying the seed the corridor parameter from 0 to 1

---

## 4.3 Continuous topology sweep parameters

For procedural families, define a continuous parameterization rather than only discrete classes.

Suggested controls:

* (p_{\text{wall}}): wall density
* (p_{\text{junction}}): probability of opening side branches
* (L_{\text{target}}): target corridor length
* (p_{\text{loop}}): probability of adding cycle-forming edges
* (b): target local branching factor

Then sample many mazes across a grid of these parameters.

---

# 5. Topology descriptors to compute for every maze

These are critical. Do not treat the maze family label as the only structural variable.

For each graph (G=(V,E)), compute:

## 5.1 Node-type counts

Classify each state by degree:

* dead-end: degree 1
* corridor node: degree 2
* junction: degree (\ge 3)

Store:

* fraction of each type
* raw counts

---

## 5.2 Mean corridor length

Compress the graph into a junction-dead-end skeleton:

* collapse maximal sequences of degree-2 nodes into corridor segments.

Then compute:

* mean corridor length
* median corridor length
* corridor length variance
* max corridor length

This is one of your most important statistics.

---

## 5.3 Junction density

[
\rho_J = \frac{# \text{junction nodes}}{# \text{nonterminal nodes}}
]

---

## 5.4 Decision count on shortest path

For a given start-goal pair, compute:

* shortest path length
* number of junction nodes encountered on shortest path
* number of corridor nodes encountered on shortest path

Then define:

[
R_{\text{corr/dec}} = \frac{# \text{corridor steps on shortest path}}{# \text{junction decisions on shortest path} + \epsilon}
]

Use a tiny (\epsilon) only to avoid divide-by-zero.

This is perhaps the cleanest task-specific topology scalar.

---

## 5.5 Branch ambiguity

At each junction on the shortest path, compute:

* number of legal outgoing actions
* number of branches whose shortest-path-to-goal cost is close to optimal

Possible metric:

[
A_{\text{branch}} = \text{average over decision nodes of } \left(# \text{plausible outgoing actions}\right)
]

where “plausible” means within some distance threshold of the best branch.

This distinguishes:

* easy branch point with one obviously good exit
  from
* ambiguous hub with several near-good continuations.

---

## 5.6 Dead-end burden

Measure how costly a wrong turn is.

For each decision node:

* compute the expected extra path length incurred by choosing a non-optimal branch.

Aggregate:

* mean wrong-turn penalty
* max wrong-turn penalty

Large wrong-turn penalty may make branch credit more important.

---

## 5.7 Loop density

Number of independent cycles, or edge surplus relative to a tree.

Can use:
[
\text{loop density} = \frac{|E| - |V| + C}{|V|}
]
where (C) is number of connected components.

---

## 5.8 Path degeneracy

Number of shortest paths from start to goal, or entropy over near-shortest paths.

If many nearly equivalent routes exist, very long traces may become less topology-specific.

---

## 5.9 Betweenness concentration

Compute node betweenness centrality and summarize:

* mean
* max
* Gini coefficient or concentration

This identifies whether the maze has critical bottlenecks.

---

# 6. Experimental variables

## 6.1 Primary independent variable

[
\lambda \in {0.0, 0.1, 0.2, \dots, 1.0}
]

You may optionally refine around high-performing regions:

* coarse sweep first
* fine sweep later, e.g. increments of 0.05 near the optimum

---

## 6.2 Secondary hyperparameters

Because (\lambda) interacts with learning dynamics, you should also tune a small grid over:

* actor learning rate
* critic learning rate
* entropy regularization coefficient
* value loss coefficient

At minimum, tune:

* a shared base learning rate over 3–5 values

Best practice:

* for each (\lambda), choose best learning rate based on validation mazes from the same family
* then evaluate that pair on held-out mazes

Otherwise you risk attributing step-size mismatch to (\lambda).

---

## 6.3 Controlled constants

Keep fixed unless explicitly sweeping:

* discount (\gamma)
* GRU size
* network architecture
* optimizer
* max episode length
* reward function
* exploration policy / entropy bonus schedule
* observation encoding
* action mask availability

---

# 7. Observation and action specification

## 7.1 State representation

Since the agent is goal-conditioned, the observation should include:

* encoding of current state as one-hot node id
* encoding of goal state as one-hot node id

To keep topology effects clean, use the same input representation across all maze families.

---

## 7.2 Goal representation

Goal should be explicit and consistent across environments:

* one-hot goal node id for abstract graph tasks

---

## 7.3 Action space

Use fixed action semantics where possible.

For abstract graph tasks:

* directional actions in grids

Illegal actions lead to no state transition (if agent tries to navigate through a wall)

---

# 8. Reward design

You should test two reward regimes.

## Regime 1: Sparse terminal reward

* reward (+1) on reaching goal
* reward (0) otherwise
* episode terminates at goal or time limit

This is the purest setting for studying (\lambda).

---

## Regime 2: Sparse reward plus step penalty

* reward (+1) on reaching goal
* step penalty (-c), e.g. (-0.01)
* optional failure timeout penalty

This encourages efficient navigation and can reveal whether (\lambda) changes optimal-path shaping.

Do not use heavy shaping toward the goal initially, because it can wash out topology effects.

---

# 9. Train protocol

## 9.1 Start-goal sampling

For each maze instance:

* define a set of start-goal pairs
* sample pairs during training

Important:

* sample pairs so that difficulty varies
* record shortest-path length and decision count for each pair

You may want stratified evaluation by start-goal difficulty.

---

## 9.2 Seeds

For every ((\text{maze family}, \lambda, \text{hyperparameter setting})), train across multiple random seeds.

Minimum:

* 10 seeds
  Better:
* 20 seeds for final curves

Sources of randomness:

* initialization
* environment sampling
* start-goal sampling
* action sampling

---

# 10. Training procedure

For each training run:

1. Sample a maze instance from total set.
2. Sample start and goal.
3. Roll out an episode with the GRU actor.
4. Store sequence:
   [
   (o_t, h_t, a_t, r_t, V_t, \log \pi_t)
   ]
5. Compute TD residuals and (\lambda)-advantages:
   [
   \delta_t = r_t + \gamma V_{t+1} - V_t
   ]
   [
   A_t^{(\lambda)} = \sum_{l\ge 0} (\gamma \lambda)^l \delta_{t+l}
   ]
6. Update actor and critic.
7. Repeat.

Use the same rollout length / truncation scheme across all conditions.

---

# 11. Core evaluation metrics

## 11.1 Learning efficiency

### a. Episodes-to-threshold

Number of episodes needed to reach:

* 50% success
* 80% success
* 95% success

### b. Steps-to-threshold

Same, but in environment steps.

### c. Area under learning curve

AUC of success rate or return over training.

This is often the best single summary of learning speed + stability.

---

## 11.2 Final policy quality

On held-out test mazes, measure:

* success rate
* average return
* mean path length
* optimality ratio:
  [
  \frac{\text{agent path length}}{\text{shortest path length}}
  ]
* timeout rate

---

## 11.3 Decision quality at junctions

This is central.

For each junction state encountered during evaluation:

* whether the selected action lies on a shortest path to goal
* whether the selected action is optimal
* action entropy at the junction
* policy margin between best and second-best actions

Aggregate:

* junction decision accuracy
* action-gap / policy-gap at decision nodes
* wrong-turn rate

Prediction:

* low-to-moderate (\lambda) should sharpen junction choices in branch-heavy mazes.

---

## 11.4 Corridor traversal reliability

For corridor segments, measure:

* probability of traversing the full corridor without deviation
* expected progress per step toward next junction or goal
* stall / oscillation rate

Prediction:

* higher (\lambda) should help in long corridors if the delayed credit is the main issue.

---

# 12. Credit-assignment diagnostics

These are the most important “mechanistic” statistics.

## 12.1 Effective backward credit distance

For each successful episode, compute the magnitude of actor and critic training signal assigned to each timestep.

One useful summary:

[
D_{\text{eff}} = \frac{\sum_{k=0}^{T-1} k \cdot |\Delta_{T-1-k}|}{\sum_{k=0}^{T-1} |\Delta_{T-1-k}|}
]

where (\Delta_t) can be:

* (|A_t^{(\lambda)}|),
* (|\delta_t|) weighted by the trace,
* or actual gradient norm contribution from timestep (t).

Interpretation:

* larger values mean reward influence reaches farther backward.

Plot:

* (D_{\text{eff}}) vs (\lambda)
* (D_{\text{eff}}) vs mean corridor length

---

## 12.2 Credit by node type

Partition steps into:

* corridor-node visits
* junction-node visits
* dead-end-node visits
* goal-node predecessor visits

Then compute total absolute credit assigned to each category:
[
C_{\text{corr}}, C_{\text{junc}}, C_{\text{dead}}
]

Useful ratios:
[
\frac{C_{\text{junc}}}{C_{\text{corr}}}, \qquad
\frac{C_{\text{dead}}}{C_{\text{corr}}}
]

Prediction:

* in branch-heavy mazes, good (\lambda) should produce relatively concentrated credit on junctions,
* in corridor-heavy mazes, good (\lambda) should spread credit more broadly through corridor states.

---

## 12.3 Decision localization score

For each successful branch resolution, define:

* the critical junction timestep (t_j)
* preceding corridor timesteps (t_j-k, \dots, t_j-1)

Then compute:
[
L_{\text{decision}} = \frac{|\Delta_{t_j}|}{\sum_{m=1}^{K} |\Delta_{t_j-m}| + \epsilon}
]

High value:

* credit is sharply localized at the decision point.

Low value:

* credit is diffused across earlier corridor states.

Prediction:

* optimal (\lambda) in junction-heavy mazes should yield larger (L_{\text{decision}}).

---

## 12.4 Wrong-branch credit contamination

At branch points, quantify how much successful return reinforces states upstream of a wrong-turn opportunity without sufficiently separating the competing actions.

Two useful measures:

### a. Action-gap at decision nodes

[
G(s) = \pi(a^* \mid s) - \max_{a \ne a^*} \pi(a \mid s)
]
or use Q-like advantage margin if available.

### b. Upstream-over-local credit ratio

Compare total credit assigned before the junction to total credit at the junction itself.

If this ratio becomes too large in branch-heavy mazes, it suggests trace smearing.

---

# 13. Value-structure diagnostics

Even in actor-critic, the critic should reflect topology-sensitive learning.

## 13.1 Value-distance correlation

For each state, compute shortest-path distance to goal (d(s,g)). Then evaluate:

* Spearman correlation between (V(s)) and (-d(s,g))
* separately for corridor and junction states

A good learned critic should increasingly align with graph-theoretic distance.

---

## 13.2 Corridor value ramp smoothness

Along optimal corridor segments, evaluate:

* monotonicity of (V) toward goal
* average slope per step
* variance of slope

Prediction:

* higher (\lambda) should produce cleaner corridor ramps earlier in training.

---

## 13.3 Junction value sharpness

At a junction, compare value estimate / policy preference for the correct continuation vs wrong branches.

This is related to decision accuracy but viewed through learned value structure.

---

# 14. Recurrent-state diagnostics

Because your agent is a GRU, you should examine hidden-state structure too.

## 14.1 Hidden-state separability by local topology

Collect (h_t) during evaluation and label each timestep by:

* corridor
* junction
* dead-end
* goal-near

Then assess whether hidden states cluster by node type.

A simple diagnostic:

* linear probe to predict node type from (h_t)

This tells you whether the GRU internally represents decision context.

---

## 14.2 Hidden-state sensitivity to goal

For the same current state but different goals, compare hidden state differences. This matters because some junctions are only consequential relative to the goal.

---

## 14.3 Hidden-state compression along corridors

In long corridors, does the GRU hidden state evolve smoothly and predictably, or does it fluctuate?
High-performing (\lambda) may correspond to a smoother internal trajectory in corridor-dominated tasks.

---

# 15. Statistical analysis plan

## 15.1 Per-maze optimal lambda

For each test maze or family/parameter bucket, compute:
[
\lambda^* = \arg\max_{\lambda} \text{PerformanceMetric}(\lambda)
]
where the performance metric is preferably:

* AUC, or
* steps-to-threshold, or
* final success rate

You may report several versions.

---

## 15.2 Regression of (\lambda^*) on topology

Fit a model:
[
\lambda^* \sim \beta_0

* \beta_1 (\text{mean corridor length})
* \beta_2 (\text{junction density})
* \beta_3 (\text{branch ambiguity})
* \beta_4 (\text{dead-end penalty})
* \beta_5 (\text{loop density})
  ]

You can begin with linear regression, then consider nonlinear fits if needed.

A very interpretable reduced model is:
[
\lambda^* \sim \beta_0 + \beta_1 R_{\text{corr/dec}} + \beta_2 A_{\text{branch}}
]

This is likely the most important analysis in the whole study.

---

## 15.3 Confidence intervals

For every major metric, report:

* mean
* 95% confidence interval across seeds and test mazes

For (\lambda^*), use bootstrap confidence intervals if possible.

---

# 16. Recommended figure set

## Figure 1

Example topology motifs:

* corridor
* single branch
* intersection
* mixed corridor-junction maze

## Figure 2

Learning curves across (\lambda) for elementary families.

## Figure 3

Heatmap:

* x-axis = (\lambda)
* y-axis = topology parameter (e.g. mean corridor length or junction density)
* color = AUC or final success

This should visually reveal topology-dependent (\lambda).

## Figure 4

Optimal (\lambda^*) vs topology descriptor:

* mean corridor length
* junction density
* (R_{\text{corr/dec}})

## Figure 5

Credit concentration diagnostics:

* junction/corridor credit ratio vs (\lambda)
* effective backward credit distance vs (\lambda)

## Figure 6

Decision quality at junctions:

* junction accuracy
* policy action-gap
* wrong-turn rate

## Figure 7

Value-structure plots:

* value ramps along corridors
* junction sharpness

---

# 17. Minimal set of ablations

These are worth doing because otherwise results may be overinterpreted.

## Ablation A: No recurrence

Replace GRU with feedforward network using same inputs.

Purpose:

* determine whether the topology-(\lambda) relationship is specific to recurrent memory.

## Ablation B: Sparse vs step-penalty reward

Purpose:

* determine whether reward shaping changes the topology dependence.

## Ablation C: Fixed start-goal distance bins

Evaluate separately for:

* short
* medium
* long shortest-path distances

This avoids mixing topology with raw episode length.

## Ablation D: Entropy regularization strength

If policy remains too stochastic, junction effects may be blurred.

---

# 18. Practical implementation details

## 18.1 Maze balancing

Do not let one family dominate by graph size alone.

When comparing families, either:

* match graph size,
* or report size as a covariate.

Otherwise corridor mazes may look like they favor high (\lambda) simply because they are longer.

---

## 18.2 Normalize difficulty

When possible, match or stratify by:

* shortest-path length
* number of decision points
* dead-end burden

This is important. “Corridor-like” should not automatically mean “easier.”

---

## 18.3 Time limit

Set timeout as:
[
T_{\max} = c \cdot d_{\text{shortest}}
]
with fixed multiplier (c), e.g. 2 to 4.

This keeps evaluation fair across maze sizes.

---

## 18.4 Goal-conditioned sampling

To avoid trivial memorization:

* sample many start-goal pairs per maze,
* include pairs with different relative positions and difficulty.

---

# 19. Canonical primary metrics to emphasize

If you want the cleanest final story, prioritize these six:

1. **AUC of learning curve**
2. **Steps-to-90%-success**
3. **Final success rate**
4. **Junction decision accuracy**
5. **Effective backward credit distance**
6. **Junction-to-corridor credit ratio**

And then the single most important relationship:

7. **Optimal (\lambda^*) as a function of (R_{\text{corr/dec}})**

---

# 20. Expected qualitative outcomes

Here is the pattern I would expect.

## Pure corridor

* best (\lambda): high
* credit spreads far backward
* value ramps become smooth
* junction metrics irrelevant

## Single-branch maze

* best (\lambda): moderate
* enough trace to connect reward across pre-branch corridor
* not so much that branch responsibility is blurred

## Dense intersection / open grid

* best (\lambda): lower
* sharper junction localization
* faster action-gap formation at decisions

## Corridor-junction hybrid

* best (\lambda): intermediate to high, scaling with corridor length between decisions

## Loopy ambiguous mazes

* best (\lambda): lower to moderate
* long traces may diffuse credit across route-equivalent or near-equivalent paths

---

# 21. One clean formal statement of the study

You can frame the whole experiment as:

> We study how the optimal TD trace parameter (\lambda) for a goal-conditioned recurrent actor-critic depends on the graph topology of navigation tasks. We hypothesize that (\lambda^*) increases with the average number of non-decision steps between consequential decisions and decreases with branching ambiguity. To test this, we generate maze families spanning corridor-heavy to junction-dense regimes, sweep (\lambda), and evaluate not only learning performance but also topology-aware credit-assignment diagnostics.

---

# 22. Suggested experiment table

A concrete first pass:

* **Families:** corridor, single-branch, intersection, corridor-chain-with-junctions, trees, procedural mazes
* **Instances per family:** 80% training episodes, 20% evaluation episodes
* **Start-goal pairs per instance:** 20 sampled
* **(\lambda) sweep:** 0.0 to 1.0 by 0.1
* **Learning-rate sweep:** 3 values
* **Seeds:** 10
* **Reward regimes:** sparse, sparse + step penalty
* **Primary evaluation metric:** AUC and steps-to-90%-success

That is already a substantial but manageable study.

---

# 23. Recommended data logging schema

For every episode, log:

* run id
* seed
* family
* maze instance id
* topology descriptors
* start state
* goal state
* shortest-path length
* decision count on shortest path
* (\lambda)
* learning rate
* episode return
* success/failure
* path length
* optimality ratio
* per-step:

  * state type
  * action
  * reward
  * value
  * TD error
  * (\lambda)-advantage
  * policy entropy
  * hidden-state norm

This will make post hoc analysis much easier.

---

# 24. Final recommendation on interpretation

Do not interpret the result as “corridors need long-term credit, junctions need short-term credit” in a purely verbal way.

Interpret it more precisely as:

* (\lambda) controls the temporal spread of training signal,
* maze topology determines how often causally decisive events occur,
* the best (\lambda) is the one whose credit horizon matches the spacing and ambiguity of those decisive events.