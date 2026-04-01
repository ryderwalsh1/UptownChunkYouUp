## Lambda Experiment: Topology-Dependent Optimal TD(λ)

This directory contains the implementation for investigating how the optimal trace parameter λ depends on maze topology for a goal-conditioned recurrent agent trained with actor-critic TD(λ).

### Files

- **topology_generators.py**: Functions to generate maze topologies
  - Elementary topologies (Family A: corridors of varying length)
  - Procedural topologies (Family D: parameterized corridor/junction mazes)

- **topology_metrics.py**: Functions to compute structural descriptors
  - Node type classification (dead-ends, corridors, junctions)
  - Mean corridor length
  - Junction density
  - Corridor-to-decision ratio (R_corr/dec)

### Main Script

Run the experiment from the parent directory:

```bash
python lambda_experiment.py
```

This will:
1. Sweep over λ values: [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
2. Test on multiple topologies (corridors + procedural mazes)
3. Train for 300 episodes per configuration
4. Use 3 random seeds for each configuration
5. Log results to CSV with topology metrics

### Testing

Test the components before running the full experiment:

```bash
python test_lambda_experiment.py
```

### Expected Output

Results are saved to a timestamped directory: `lambda_experiment_results_YYYYMMDD_HHMMSS/`

- **results.csv**: Episode-level data with columns:
  - Configuration: topology, lambda, seed, episode
  - Performance: episode_reward, success, episode_length, optimality_ratio
  - Topology metrics: topo_mean_corridor_length, topo_junction_density, path_corr_dec_ratio
  - Training metrics: loss, policy_loss, value_loss, entropy_loss

- **config.json**: Experiment configuration

### Configuration

Edit the `config` dictionary in `lambda_experiment.py` to modify:
- `lambda_values`: List of λ values to test
- `seeds`: List of random seeds
- `num_episodes`: Training episodes per configuration
- `topologies`: List of topology names to test
- Hyperparameters: lr, gamma, entropy_coef, value_coef

### Topologies Available

**Elementary (Family A - Pure Corridors):**
- `corridor_short`: 5 nodes
- `corridor_medium`: 10 nodes
- `corridor_long`: 15 nodes
- `corridor_verylong`: 20 nodes

**Procedural (Family D - Mixed):**
- `proc_junction_heavy`: 8×8 with corridor=0.0 (many junctions)
- `proc_mixed`: 8×8 with corridor=0.5 (balanced)
- `proc_corridor_heavy`: 8×8 with corridor=0.9 (mostly corridors)

### Analysis

After running the experiment, analyze `results.csv` to:
1. Compute optimal λ* for each topology
2. Plot learning curves across λ values
3. Regress λ* on topology descriptors (especially R_corr/dec)
4. Create heatmaps of performance vs (λ, topology)
