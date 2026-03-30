"""
Interactive Agent Inspection Script

Load a trained agent and inspect its outputs for specific input states.
Shows all internal computations: fast network, slow memory, controller decisions, etc.

Usage Examples:
---------------

1. Interactive mode (default):
   python inspect_agent.py

   Then use commands:
   > state 0 goal 56        # Inspect specific state-goal pair
   > random                 # Inspect random state-goal pair
   > endpoint               # Inspect random endpoint-to-endpoint pair
   > stats                  # Show agent statistics
   > quit                   # Exit

2. Batch mode (test multiple samples):
   python inspect_agent.py --mode batch --num-samples 20

3. Load different checkpoint:
   python inspect_agent.py --checkpoint checkpoints/agent_stage1.pt

4. Different maze configuration:
   python inspect_agent.py --length 10 --width 10 --corridor 0.7

Output Explanation:
-------------------

The inspection shows the full cognitive architecture computation:

📍 POSITIONS:
   - Current state and goal positions in the maze
   - Optimal path length between them
   - Optimal next action (from slow memory's shortest path)

🧠 FAST NETWORK:
   - Value estimate: predicted future return
   - Entropy: measure of policy confidence (low = high confidence)
   - Top-5 actions: most probable actions from fast habitual policy

💾 SLOW MEMORY:
   - Retrieved action: optimal action from episodic memory lookup
   - Top-5 actions: memory retrieval is deterministic (returns one action)

⚔️ CONFLICT ANALYSIS:
   - KL divergence: disagreement between fast and slow policies
   - Conflict map value: long-term EMA of KL at this state
   - Fast entropy: confidence of fast network

🎮 META-CONTROLLER:
   - Delta (Q_slow - Q_fast): advantage of using slow processing
   - Control cost threshold: cost that delta must exceed to prefer slow
   - p(slow): probability of selecting slow processing
   - Selected: which system was actually used

📊 LAMBDA MODULATION:
   - Lambda: TD(λ) eligibility trace parameter
   - High λ (>0.7): long-horizon credit assignment (chunking)
   - Low λ (<0.4): local credit assignment (high control demand)

✅ FINAL ACTION:
   - Selected action: actual action taken
   - Action source: which system produced it
   - Matches optimal: whether it matches slow memory's optimal action
"""

import torch
import numpy as np
from maze_env import MazeEnvironment
from agent import CognitiveAgent
from corridors import MazeGraph
import argparse


def format_policy_distribution(logits, top_k=5):
    """Format policy distribution showing top-k actions."""
    probs = torch.softmax(logits, dim=-1).squeeze().detach().numpy()
    top_indices = np.argsort(probs)[::-1][:top_k]

    lines = []
    for idx in top_indices:
        lines.append(f"    Action {idx}: {probs[idx]:.4f}")
    return "\n".join(lines)


def inspect_state(agent, env, state_idx, goal_idx, verbose=True):
    """
    Inspect agent's full computation for a given state-goal pair.

    Parameters:
    -----------
    agent : CognitiveAgent
        The trained agent
    env : MazeEnvironment
        The environment
    state_idx : int
        Current state node index
    goal_idx : int
        Goal state node index
    verbose : bool
        Whether to print detailed output

    Returns:
    --------
    step_info : dict
        Complete step information
    """
    # Reset agent
    agent.reset()

    # Create state encodings
    state_encoding = torch.zeros(1, agent.num_nodes)
    state_encoding[0, state_idx] = 1.0

    goal_encoding = torch.zeros(1, agent.num_nodes)
    goal_encoding[0, goal_idx] = 1.0

    # Get state positions
    state_pos = env.idx_to_node[state_idx]
    goal_pos = env.idx_to_node[goal_idx]

    # Compute optimal path info
    optimal_action = env.get_optimal_next_action(state_pos, goal_pos)
    optimal_path_length = env.get_optimal_path_length(state_pos, goal_pos)

    # Run agent step
    step_info = agent.step(state_encoding, goal_encoding, temperature=1.0, train_mode=False)

    # Compute lambda
    lambda_val = agent.compute_lambda(step_info['conflict_value'], step_info['p_slow'])

    if verbose:
        print("=" * 80)
        print(f"STATE INSPECTION: State {state_idx} -> Goal {goal_idx}")
        print("=" * 80)

        print(f"\n📍 POSITIONS:")
        print(f"  Current: {state_pos} (node {state_idx})")
        print(f"  Goal: {goal_pos} (node {goal_idx})")
        print(f"  Optimal path length: {optimal_path_length}")
        print(f"  Optimal next action: {optimal_action}")

        print(f"\n🧠 FAST NETWORK (Habitual/Intuitive):")
        print(f"  Value estimate: {step_info['fast_value'].item():.4f}")
        print(f"  Entropy (confidence): {step_info['fast_entropy']:.4f}")
        print(f"  Top-5 action probabilities:")
        print(format_policy_distribution(step_info['fast_logits']))

        print(f"\n💾 SLOW MEMORY (Episodic Retrieval):")
        print(f"  Retrieved action: {step_info['slow_logits'].argmax().item()}")
        print(f"  Top-5 retrieved action probabilities:")
        print(format_policy_distribution(step_info['slow_logits']))

        print(f"\n⚔️  CONFLICT ANALYSIS:")
        print(f"  KL divergence (fast vs slow): {step_info['kl_divergence']:.4f}")
        print(f"  Conflict map value: {step_info['conflict_value']:.4f}")
        print(f"  Fast entropy: {step_info['fast_entropy']:.4f}")

        print(f"\n🎮 META-CONTROLLER:")
        print(f"  Delta (Q_slow - Q_fast): {step_info['delta'].item():.4f}")
        print(f"  Control cost threshold: {agent.controller.control_cost:.4f}")
        print(f"  Decision rule: delta > {agent.controller.control_cost:.4f} → prefer slow")
        print(f"  p(slow): {step_info['p_slow']:.4f}")
        print(f"  Selected: {'SLOW' if step_info['used_slow'] else 'FAST'}")

        print(f"\n📊 LAMBDA MODULATION:")
        print(f"  Lambda (TD trace decay): {lambda_val:.4f}")
        print(f"  Interpretation: {'Long-horizon backup (chunking)' if lambda_val > 0.7 else 'Medium backup' if lambda_val > 0.4 else 'Local credit assignment'}")

        print(f"\n✅ FINAL ACTION:")
        print(f"  Selected action: {step_info['action']}")
        print(f"  Action source: {'Slow memory' if step_info['used_slow'] else 'Fast network'}")
        print(f"  Matches optimal: {step_info['action'] == optimal_action}")

        print("\n" + "=" * 80)

    # Add computed values to step_info
    step_info['lambda'] = lambda_val
    step_info['optimal_action'] = optimal_action
    step_info['optimal_path_length'] = optimal_path_length
    step_info['state_pos'] = state_pos
    step_info['goal_pos'] = goal_pos
    step_info['state_idx'] = state_idx
    step_info['goal_idx'] = goal_idx

    return step_info


def interactive_mode(agent, env):
    """Interactive mode for exploring agent decisions."""
    print("\n" + "=" * 80)
    print("INTERACTIVE AGENT INSPECTION")
    print("=" * 80)
    print("\nCommands:")
    print("  'state <idx> goal <idx>' - Inspect specific state-goal pair")
    print("  'random' - Inspect random state-goal pair")
    print("  'endpoint' - Inspect random endpoint-to-endpoint pair")
    print("  'stats' - Show agent statistics")
    print("  'quit' or 'exit' - Exit")
    print("=" * 80)

    while True:
        try:
            cmd = input("\n> ").strip().lower()

            if cmd in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            elif cmd == 'stats':
                stats = agent.get_statistics()
                print("\n📊 AGENT STATISTICS:")
                for key, value in stats.items():
                    if key != 'conflict_map':
                        print(f"  {key}: {value}")
                print("\n  Conflict Map:")
                for key, value in stats['conflict_map'].items():
                    print(f"    {key}: {value}")

            elif cmd == 'random':
                state_idx = np.random.randint(env.num_nodes)
                goal_idx = np.random.randint(env.num_nodes)
                while goal_idx == state_idx:
                    goal_idx = np.random.randint(env.num_nodes)
                inspect_state(agent, env, state_idx, goal_idx)

            elif cmd == 'endpoint':
                if len(env.endpoint_nodes) < 2:
                    print("Not enough endpoint nodes!")
                    continue
                state_pos = env.endpoint_nodes[np.random.randint(len(env.endpoint_nodes))]
                goal_pos = env.endpoint_nodes[np.random.randint(len(env.endpoint_nodes))]
                while goal_pos == state_pos:
                    goal_pos = env.endpoint_nodes[np.random.randint(len(env.endpoint_nodes))]
                state_idx = env.node_to_idx[state_pos]
                goal_idx = env.node_to_idx[goal_pos]
                inspect_state(agent, env, state_idx, goal_idx)

            elif cmd.startswith('state'):
                parts = cmd.split()
                if len(parts) == 4 and parts[2] == 'goal':
                    try:
                        state_idx = int(parts[1])
                        goal_idx = int(parts[3])
                        if 0 <= state_idx < env.num_nodes and 0 <= goal_idx < env.num_nodes:
                            inspect_state(agent, env, state_idx, goal_idx)
                        else:
                            print(f"Error: Indices must be in range [0, {env.num_nodes-1}]")
                    except ValueError:
                        print("Error: Invalid indices")
                else:
                    print("Error: Use format 'state <idx> goal <idx>'")

            else:
                print("Unknown command. Type 'quit' to exit.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_inspection(agent, env, num_samples=10, endpoint_only=True):
    """
    Inspect multiple random state-goal pairs and show statistics.

    Parameters:
    -----------
    agent : CognitiveAgent
        The trained agent
    env : MazeEnvironment
        The environment
    num_samples : int
        Number of random samples to inspect
    endpoint_only : bool
        Whether to only sample from endpoints
    """
    print("\n" + "=" * 80)
    print(f"BATCH INSPECTION: {num_samples} samples")
    print("=" * 80)

    results = []

    for i in range(num_samples):
        if endpoint_only and len(env.endpoint_nodes) >= 2:
            state_pos = env.endpoint_nodes[np.random.randint(len(env.endpoint_nodes))]
            goal_pos = env.endpoint_nodes[np.random.randint(len(env.endpoint_nodes))]
            while goal_pos == state_pos:
                goal_pos = env.endpoint_nodes[np.random.randint(len(env.endpoint_nodes))]
            state_idx = env.node_to_idx[state_pos]
            goal_idx = env.node_to_idx[goal_pos]
        else:
            state_idx = np.random.randint(env.num_nodes)
            goal_idx = np.random.randint(env.num_nodes)
            while goal_idx == state_idx:
                goal_idx = np.random.randint(env.num_nodes)

        step_info = inspect_state(agent, env, state_idx, goal_idx, verbose=False)
        results.append(step_info)

    # Compute statistics
    used_slow_count = sum([1 for r in results if r['used_slow']])
    correct_count = sum([1 for r in results if r['action'] == r['optimal_action']])
    mean_kl = np.mean([r['kl_divergence'] for r in results])
    mean_entropy = np.mean([r['fast_entropy'] for r in results])
    mean_lambda = np.mean([r['lambda'] for r in results])
    mean_path_length = np.mean([r['optimal_path_length'] for r in results])

    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"  Samples: {num_samples}")
    print(f"  Used slow: {used_slow_count}/{num_samples} ({used_slow_count/num_samples:.1%})")
    print(f"  Correct actions: {correct_count}/{num_samples} ({correct_count/num_samples:.1%})")
    print(f"  Mean KL divergence: {mean_kl:.4f}")
    print(f"  Mean fast entropy: {mean_entropy:.4f}")
    print(f"  Mean lambda: {mean_lambda:.4f}")
    print(f"  Mean optimal path length: {mean_path_length:.1f}")

    print(f"\n📋 DETAILED RESULTS:")
    print(f"{'#':<4} {'State':<8} {'Goal':<8} {'Used':<6} {'Action':<8} {'Optimal':<8} {'Correct':<8} {'KL':<8} {'λ':<8}")
    print("-" * 80)
    for i, r in enumerate(results):
        print(f"{i+1:<4} {r['state_idx']:<8} {r['goal_idx']:<8} "
              f"{'Slow' if r['used_slow'] else 'Fast':<6} "
              f"{r['action']:<8} {r['optimal_action']:<8} "
              f"{'✓' if r['action'] == r['optimal_action'] else '✗':<8} "
              f"{r['kl_divergence']:<8.4f} {r['lambda']:<8.4f}")


def main():
    parser = argparse.ArgumentParser(description='Inspect trained agent outputs')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/agent_final.pt',
                        help='Path to agent checkpoint')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'batch'],
                        help='Inspection mode')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples for batch mode')
    parser.add_argument('--endpoint-only', action='store_true', default=True,
                        help='Only sample from endpoint nodes in batch mode')
    parser.add_argument('--seed', type=int, default=60,
                        help='Random seed for maze generation')
    parser.add_argument('--length', type=int, default=8,
                        help='Maze length')
    parser.add_argument('--width', type=int, default=8,
                        help='Maze width')
    parser.add_argument('--corridor', type=float, default=0.5,
                        help='Corridor parameter')
    parser.add_argument('--control-cost', type=float, default=0.3,
                        help='Control cost')

    args = parser.parse_args()

    print("=" * 80)
    print("AGENT INSPECTION SCRIPT")
    print("=" * 80)

    # Create environment
    print(f"\nCreating environment...")
    maze = MazeGraph(length=args.length, width=args.width,
                     corridor=args.corridor, seed=args.seed)
    env = MazeEnvironment(length=args.length, width=args.width,
                         corridor=args.corridor, seed=args.seed,
                         control_cost=args.control_cost)

    print(f"  Nodes: {env.num_nodes}")
    print(f"  Endpoint nodes: {len(env.endpoint_nodes)}")

    # Create agent
    print(f"\nCreating agent...")
    agent = CognitiveAgent(
        num_nodes=env.num_nodes,
        num_actions=env.num_actions,
        maze_graph=maze.get_graph(),
        control_cost=args.control_cost
    )

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    try:
        agent.load(args.checkpoint)
        print("  ✓ Checkpoint loaded successfully")
    except FileNotFoundError:
        print(f"  ✗ Checkpoint not found: {args.checkpoint}")
        print("  Using untrained agent instead")
    except Exception as e:
        print(f"  ✗ Error loading checkpoint: {e}")
        print("  Using untrained agent instead")

    # Run inspection
    if args.mode == 'interactive':
        interactive_mode(agent, env)
    elif args.mode == 'batch':
        batch_inspection(agent, env, num_samples=args.num_samples,
                        endpoint_only=args.endpoint_only)


if __name__ == "__main__":
    main()
