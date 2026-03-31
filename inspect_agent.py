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
   > video 0 56             # Generate trajectory video from node 0 to 56
   > video 0 56 my_traj.mp4 # Custom output path
   > stats                  # Show agent statistics
   > quit                   # Exit

2. Batch mode (test multiple samples):
   python inspect_agent.py --mode batch --num-samples 20

3. Video mode (generate trajectory visualization):
   python inspect_agent.py --mode video --start-node 0 --goal-node 56
   python inspect_agent.py --mode video --start-node 0 --goal-node 56 --output trajectory.mp4
   python inspect_agent.py --mode video --start-node 0 --goal-node 56 --fps 3 --max-steps 100

4. Load different checkpoint:
   python inspect_agent.py --checkpoint checkpoints/agent_stage1.pt

5. Different maze configuration:
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
   - Top-5 actions: most probable direction actions from fast habitual policy
     (UP/DOWN/LEFT/RIGHT/IDENTIFY_GOAL)

🔮 PROSPECTION HEAD:
   - Top-5 predicted nodes: which future states the network predicts visiting
   - Trained on λ-weighted future trajectory
   - Shows what "chunk endpoint" the network is compressing toward

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

🎬 VIDEO OUTPUT (Video Mode):
   - Animated trajectory showing agent movement through maze
   - Node colors: Prospection head probabilities (brighter = higher probability)
   - Agent color: RED = fast network, BLUE = slow network
   - Info overlay: Step number, mode (FAST/SLOW), lambda value, conflict value
   - Colorbar: Shows prospection probability scale
   - Goal marked with green star
   - Saves as MP4 (with FFMpeg) or GIF (fallback)
"""

import torch
import numpy as np
from maze_env import MazeEnvironment
from agent import CognitiveAgent
from corridors import MazeGraph
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import warnings


def format_policy_distribution(logits, top_k=5, action_names=None):
    """Format policy distribution showing top-k actions."""
    probs = torch.softmax(logits, dim=-1).squeeze().detach().numpy()
    top_indices = np.argsort(probs)[::-1][:top_k]

    lines = []
    for idx in top_indices:
        if action_names and idx < len(action_names):
            action_str = f"{action_names[idx]} ({idx})"
        else:
            action_str = f"Action {idx}"
        lines.append(f"    {action_str}: {probs[idx]:.4f}")
    return "\n".join(lines)


def format_prospection_distribution(logits, env, top_k=5):
    """Format prospection head output showing top-k predicted nodes."""
    probs = torch.softmax(logits, dim=-1).squeeze().detach().numpy()
    top_indices = np.argsort(probs)[::-1][:top_k]

    lines = []
    for idx in top_indices:
        node_pos = env.idx_to_node[idx]
        lines.append(f"    Node {idx} {node_pos}: {probs[idx]:.4f}")
    return "\n".join(lines)


def simulate_trajectory(agent, env, start_pos, goal_pos, max_steps=100):
    """
    Simulate agent trajectory from start to goal, collecting all step data.

    Parameters:
    -----------
    agent : CognitiveAgent
        The trained agent
    env : MazeEnvironment
        The environment
    start_pos : tuple
        Starting position (row, col)
    goal_pos : tuple
        Goal position (row, col)
    max_steps : int
        Maximum number of steps

    Returns:
    --------
    trajectory_data : dict
        Dictionary containing trajectory information:
        - positions: list of (row, col) positions
        - actions: list of actions taken
        - used_slow: list of bool (fast vs slow)
        - lambda_values: list of lambda values
        - conflict_values: list of conflict values
        - prospection_logits: list of prospection head outputs
        - prospection_probs: list of prospection probabilities
        - rewards: list of rewards
        - success: bool (whether goal was reached)
        - steps: int (number of steps taken)
    """
    print(f"Simulating trajectory from {start_pos} to {goal_pos}...")

    # Reset environment with specified start and goal
    state = env.reset(start_pos=start_pos, goal_pos=goal_pos)
    agent.reset()

    # Storage
    positions = [start_pos]
    actions = []
    used_slow_list = []
    lambda_values = []
    conflict_values = []
    prospection_logits_list = []
    prospection_probs_list = []
    rewards = []
    success = False

    for step in range(max_steps):
        # Get current state encoding
        state_encoding = torch.tensor(state['current_encoding'], dtype=torch.float32).unsqueeze(0)
        goal_encoding = torch.tensor(state['goal_encoding'], dtype=torch.float32).unsqueeze(0)

        # Agent step
        step_info = agent.step(state_encoding, goal_encoding, temperature=1.0, train_mode=False)

        # Compute lambda
        lambda_val = agent.compute_lambda(step_info['conflict_value'], step_info['p_slow'])

        # Store prospection data
        prospection_logits = step_info['prospection_logits'].squeeze().detach()
        prospection_probs = torch.softmax(prospection_logits, dim=-1).numpy()

        # Take environment step
        next_state, reward, done, info = env.step(step_info['action'], used_slow=step_info['used_slow'])

        # Store data
        actions.append(step_info['action'])
        used_slow_list.append(step_info['used_slow'])
        lambda_values.append(lambda_val)
        conflict_values.append(step_info['conflict_value'])
        prospection_logits_list.append(prospection_logits)
        prospection_probs_list.append(prospection_probs)
        rewards.append(reward)
        positions.append(env.current_pos)

        state = next_state

        if done:
            success = info['reached_goal']
            break

    trajectory_data = {
        'positions': positions,
        'actions': actions,
        'used_slow': used_slow_list,
        'lambda_values': lambda_values,
        'conflict_values': conflict_values,
        'prospection_logits': prospection_logits_list,
        'prospection_probs': prospection_probs_list,
        'rewards': rewards,
        'success': success,
        'steps': len(actions),
        'start_pos': start_pos,
        'goal_pos': goal_pos
    }

    print(f"  Episode complete: {trajectory_data['steps']} steps, success={success}")

    return trajectory_data


def render_frame(env, agent_pos, goal_pos, prospection_probs, used_slow,
                 lambda_val, conflict_val, step_num, ax, fig, cbar_ref):
    """
    Render a single frame of the trajectory visualization.
    """
    ax.clear()
    ax.set_facecolor('#F8F9FA')

    # Create node position mapping
    node_positions = {(r, c): (c + 0.5, r + 0.5)
                      for r in range(env.length) for c in range(env.width)}

    # Draw edges (maze structure)
    for edge in env.graph.edges():
        node1, node2 = edge
        x1, y1 = node_positions[node1]
        x2, y2 = node_positions[node2]
        ax.plot([x1, x2], [y1, y2], color='#CCCCCC', linewidth=2, alpha=0.4, zorder=1)

    # Draw nodes colored by prospection probabilities
    vmax = max(float(prospection_probs.max()), 1e-8)  # avoid zero-range normalization
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap = plt.cm.plasma

    for node in env.nodes_list:
        node_idx = env.node_to_idx[node]
        prob = prospection_probs[node_idx]
        x, y = node_positions[node]

        if node not in [agent_pos, goal_pos]:
            color = cmap(norm(prob))
            ax.scatter(x, y, s=200, c=[color], alpha=0.8,
                       edgecolors='white', linewidths=1.5, zorder=2)

    # Draw goal position
    gx, gy = node_positions[goal_pos]
    ax.scatter(gx, gy, s=400, c='#2A9D8F', alpha=0.95,
               edgecolors='white', linewidths=2.5, zorder=4, marker='*')
    ax.text(gx, gy - 0.4, 'GOAL', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=5)

    # Draw agent position
    ax_x, ax_y = node_positions[agent_pos]
    agent_color = '#3B82F6' if used_slow else '#EF4444'
    ax.scatter(ax_x, ax_y, s=450, c=agent_color, alpha=0.95,
               edgecolors='white', linewidths=3, zorder=5, marker='o')

    label_text = 'SLOW' if used_slow else 'FAST'
    ax.text(ax_x, ax_y, label_text, ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=6)

    ax.set_xlim(-0.2, env.width + 0.2)
    ax.set_ylim(-0.2, env.length + 0.2)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    # Create/update colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    if cbar_ref[0] is None:
        cbar_ref[0] = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar_ref[0].set_label('Prospection Probability', fontsize=10)
    else:
        cbar_ref[0].update_normal(sm)

    # Add info text box
    info_text = (f"Step: {step_num}\n"
                 f"Mode: {'SLOW' if used_slow else 'FAST'}\n"
                 f"Lambda: {lambda_val:.3f}\n"
                 f"Conflict: {conflict_val:.3f}")

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#333'),
            family='monospace')

    ax.text(env.width / 2, -0.7, 'Agent Trajectory Visualization',
            fontsize=14, fontweight='bold', ha='center', va='top')

    return ax


def generate_trajectory_video(agent, env, trajectory_data, save_path='trajectory.mp4', fps=2, dpi=150):
    """
    Generate video from trajectory data.

    Parameters:
    -----------
    agent : CognitiveAgent
        The agent
    env : MazeEnvironment
        The environment
    trajectory_data : dict
        Dictionary from simulate_trajectory()
    save_path : str
        Output video filename
    fps : int
        Frames per second
    dpi : int
        Resolution

    Returns:
    --------
    None (saves video to file)
    """
    print(f"\nGenerating video...")
    print(f"  Frames: {trajectory_data['steps']}")
    print(f"  FPS: {fps}")
    print(f"  Output: {save_path}")

    # Create figure with space for colorbar
    fig = plt.figure(figsize=(13, 12), facecolor='white')
    ax = fig.add_subplot(111)

    # Store colorbar reference to avoid recreating it
    cbar_ref = [None]

    # Animation update function
    def update_frame(frame_num):
        if frame_num < trajectory_data['steps']:
            render_frame(
                env=env,
                agent_pos=trajectory_data['positions'][frame_num],
                goal_pos=trajectory_data['goal_pos'],
                prospection_probs=trajectory_data['prospection_probs'][frame_num],
                used_slow=trajectory_data['used_slow'][frame_num],
                lambda_val=trajectory_data['lambda_values'][frame_num],
                conflict_val=trajectory_data['conflict_values'][frame_num],
                step_num=frame_num + 1,
                ax=ax,
                fig=fig,
                cbar_ref=cbar_ref
            )
        return ax,

    # Create animation
    anim = animation.FuncAnimation(
        fig, update_frame,
        frames=trajectory_data['steps'],
        interval=1000/fps,
        blit=False,
        repeat=True
    )

    # Try to save with different writers
    try:
        # Try FFMpeg first (best quality)
        Writer = animation.FFMpegWriter
        writer = Writer(fps=fps, bitrate=1800, codec='libx264')
        anim.save(save_path, writer=writer, dpi=dpi)
        print(f"  ✓ Saved using FFMpeg")
    except (RuntimeError, FileNotFoundError):
        try:
            # Fall back to Pillow (GIF)
            if not save_path.endswith('.gif'):
                save_path = save_path.rsplit('.', 1)[0] + '.gif'
            Writer = animation.PillowWriter
            writer = Writer(fps=fps)
            anim.save(save_path, writer=writer, dpi=dpi)
            print(f"  ✓ Saved as GIF using Pillow (FFMpeg not available)")
        except Exception as e:
            print(f"  ✗ Error saving video: {e}")
            print(f"  Tip: Install ffmpeg for better video quality")
            plt.close(fig)
            return

    plt.close(fig)

    # Get file size
    try:
        import os
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"\n✓ Video saved: {save_path} ({size_mb:.2f} MB)")
    except:
        print(f"\n✓ Video saved: {save_path}")


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

    # Action names for direction-based actions
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'IDENTIFY_GOAL'}

    if verbose:
        print("=" * 80)
        print(f"STATE INSPECTION: State {state_idx} -> Goal {goal_idx}")
        print("=" * 80)

        print(f"\n📍 POSITIONS:")
        print(f"  Current: {state_pos} (node {state_idx})")
        print(f"  Goal: {goal_pos} (node {goal_idx})")
        print(f"  Optimal path length: {optimal_path_length}")
        if optimal_action is not None:
            print(f"  Optimal next action: {action_names.get(optimal_action, optimal_action)}")
        else:
            print(f"  Optimal next action: None (already at goal or no path)")

        print(f"\n🧠 FAST NETWORK (Habitual/Intuitive):")
        print(f"  Value estimate: {step_info['fast_value'].item():.4f}")
        print(f"  Entropy (confidence): {step_info['fast_entropy']:.4f}")
        print(f"  Top-5 action probabilities:")
        print(format_policy_distribution(step_info['fast_logits'], action_names=action_names))

        print(f"\n🔮 PROSPECTION HEAD (Future State Prediction):")
        print(f"  Top-5 predicted future nodes:")
        print(format_prospection_distribution(step_info['prospection_logits'], env))

        print(f"\n💾 SLOW MEMORY (Episodic Retrieval):")
        retrieved_action = step_info['slow_logits'].argmax().item()
        print(f"  Retrieved action: {action_names.get(retrieved_action, retrieved_action)}")
        print(f"  Top-5 retrieved action probabilities:")
        print(format_policy_distribution(step_info['slow_logits'], action_names=action_names))

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
        selected_action_name = action_names.get(step_info['action'], step_info['action'])
        print(f"  Selected action: {selected_action_name} ({step_info['action']})")
        print(f"  Action source: {'Slow memory' if step_info['used_slow'] else 'Fast network'}")
        if optimal_action is not None:
            print(f"  Matches optimal: {step_info['action'] == optimal_action}")
        else:
            print(f"  Matches optimal: N/A")

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
    print("  'video <start_idx> <goal_idx> [output_path]' - Generate trajectory video")
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
                if len(env.deadend_nodes) < 2:
                    print("Not enough dead-end nodes!")
                    continue
                state_pos = env.deadend_nodes[np.random.randint(len(env.deadend_nodes))]
                goal_pos = env.deadend_nodes[np.random.randint(len(env.deadend_nodes))]
                while goal_pos == state_pos:
                    goal_pos = env.deadend_nodes[np.random.randint(len(env.deadend_nodes))]
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

            elif cmd.startswith('video'):
                parts = cmd.split()
                if len(parts) >= 3:
                    try:
                        start_idx = int(parts[1])
                        goal_idx = int(parts[2])
                        output_path = parts[3] if len(parts) > 3 else 'trajectory.mp4'

                        if 0 <= start_idx < env.num_nodes and 0 <= goal_idx < env.num_nodes:
                            start_pos = env.idx_to_node[start_idx]
                            goal_pos = env.idx_to_node[goal_idx]

                            # Simulate trajectory
                            trajectory_data = simulate_trajectory(agent, env, start_pos, goal_pos, max_steps=100)

                            # Generate video
                            generate_trajectory_video(agent, env, trajectory_data, save_path=output_path, fps=2)
                        else:
                            print(f"Error: Indices must be in range [0, {env.num_nodes-1}]")
                    except ValueError:
                        print("Error: Invalid indices")
                    except Exception as e:
                        print(f"Error generating video: {e}")
                else:
                    print("Error: Use format 'video <start_idx> <goal_idx> [output_path]'")

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
        if endpoint_only and len(env.deadend_nodes) >= 2:
            state_pos = env.deadend_nodes[np.random.randint(len(env.deadend_nodes))]
            goal_pos = env.deadend_nodes[np.random.randint(len(env.deadend_nodes))]
            while goal_pos == state_pos:
                goal_pos = env.deadend_nodes[np.random.randint(len(env.deadend_nodes))]
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
                        choices=['interactive', 'batch', 'video'],
                        help='Inspection mode')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples for batch mode')
    parser.add_argument('--endpoint-only', action='store_true', default=True,
                        help='Only sample from endpoint nodes in batch mode')

    # Video mode arguments
    parser.add_argument('--start-node', type=int, default=None,
                        help='Start node index (required for video mode)')
    parser.add_argument('--goal-node', type=int, default=None,
                        help='Goal node index (required for video mode)')
    parser.add_argument('--output', type=str, default='trajectory.mp4',
                        help='Output video path (for video mode)')
    parser.add_argument('--fps', type=int, default=2,
                        help='Frames per second for video')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Maximum trajectory steps for video mode')
    parser.add_argument('--seed', type=int, default=60,
                        help='Random seed for maze generation')
    parser.add_argument('--length', type=int, default=8,
                        help='Maze length')
    parser.add_argument('--width', type=int, default=8,
                        help='Maze width')
    parser.add_argument('--corridor', type=float, default=0.5,
                        help='Corridor parameter')
    parser.add_argument('--control-cost', type=float, default=0.15,
                        help='Control cost')

    args = parser.parse_args()

    # Validate video mode arguments
    if args.mode == 'video':
        if args.start_node is None or args.goal_node is None:
            parser.error("--mode video requires --start-node and --goal-node")

    print("=" * 80)
    print("AGENT INSPECTION SCRIPT")
    print("=" * 80)

    # Create environment
    print(f"\nCreating environment...")
    maze = MazeGraph(length=args.length, width=args.width,
                     corridor=args.corridor, seed=args.seed)
    env = MazeEnvironment(length=args.length, width=args.width,
                         corridor=args.corridor, seed=args.seed,
                         control_cost=args.control_cost,
                         fixed_start_node=None,
                         goal_is_deadend=False)

    print(f"  Nodes: {env.num_nodes}")
    print(f"  Actions: {env.num_actions} (UP/DOWN/LEFT/RIGHT/IDENTIFY_GOAL)")
    print(f"  Dead-end nodes: {len(env.deadend_nodes)}")

    # Create agent
    print(f"\nCreating agent...")
    agent = CognitiveAgent(
        num_nodes=env.num_nodes,
        num_actions=env.num_actions,
        maze_graph=maze,  # Pass MazeGraph object
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
    elif args.mode == 'video':
        # Validate node indices
        if not (0 <= args.start_node < env.num_nodes):
            print(f"Error: --start-node must be in range [0, {env.num_nodes-1}]")
            return
        if not (0 <= args.goal_node < env.num_nodes):
            print(f"Error: --goal-node must be in range [0, {env.num_nodes-1}]")
            return

        # Convert indices to positions
        start_pos = env.idx_to_node[args.start_node]
        goal_pos = env.idx_to_node[args.goal_node]

        # Simulate trajectory
        trajectory_data = simulate_trajectory(
            agent, env, start_pos, goal_pos,
            max_steps=args.max_steps
        )

        # Generate video
        generate_trajectory_video(
            agent, env, trajectory_data,
            save_path=args.output,
            fps=args.fps
        )


if __name__ == "__main__":
    main()
