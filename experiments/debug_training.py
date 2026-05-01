"""
Debug script for single training instance.

Tests a single agent training on a simple maze to understand why learning isn't working.
Now includes entropy-gated memory consultation mechanism.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from maze_env import MazeEnvironment
from fast import FastNetwork, FastNetworkTrainer
from slow import SlowMemory
from corridors import MazeGraph

def test_single_training_run():
    """Run a single training session with detailed debugging output."""

    # Create a simple maze
    print("=" * 80)
    print("DEBUG: Single Training Instance")
    print("=" * 80)

    # Set seeds for reproducibility
    seed = 103
    graph_length = 8
    graph_width = 8
    corridor_val = 1.0
    fixed_start_node = (0, 0)
    goal_is_deadend = True

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create maze
    maze = MazeGraph(length=graph_length, width=graph_width, corridor=corridor_val, seed=seed)
    graph = maze.get_graph()

    print(f"\nMaze created:")
    print(f"  Nodes: {len(graph.nodes())}")
    print(f"  Edges: {len(graph.edges())}")

    # Create environment with reward shaping
    env = MazeEnvironment(
        length=graph_length,
        width=graph_width,
        corridor=corridor_val,
        seed=seed,
        control_cost=0.0,
        fixed_start_node=fixed_start_node,
        goal_is_deadend=goal_is_deadend
    )

    print(f"  Max steps: {env.max_steps}")

    # Create fast network
    print("\nUsing FastNetwork with SlowMemory for entropy-gated consultation")
    network = FastNetwork(
        num_nodes=env.num_nodes,
        num_actions=env.num_actions,
        embedding_dim=64,
        hidden_dim=128,
        prospection_head=False
    )

    # Create slow memory and initialize with optimal paths
    slow_memory = SlowMemory(
        num_nodes=env.num_nodes,
        num_actions=env.num_actions
    )
    slow_memory.initialize_memory(maze)

    # Create trainer for fast network
    trainer = FastNetworkTrainer(
        network=network,
        lr=6e-4,
        gamma=0.99,
        lambda_=0.6,
        entropy_coef=0.0,
        teacher_coef=3,
        value_coef=0.5
    )

    # Entropy-gated memory consultation parameters
    tau = 0.6  # Threshold entropy for memory consultation
    consultation_temperature = 0.5  # Temperature for sigmoid gating
    hard_teacher_force = False  # Whether to correct wrong policy samples
    memory_consultation_cost = 0.0  # Reward penalty for consulting memory
    memory_correction_cost = 0.0  # Reward penalty for being corrected by memory

    # Training metrics
    rewards_history = []
    success_history = []
    entropy_history = []
    policy_loss_history = []
    value_loss_history = []
    episode_lengths = []
    consultation_rate_history = []
    correction_rate_history = []
    memory_cost_history = []

    num_episodes = 3000

    print(f"\nTraining for {num_episodes} episodes...")
    print("-" * 80)

    for episode in range(num_episodes):
        # Reset environment and network
        state = env.reset()
        network.reset_hidden(batch_size=1)

        # Episode info
        optimal_length = env.get_optimal_path_length()

        # Trajectory storage
        trajectory = {
            'states': [],
            'goals': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': [],
            'values': [],
            'hiddens': [],
            'next_state': None,
            'next_goal': None,
            'used_slow': [],  # For teacher forcing
            'policy_entropies': [],  # For tracking consultation behavior
            'consulted_memory': [],  # Whether memory was consulted
            'policy_corrected': [],  # Whether policy was corrected
        }

        episode_reward = 0.0
        episode_length = 0
        episode_memory_cost = 0.0
        success = False

        # Episode rollout with entropy-gated memory consultation
        for step in range(env.max_steps):
            # Get state encoding
            state_encoding = torch.tensor(state['current_encoding'], dtype=torch.float32).unsqueeze(0)
            goal_encoding = torch.tensor(state['goal_encoding'], dtype=torch.float32).unsqueeze(0)

            # Forward pass through fast network to get action logits
            action_logits, _, value, hidden = network(
                state_encoding, goal_encoding, network.hidden
            )

            # Compute policy entropy
            policy_entropy = network.compute_entropy(action_logits).item()

            # Entropy-gated memory consultation
            # Probability of consulting memory: sigmoid((entropy - tau) / temperature)
            consultation_logit = (policy_entropy - tau) / consultation_temperature
            consultation_prob = torch.sigmoid(torch.tensor(consultation_logit)).item()
            consult_memory = np.random.random() < consultation_prob

            # Get optimal action from slow memory
            memory_action_logits = slow_memory.query(state_encoding, goal_encoding)
            memory_action_idx = memory_action_logits.argmax().item()

            # Decide on action and whether to use teacher forcing
            teacher_force_this_step = False
            policy_corrected = False
            memory_cost = 0.0  # Cost for memory intervention

            if consult_memory:
                # Memory consulted: use memory action and apply teacher forcing
                action_to_take = memory_action_idx
                teacher_force_this_step = True
                consulted = True
                memory_cost = memory_consultation_cost
            else:
                # Memory not consulted: sample from policy
                sampled_action, sampled_log_prob = network.sample_action(action_logits)
                sampled_action_idx = sampled_action.item()
                consulted = False

                if hard_teacher_force:
                    # Check if sampled action matches memory
                    if sampled_action_idx != memory_action_idx:
                        # Mismatch: correct to memory action and apply teacher forcing
                        action_to_take = memory_action_idx
                        teacher_force_this_step = True
                        policy_corrected = True
                        memory_cost = memory_correction_cost
                    else:
                        # Match: use sampled action, no teacher forcing
                        action_to_take = sampled_action_idx
                        teacher_force_this_step = False
                else:
                    # hard_teacher_force=False: always use sampled action
                    action_to_take = sampled_action_idx
                    teacher_force_this_step = False

            # Get log probability for the action we're storing
            action_tensor = torch.tensor([action_to_take], dtype=torch.long)
            # Use memory logits if teacher forcing, otherwise use fast network logits
            if teacher_force_this_step:
                log_prob = network.get_log_prob(memory_action_logits, action_tensor)
            else:
                log_prob = network.get_log_prob(action_logits, action_tensor)

            # Store trajectory
            trajectory['states'].append(torch.tensor(state['current_encoding']))
            trajectory['goals'].append(torch.tensor(state['goal_encoding']))
            trajectory['actions'].append(action_to_take)
            trajectory['log_probs'].append(log_prob)
            trajectory['values'].append(value.squeeze())
            trajectory['hiddens'].append(hidden)
            trajectory['used_slow'].append(teacher_force_this_step)
            trajectory['policy_entropies'].append(policy_entropy)
            trajectory['consulted_memory'].append(consulted)
            trajectory['policy_corrected'].append(policy_corrected)

            # Take step
            next_state, reward, done, info = env.step(action_to_take, used_slow=False)

            # Apply memory intervention cost to reward
            reward_with_cost = reward - memory_cost

            trajectory['rewards'].append(reward_with_cost)
            trajectory['dones'].append(done)

            episode_reward += reward_with_cost
            episode_memory_cost += memory_cost
            episode_length += 1

            if done:
                success = info['reached_goal']
                trajectory['next_state'] = torch.tensor(next_state['current_encoding'])
                trajectory['next_goal'] = torch.tensor(next_state['goal_encoding'])
                break

            state = next_state

        # If didn't finish, store final state
        if trajectory['next_state'] is None:
            trajectory['next_state'] = torch.tensor(state['current_encoding'])
            trajectory['next_goal'] = torch.tensor(state['goal_encoding'])

        # Train (entropy coefficient is fixed, no annealing)
        loss_dict = trainer.train_step(trajectory)

        # Compute consultation and correction statistics
        consultation_rate = np.mean(trajectory['consulted_memory']) if len(trajectory['consulted_memory']) > 0 else 0.0
        correction_rate = np.mean(trajectory['policy_corrected']) if len(trajectory['policy_corrected']) > 0 else 0.0

        # Store metrics
        rewards_history.append(episode_reward)
        success_history.append(1.0 if success else 0.0)
        entropy_history.append(loss_dict['mean_entropy'])
        policy_loss_history.append(loss_dict['policy_loss'])
        value_loss_history.append(loss_dict['value_loss'])
        episode_lengths.append(episode_length)
        consultation_rate_history.append(consultation_rate)
        correction_rate_history.append(correction_rate)
        memory_cost_history.append(episode_memory_cost)

        # Print progress
        if episode % 100 == 0:
            success_rate = np.mean(success_history[-10:]) * 100
            avg_reward = np.mean(rewards_history[-10:])
            avg_entropy = np.mean(entropy_history[-10:])
            avg_consult = np.mean(consultation_rate_history[-10:]) * 100
            avg_correct = np.mean(correction_rate_history[-10:]) * 100
            avg_mem_cost = np.mean(memory_cost_history[-10:])

            print(f"Ep {episode:3d} | Reward: {episode_reward:6.2f} | Len: {episode_length:3d}/{optimal_length:2d} | "
                  f"Success: {success} | SuccRate(10): {success_rate:5.1f}% | "
                  f"Entropy: {avg_entropy:.3f} | Consult: {avg_consult:4.1f}% | Correct: {avg_correct:4.1f}% | MemCost: {avg_mem_cost:.3f} | "
                  f"PLoss: {loss_dict['policy_loss']:7.4f} | VLoss: {loss_dict['value_loss']:7.4f}")

            # Extra detail for first few episodes
            print(f"       Rewards in episode: {trajectory['rewards']}")
            print(f"       Actions taken: {trajectory['actions']}")
            print(f"       Entropies: {trajectory['policy_entropies']}")
            print(f"       Optimal path length: {optimal_length}")
            print(f"       Consultation rate: {consultation_rate:.3f}, Correction rate: {correction_rate:.3f}")
            print(f"       Memory cost: {episode_memory_cost:.3f}")

    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)

    # Compute rolling averages
    window = 10
    success_rate = np.convolve(success_history, np.ones(window)/window, mode='valid')
    avg_reward = np.convolve(rewards_history, np.ones(window)/window, mode='valid')

    print(f"\nFinal {window}-episode metrics:")
    print(f"  Success rate: {success_rate[-1]*100:.1f}%")
    print(f"  Average reward: {avg_reward[-1]:.2f}")
    print(f"  Average episode length: {np.mean(episode_lengths[-window:]):.1f}")
    print(f"  Average entropy: {np.mean(entropy_history[-window:]):.3f}")
    print(f"  Max entropy (uniform): {np.log(env.num_actions):.3f}")
    print(f"  Average consultation rate: {np.mean(consultation_rate_history[-window:])*100:.1f}%")
    print(f"  Average correction rate: {np.mean(correction_rate_history[-window:])*100:.1f}%")
    print(f"  Average memory cost per episode: {np.mean(memory_cost_history[-window:]):.3f}")

    # Plot results with smoothing
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Smoothing window for plots
    smooth_window = 50

    def smooth(data, window=smooth_window):
        """Apply moving average smoothing to data."""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Success rate
    axes[0, 0].plot(success_rate * 100, linewidth=2)
    axes[0, 0].set_title('Success Rate (rolling avg)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Success Rate (%)')
    axes[0, 0].grid(True, alpha=0.3)

    # Episode reward
    axes[0, 1].plot(rewards_history, alpha=0.2, color='gray', label='Raw')
    axes[0, 1].plot(avg_reward, linewidth=2, label='Rolling avg')
    axes[0, 1].set_title('Episode Reward')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Episode length
    smoothed_lengths = smooth(episode_lengths)
    axes[0, 2].plot(episode_lengths, alpha=0.2, color='gray')
    axes[0, 2].plot(range(smooth_window//2, smooth_window//2 + len(smoothed_lengths)),
                    smoothed_lengths, linewidth=2, label='Smoothed')
    axes[0, 2].axhline(y=env.max_steps, color='r', linestyle='--', label='Max steps')
    axes[0, 2].set_title('Episode Length')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Entropy
    smoothed_entropy = smooth(entropy_history)
    axes[1, 0].plot(entropy_history, alpha=0.2, color='gray')
    axes[1, 0].plot(range(smooth_window//2, smooth_window//2 + len(smoothed_entropy)),
                    smoothed_entropy, linewidth=2, label='Smoothed')
    axes[1, 0].axhline(y=np.log(env.num_actions), color='r', linestyle='--', label='Max entropy')
    axes[1, 0].axhline(y=tau, color='b', linestyle='--', label='Entropy treshold (tau)')
    axes[1, 0].set_title('Policy Entropy')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Policy loss
    smoothed_policy_loss = smooth(policy_loss_history)
    axes[1, 1].plot(policy_loss_history, alpha=0.2, color='gray')
    axes[1, 1].plot(range(smooth_window//2, smooth_window//2 + len(smoothed_policy_loss)),
                    smoothed_policy_loss, linewidth=2, label='Smoothed')
    axes[1, 1].set_title('Policy Loss')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Value loss
    smoothed_value_loss = smooth(value_loss_history)
    axes[1, 2].plot(value_loss_history, alpha=0.2, color='gray')
    axes[1, 2].plot(range(smooth_window//2, smooth_window//2 + len(smoothed_value_loss)),
                    smoothed_value_loss, linewidth=2, label='Smoothed')
    axes[1, 2].set_title('Value Loss')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Consultation rate
    consultation_pct = np.array(consultation_rate_history) * 100
    smoothed_consult = smooth(consultation_pct)
    axes[2, 0].plot(consultation_pct, alpha=0.2, color='gray')
    axes[2, 0].plot(range(smooth_window//2, smooth_window//2 + len(smoothed_consult)),
                    smoothed_consult, linewidth=2, label='Smoothed')
    axes[2, 0].set_title('Memory Consultation Rate')
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Consultation Rate (%)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Correction rate
    correction_pct = np.array(correction_rate_history) * 100
    smoothed_correct = smooth(correction_pct)
    axes[2, 1].plot(correction_pct, alpha=0.2, color='gray')
    axes[2, 1].plot(range(smooth_window//2, smooth_window//2 + len(smoothed_correct)),
                    smoothed_correct, linewidth=2, label='Smoothed')
    axes[2, 1].set_title('Policy Correction Rate')
    axes[2, 1].set_xlabel('Episode')
    axes[2, 1].set_ylabel('Correction Rate (%)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # Combined consultation + correction
    axes[2, 2].plot(consultation_pct, alpha=0.1, color='blue')
    axes[2, 2].plot(range(smooth_window//2, smooth_window//2 + len(smoothed_consult)),
                    smoothed_consult, linewidth=2, label='Consultation', color='blue')
    axes[2, 2].plot(correction_pct, alpha=0.1, color='orange')
    axes[2, 2].plot(range(smooth_window//2, smooth_window//2 + len(smoothed_correct)),
                    smoothed_correct, linewidth=2, label='Correction', color='orange')
    axes[2, 2].set_title('Memory Intervention Rates')
    axes[2, 2].set_xlabel('Episode')
    axes[2, 2].set_ylabel('Rate (%)')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('debug_training_results.png', dpi=150)
    print(f"\nPlot saved to: debug_training_results.png")
    # plt.show()

    # Test final policy
    print("\n" + "=" * 80)
    print("Testing Final Policy")
    print("=" * 80)

    test_episodes = 10
    print("\n--- Policy with Memory Consultation (same as training) ---")
    test_successes = 0
    for ep in range(test_episodes):
        state = env.reset()
        network.reset_hidden(batch_size=1)

        episode_reward = 0.0
        for step in range(env.max_steps):
            state_encoding = torch.tensor(state['current_encoding'], dtype=torch.float32).unsqueeze(0)
            goal_encoding = torch.tensor(state['goal_encoding'], dtype=torch.float32).unsqueeze(0)

            # Use same entropy-gated consultation as training
            with torch.no_grad():
                action_logits, _, _, _ = network(state_encoding, goal_encoding, network.hidden)
                policy_entropy = network.compute_entropy(action_logits).item()

                # Check if at goal
                at_goal = (state['current_pos'] == state['goal_pos'])
                if at_goal:
                    consult_memory = True
                else:
                    consultation_logit = (policy_entropy - tau) / consultation_temperature
                    consultation_prob = torch.sigmoid(torch.tensor(consultation_logit)).item()
                    consult_memory = np.random.random() < consultation_prob

                # Get memory action
                memory_action_logits = slow_memory.query(state_encoding, goal_encoding)
                memory_action_idx = memory_action_logits.argmax().item()

                if consult_memory:
                    action = memory_action_idx
                else:
                    action, _ = network.sample_action(action_logits)
                    action = action.item()

            next_state, reward, done, info = env.step(action, used_slow=False)
            episode_reward += reward

            if done:
                if info['reached_goal']:
                    test_successes += 1
                if ep < 3:  # Only print first few
                    print(f"  Test {ep+1}: Start={state['current_pos']}, Goal={env.goal_pos}, Steps={step+1}, Reward={episode_reward:.2f}, Success={info['reached_goal']}")
                break

            state = next_state

    print(f"\nTest success rate: {test_successes}/{test_episodes} = {test_successes/test_episodes*100:.1f}%")

    print("\n--- Pure Fast Network (no memory, for comparison) ---")
    sample_successes = 0
    for ep in range(test_episodes):
        state = env.reset()
        network.reset_hidden(batch_size=1)

        episode_reward = 0.0
        for step in range(env.max_steps):
            state_encoding = torch.tensor(state['current_encoding'], dtype=torch.float32).unsqueeze(0)
            goal_encoding = torch.tensor(state['goal_encoding'], dtype=torch.float32).unsqueeze(0)

            # Use network with sampling
            with torch.no_grad():
                action_logits, _, _, _ = network(state_encoding, goal_encoding, network.hidden)
                action, _ = network.sample_action(action_logits)
                action = action.item()

            next_state, reward, done, info = env.step(action, used_slow=False)
            episode_reward += reward

            if done:
                if info['reached_goal']:
                    sample_successes += 1
                if ep < 3:  # Only print first few
                    print(f"  Test {ep+1}: Start={state['current_pos']}, Goal={env.goal_pos}, Steps={step+1}, Reward={episode_reward:.2f}, Success={info['reached_goal']}")
                break

            state = next_state

    print(f"\nSampling success rate: {sample_successes}/{test_episodes} = {sample_successes/test_episodes*100:.1f}%")


if __name__ == "__main__":
    test_single_training_run()
