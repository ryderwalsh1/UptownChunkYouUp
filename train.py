"""
Training Script

Two-stage training approach:
1. Stage 1: Pretrain fast network alone
2. Stage 2: Add slow memory and meta-controller with RL

Implements TD(λ) with lambda-modulated eligibility traces and
objective: reward + efficiency - control cost

Note: Slow memory is NOT trained - it's pure episodic retrieval
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from maze_env import MazeEnvironment
from agent import CognitiveAgent
from fast import FastNetworkTrainer
from controller import MetaControllerTrainer


class Stage1Trainer:
    """Stage 1: Pretrain fast network alone."""

    def __init__(self, env, agent, lr=3e-4, gamma=0.99, lambda_=0.95):
        """
        Initialize Stage 1 trainer.

        Parameters:
        -----------
        env : MazeEnvironment
            Training environment
        agent : CognitiveAgent
            The cognitive agent
        lr : float
            Learning rate
        gamma : float
            Discount factor
        lambda_ : float
            Initial lambda for TD(λ)
        """
        self.env = env
        self.agent = agent
        self.fast_trainer = FastNetworkTrainer(
            agent.fast_network,
            lr=lr,
            gamma=gamma,
            lambda_=lambda_
        )

    def collect_trajectory(self, max_steps=50):
        """
        Collect trajectory using only fast network.

        Returns:
        --------
        trajectory : dict
            Trajectory data for training
        episode_reward : float
            Total episode reward
        episode_length : int
            Episode length
        success : bool
            Whether goal was reached
        """
        state = self.env.reset()
        self.agent.reset()

        states = []
        goals = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        hiddens = []

        episode_reward = 0.0
        success = False

        for step in range(max_steps):
            # Get state encoding
            state_encoding = torch.tensor(state['current_encoding'], dtype=torch.float32).unsqueeze(0)
            goal_encoding = torch.tensor(state['goal_encoding'], dtype=torch.float32).unsqueeze(0)

            # Fast network only
            fast_logits, fast_value, self.agent.fast_hidden = self.agent.fast_network(
                state_encoding, goal_encoding, self.agent.fast_hidden
            )

            # Sample action
            action, log_prob = self.agent.fast_network.sample_action(fast_logits)

            # Take step
            next_state, reward, done, info = self.env.step(action.item(), used_slow=False)

            # Store trajectory
            states.append(state_encoding.squeeze(0))
            goals.append(goal_encoding.squeeze(0))
            actions.append(action.item())
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(fast_value.squeeze())
            hiddens.append(self.agent.fast_hidden)

            episode_reward += reward
            state = next_state

            if done:
                success = info['reached_goal']
                break

        # Store final state for bootstrapping
        next_state_encoding = torch.tensor(state['current_encoding'], dtype=torch.float32)
        next_goal_encoding = torch.tensor(state['goal_encoding'], dtype=torch.float32)

        trajectory = {
            'states': states,
            'goals': goals,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'log_probs': log_probs,
            'values': values,
            'next_state': next_state_encoding,
            'next_goal': next_goal_encoding,
            'hiddens': hiddens
        }

        return trajectory, episode_reward, len(states), success

    def train(self, num_episodes=1000, log_interval=100):
        """
        Train fast network.

        Parameters:
        -----------
        num_episodes : int
            Number of training episodes
        log_interval : int
            Logging interval

        Returns:
        --------
        metrics : dict
            Training metrics
        """
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'losses': []
        }

        print("=" * 70)
        print("STAGE 1: PRETRAINING FAST NETWORK")
        print("=" * 70)

        for episode in tqdm(range(num_episodes), desc="Stage 1"):
            # Collect trajectory
            trajectory, episode_reward, episode_length, success = self.collect_trajectory()

            # Train on trajectory
            loss_dict = self.fast_trainer.train_step(trajectory)

            # Log metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_lengths'].append(episode_length)
            metrics['success_rate'].append(1.0 if success else 0.0)
            metrics['losses'].append(loss_dict['loss'])

            # Periodic logging
            if (episode + 1) % log_interval == 0:
                recent_rewards = metrics['episode_rewards'][-log_interval:]
                recent_success = metrics['success_rate'][-log_interval:]
                recent_lengths = metrics['episode_lengths'][-log_interval:]

                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Mean reward: {np.mean(recent_rewards):.2f}")
                print(f"  Mean length: {np.mean(recent_lengths):.1f}")
                print(f"  Success rate: {np.mean(recent_success):.2%}")
                print(f"  Mean loss: {np.mean(metrics['losses'][-log_interval:]):.4f}")

        print("\n" + "=" * 70)
        print("STAGE 1 COMPLETE")
        print("=" * 70)

        return metrics


class Stage2Trainer:
    """Stage 2: Add slow network and meta-controller."""

    def __init__(self, env, agent, lr_slow=3e-4, lr_controller=1e-3,
                 gamma=0.99, lambda_init=0.95):
        """
        Initialize Stage 2 trainer.

        Parameters:
        -----------
        env : MazeEnvironment
            Training environment
        agent : CognitiveAgent
            The cognitive agent
        lr_slow : float
            Learning rate for slow network
        lr_controller : float
            Learning rate for controller
        gamma : float
            Discount factor
        lambda_init : float
            Initial lambda (will be modulated)
        """
        self.env = env
        self.agent = agent
        self.gamma = gamma
        self.lambda_init = lambda_init

        # Trainers for each component
        # Note: No slow trainer - memory is not trained, only retrieved
        self.fast_trainer = FastNetworkTrainer(agent.fast_network, lr=3e-4, gamma=gamma)
        self.controller_trainer = MetaControllerTrainer(agent.controller, lr=lr_controller, gamma=gamma)

    def collect_trajectory(self, max_steps=50, temperature=1.0):
        """
        Collect trajectory using full agent (fast + slow + controller).

        Returns:
        --------
        trajectory : dict
            Full trajectory data
        episode_reward : float
            Total episode reward
        episode_length : int
            Episode length
        success : bool
            Whether goal was reached
        """
        state = self.env.reset()
        self.agent.reset()

        # Trajectory storage
        trajectory = {
            # For fast network
            'fast_states': [],
            'fast_goals': [],
            'fast_actions': [],
            'fast_log_probs': [],
            'fast_values': [],
            'fast_rewards': [],
            'fast_dones': [],
            'fast_lambdas': [],

            # Note: No slow network trajectory - memory is not trained

            # For controller
            'control_states': [],
            'control_fast_entropies': [],
            'control_kl_divergences': [],
            'control_conflict_values': [],
            'control_actions': [],
            'control_log_probs': [],
            'control_rewards': [],
            'control_dones': [],

            # For analysis
            'step_info': []
        }

        episode_reward = 0.0
        success = False

        for step in range(max_steps):
            # Get state encoding
            state_encoding = torch.tensor(state['current_encoding'], dtype=torch.float32).unsqueeze(0)
            goal_encoding = torch.tensor(state['goal_encoding'], dtype=torch.float32).unsqueeze(0)

            # Full agent step
            step_info = self.agent.step(state_encoding, goal_encoding, temperature=temperature)

            # Take environment step
            next_state, reward, done, info = self.env.step(
                step_info['action'],
                used_slow=step_info['used_slow']
            )

            # Compute lambda for this step
            lambda_val = self.agent.compute_lambda(
                step_info['conflict_value'],
                step_info['p_slow']
            )

            # Update conflict map
            self.agent.update_conflict_map(step_info['state_idx'], step_info['kl_divergence'])

            # Store trajectory data for fast network
            # (Always store, regardless of whether fast or slow was used,
            #  because we use fast_value for both branches)
            trajectory['fast_states'].append(state_encoding.squeeze(0))
            trajectory['fast_goals'].append(goal_encoding.squeeze(0))
            trajectory['fast_actions'].append(step_info['action'])
            trajectory['fast_log_probs'].append(step_info['action_log_prob'])
            trajectory['fast_values'].append(step_info['fast_value'].squeeze())
            trajectory['fast_rewards'].append(reward)
            trajectory['fast_dones'].append(done)
            trajectory['fast_lambdas'].append(lambda_val)

            # Note: Slow memory trajectory not needed - it's not trained

            # Controller trajectory (always collected)
            trajectory['control_states'].append(state_encoding.squeeze(0))
            trajectory['control_fast_entropies'].append(torch.tensor(step_info['fast_entropy']))
            trajectory['control_kl_divergences'].append(torch.tensor(step_info['kl_divergence']))
            trajectory['control_conflict_values'].append(torch.tensor(step_info['conflict_value']))
            trajectory['control_actions'].append(step_info['control_action'])
            trajectory['control_log_probs'].append(step_info['control_log_prob'])
            trajectory['control_rewards'].append(reward)  # Full reward including control cost
            trajectory['control_dones'].append(done)

            # Store full step info for analysis
            trajectory['step_info'].append(step_info)

            episode_reward += reward
            state = next_state

            if done:
                success = info['reached_goal']
                break

        # Store final states for bootstrapping
        final_state_encoding = torch.tensor(state['current_encoding'], dtype=torch.float32)
        final_goal_encoding = torch.tensor(state['goal_encoding'], dtype=torch.float32)

        # Get final step info for retrievals
        final_step_info = self.agent.step(
            final_state_encoding.unsqueeze(0),
            final_goal_encoding.unsqueeze(0),
            temperature=temperature
        )

        trajectory['next_state'] = final_state_encoding
        trajectory['next_goal'] = final_goal_encoding
        # Note: No next_memory needed since slow memory is not trained

        return trajectory, episode_reward, len(trajectory['control_states']), success

    def train_step(self, trajectory):
        """
        Train all components on collected trajectory.

        Returns:
        --------
        loss_dict : dict
            Combined loss dictionary
        """
        loss_dict = {}

        # Train fast network if it was used
        if len(trajectory['fast_states']) > 0:
            # Extract used_slow flags from step_info
            used_slow_flags = [info['used_slow'] for info in trajectory['step_info']]

            fast_traj = {
                'states': trajectory['fast_states'],
                'goals': trajectory['fast_goals'],
                'actions': trajectory['fast_actions'],
                'rewards': trajectory['fast_rewards'],
                'dones': trajectory['fast_dones'],
                'log_probs': trajectory['fast_log_probs'],
                'values': trajectory['fast_values'],
                'next_state': trajectory['next_state'],
                'next_goal': trajectory['next_goal'],
                'hiddens': [None] * len(trajectory['fast_states']),
                'used_slow': used_slow_flags  # Add teacher forcing information
            }
            fast_loss = self.fast_trainer.train_step(fast_traj)
            loss_dict['fast'] = fast_loss

        # Note: Slow memory is NOT trained - it's pure episodic retrieval

        # Train controller
        control_traj = {
            'states': trajectory['control_states'],
            'fast_entropies': trajectory['control_fast_entropies'],
            'kl_divergences': trajectory['control_kl_divergences'],
            'conflict_values': trajectory['control_conflict_values'],
            'control_actions': trajectory['control_actions'],
            'control_log_probs': trajectory['control_log_probs'],
            'rewards': trajectory['control_rewards'],
            'dones': trajectory['control_dones']
        }
        controller_loss = self.controller_trainer.train_step(control_traj)
        loss_dict['controller'] = controller_loss

        return loss_dict

    def train(self, num_episodes=2000, log_interval=100, temperature_schedule=None):
        """
        Train full agent (Stage 2).

        Parameters:
        -----------
        num_episodes : int
            Number of training episodes
        log_interval : int
            Logging interval
        temperature_schedule : callable, optional
            Function mapping episode -> temperature

        Returns:
        --------
        metrics : dict
            Training metrics
        """
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'p_slow': [],
            'p_fast': [],
            'mean_lambda': [],
            'lambda_values': [],  # Store all lambdas per episode for overlay plots
            'mean_fast_entropy': [],
            'mean_kl_divergence': [],
            'used_slow_count': [],
            'used_fast_count': [],
            'mean_q_fast': [],  # Mean Q-value for fast control action
            'mean_q_slow': [],  # Mean Q-value for slow control action
            'losses': []
        }

        print("=" * 70)
        print("STAGE 2: TRAINING WITH SLOW NETWORK AND CONTROLLER")
        print("=" * 70)

        for episode in tqdm(range(num_episodes), desc="Stage 2"):
            # Get temperature for this episode
            if temperature_schedule is not None:
                temperature = temperature_schedule(episode)
            else:
                temperature = 1.0

            # Collect trajectory
            trajectory, episode_reward, episode_length, success = self.collect_trajectory(
                temperature=temperature
            )

            # Train on trajectory
            loss_dict = self.train_step(trajectory)

            # Compute metrics
            step_infos = trajectory['step_info']
            mean_p_slow = np.mean([info['p_slow'] for info in step_infos])
            mean_p_fast = 1.0 - mean_p_slow

            lambdas = []
            for info in step_infos:
                lam = self.agent.compute_lambda(info['conflict_value'], info['p_slow'])
                lambdas.append(lam)
            mean_lambda = np.mean(lambdas)

            # Compute fast entropy and KL divergence
            mean_fast_entropy = np.mean([info['fast_entropy'] for info in step_infos])
            mean_kl_divergence = np.mean([info['kl_divergence'] for info in step_infos])

            # Count actual fast/slow usage
            used_slow_count = sum([1 for info in step_infos if info['used_slow']])
            used_fast_count = len(step_infos) - used_slow_count

            # Compute mean Q-values for controller
            mean_q_fast = np.mean([info['meta_values'][0, 0].item() for info in step_infos])
            mean_q_slow = np.mean([info['meta_values'][0, 1].item() for info in step_infos])

            # Log metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_lengths'].append(episode_length)
            metrics['success_rate'].append(1.0 if success else 0.0)
            metrics['p_slow'].append(mean_p_slow)
            metrics['p_fast'].append(mean_p_fast)
            metrics['mean_lambda'].append(mean_lambda)
            metrics['lambda_values'].append(lambdas)  # Store all lambda values
            metrics['mean_fast_entropy'].append(mean_fast_entropy)
            metrics['mean_kl_divergence'].append(mean_kl_divergence)
            metrics['used_slow_count'].append(used_slow_count)
            metrics['used_fast_count'].append(used_fast_count)
            metrics['mean_q_fast'].append(mean_q_fast)
            metrics['mean_q_slow'].append(mean_q_slow)
            metrics['losses'].append(loss_dict)

            # Periodic logging
            if (episode + 1) % log_interval == 0:
                recent_rewards = metrics['episode_rewards'][-log_interval:]
                recent_success = metrics['success_rate'][-log_interval:]
                recent_lengths = metrics['episode_lengths'][-log_interval:]
                recent_p_slow = metrics['p_slow'][-log_interval:]
                recent_lambda = metrics['mean_lambda'][-log_interval:]

                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Mean reward: {np.mean(recent_rewards):.2f}")
                print(f"  Mean length: {np.mean(recent_lengths):.1f}")
                print(f"  Success rate: {np.mean(recent_success):.2%}")
                print(f"  Mean p(slow): {np.mean(recent_p_slow):.3f}")
                print(f"  Mean lambda: {np.mean(recent_lambda):.3f}")

                agent_stats = self.agent.get_statistics()
                print(f"  Empirical p(slow): {agent_stats['p_slow_empirical']:.3f}")

        print("\n" + "=" * 70)
        print("STAGE 2 COMPLETE")
        print("=" * 70)

        return metrics


def plot_stage1_curves(stage1_metrics, save_path='stage1_training.png'):
    """
    Plot Stage 1 training curves (fast network pretraining).

    Parameters:
    -----------
    stage1_metrics : dict
        Stage 1 training metrics
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    window = 50

    # Reward
    ax = axes[0]
    rewards = stage1_metrics['episode_rewards']
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, color='#2E86AB', linewidth=2)
    else:
        ax.plot(rewards, color='#2E86AB', linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Reward', fontsize=11)
    ax.set_title('Stage 1: Episode Reward', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    # Success rate
    ax = axes[1]
    success = stage1_metrics['success_rate']
    if len(success) > window:
        smoothed = np.convolve(success, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, color='#06A77D', linewidth=2)
    else:
        ax.plot(success, color='#06A77D', linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Success Rate', fontsize=11)
    ax.set_title('Stage 1: Success Rate', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)

    # Episode length
    ax = axes[2]
    lengths = stage1_metrics['episode_lengths']
    if len(lengths) > window:
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, color='#D741A7', linewidth=2)
    else:
        ax.plot(lengths, color='#D741A7', linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Episode Length', fontsize=11)
    ax.set_title('Stage 1: Episode Length', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Stage 1 training curves saved to {save_path}")


def plot_stage2_curves(stage2_metrics, save_path='stage2_training.png'):
    """
    Plot comprehensive Stage 2 training curves with 3x3 grid.

    Parameters:
    -----------
    stage2_metrics : dict
        Stage 2 training metrics
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    window = 50

    # 1. Reward
    ax = axes[0, 0]
    rewards = stage2_metrics['episode_rewards']
    episodes = np.arange(len(rewards))
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        smoothed_episodes = np.arange(window//2, window//2 + len(smoothed))
        ax.plot(smoothed_episodes, smoothed, color='#2E86AB', linewidth=2, label='Smoothed')
    else:
        ax.plot(episodes, rewards, color='#2E86AB', linewidth=2)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Reward', fontsize=10)
    ax.set_title('Episode Reward', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    if len(rewards) > window:
        ax.legend(fontsize=9)

    # 2. Success Rate
    ax = axes[0, 1]
    success = stage2_metrics['success_rate']
    episodes = np.arange(len(success))
    if len(success) > window:
        smoothed = np.convolve(success, np.ones(window)/window, mode='valid')
        smoothed_episodes = np.arange(window//2, window//2 + len(smoothed))
        ax.plot(smoothed_episodes, smoothed, color='#06A77D', linewidth=2, label='Smoothed')
    else:
        ax.plot(episodes, success, color='#06A77D', linewidth=2)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Success Rate', fontsize=10)
    ax.set_title('Success Rate', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    if len(success) > window:
        ax.legend(fontsize=9)

    # 3. Episode Length
    ax = axes[0, 2]
    lengths = stage2_metrics['episode_lengths']
    episodes = np.arange(len(lengths))
    if len(lengths) > window:
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        smoothed_episodes = np.arange(window//2, window//2 + len(smoothed))
        ax.plot(smoothed_episodes, smoothed, color='#D741A7', linewidth=2, label='Smoothed')
    else:
        ax.plot(episodes, lengths, color='#D741A7', linewidth=2)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Episode Length', fontsize=10)
    ax.set_title('Episode Length', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    if len(lengths) > window:
        ax.legend(fontsize=9)

    # 4. Lambda (raw + smoothed overlay)
    ax = axes[1, 0]
    mean_lambda = stage2_metrics['mean_lambda']
    episodes = np.arange(len(mean_lambda))
    # Raw values
    ax.plot(episodes, mean_lambda, color='#FFA630', alpha=0.3, linewidth=1, label='Raw')
    # Smoothed
    if len(mean_lambda) > window:
        smoothed = np.convolve(mean_lambda, np.ones(window)/window, mode='valid')
        smoothed_episodes = np.arange(window//2, window//2 + len(smoothed))
        ax.plot(smoothed_episodes, smoothed, color='#FFA630', linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Lambda (λ)', fontsize=10)
    ax.set_title('TD(λ) Parameter', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # 5. p(slow) and p(fast) overlaid (raw + smoothed)
    ax = axes[1, 1]
    p_slow = stage2_metrics['p_slow']
    p_fast = stage2_metrics['p_fast']
    episodes = np.arange(len(p_slow))

    # Smoothed values
    if len(p_slow) > window:
        p_slow_smoothed = np.convolve(p_slow, np.ones(window)/window, mode='valid')
        p_fast_smoothed = np.convolve(p_fast, np.ones(window)/window, mode='valid')
        smoothed_episodes = np.arange(window//2, window//2 + len(p_slow_smoothed))
        ax.plot(smoothed_episodes, p_slow_smoothed, color='#E63946', linewidth=2, label='p(slow)')
        ax.plot(smoothed_episodes, p_fast_smoothed, color='#457B9D', linewidth=2, label='p(fast)')
    else:
        ax.plot(episodes, p_slow, color='#E63946', linewidth=2, label='p(slow)')
        ax.plot(episodes, p_fast, color='#457B9D', linewidth=2, label='p(fast)')

    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title('Policy Selection Probabilities', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # 6. Controller Q-Values overlaid (raw + smoothed)
    ax = axes[1, 2]
    q_fast = stage2_metrics['mean_q_fast']
    q_slow = stage2_metrics['mean_q_slow']
    episodes = np.arange(len(q_fast))

    # Smoothed values
    if len(q_fast) > window:
        q_fast_smoothed = np.convolve(q_fast, np.ones(window)/window, mode='valid')
        q_slow_smoothed = np.convolve(q_slow, np.ones(window)/window, mode='valid')
        smoothed_episodes = np.arange(window//2, window//2 + len(q_fast_smoothed))
        ax.plot(smoothed_episodes, q_fast_smoothed, color='#457B9D', linewidth=2, label='Q(fast)')
        ax.plot(smoothed_episodes, q_slow_smoothed, color='#E63946', linewidth=2, label='Q(slow)')
    else:
        ax.plot(episodes, q_fast, color='#457B9D', linewidth=2, label='Q(fast)')
        ax.plot(episodes, q_slow, color='#E63946', linewidth=2, label='Q(slow)')

    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Q-Value', fontsize=10)
    ax.set_title('Controller Q-Values', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # 7. Fast Entropy
    ax = axes[2, 0]
    fast_entropy = stage2_metrics['mean_fast_entropy']
    episodes = np.arange(len(fast_entropy))
    # Raw values
    ax.plot(episodes, fast_entropy, color='#06A77D', alpha=0.3, linewidth=1, label='Raw')
    # Smoothed
    if len(fast_entropy) > window:
        smoothed = np.convolve(fast_entropy, np.ones(window)/window, mode='valid')
        smoothed_episodes = np.arange(window//2, window//2 + len(smoothed))
        ax.plot(smoothed_episodes, smoothed, color='#06A77D', linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Entropy', fontsize=10)
    ax.set_title('Fast Network Entropy', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # 8. Control Action Distribution (stacked area)
    ax = axes[2, 1]
    used_slow = np.array(stage2_metrics['used_slow_count'])
    used_fast = np.array(stage2_metrics['used_fast_count'])
    total = used_slow + used_fast
    # Convert to proportions
    prop_slow = used_slow / (total + 1e-10)
    prop_fast = used_fast / (total + 1e-10)
    episodes = np.arange(len(prop_slow))

    # Smooth the proportions
    if len(prop_slow) > window:
        prop_slow_smooth = np.convolve(prop_slow, np.ones(window)/window, mode='valid')
        prop_fast_smooth = np.convolve(prop_fast, np.ones(window)/window, mode='valid')
        smoothed_episodes = np.arange(window//2, window//2 + len(prop_slow_smooth))
        ax.fill_between(smoothed_episodes, 0, prop_fast_smooth, color='#457B9D', alpha=0.7, label='Fast')
        ax.fill_between(smoothed_episodes, prop_fast_smooth, 1, color='#E63946', alpha=0.7, label='Slow')
    else:
        ax.fill_between(episodes, 0, prop_fast, color='#457B9D', alpha=0.7, label='Fast')
        ax.fill_between(episodes, prop_fast, 1, color='#E63946', alpha=0.7, label='Slow')

    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Proportion', fontsize=10)
    ax.set_title('Actual Control Usage', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='best')

    # 9. KL Divergence
    ax = axes[2, 2]
    kl_div = stage2_metrics['mean_kl_divergence']
    episodes = np.arange(len(kl_div))
    # Raw values
    ax.plot(episodes, kl_div, color='#A8DADC', alpha=0.3, linewidth=1, label='Raw')
    # Smoothed
    if len(kl_div) > window:
        smoothed = np.convolve(kl_div, np.ones(window)/window, mode='valid')
        smoothed_episodes = np.arange(window//2, window//2 + len(smoothed))
        ax.plot(smoothed_episodes, smoothed, color='#1D3557', linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('KL Divergence', fontsize=10)
    ax.set_title('Fast-Slow KL Divergence', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    plt.suptitle('Stage 2: Full Agent Training', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Stage 2 training curves saved to {save_path}")


def plot_conflict_map_heatmap(agent, maze, save_path='conflict_map_heatmap.png'):
    """
    Plot conflict map as a 2D heatmap overlaid on maze structure.

    Parameters:
    -----------
    agent : CognitiveAgent
        The trained agent with conflict map
    maze : MazeGraph
        The maze graph structure
    save_path : str
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get conflict values
    conflict_values = agent.conflict_map.conflict_values

    # Get maze dimensions and graph
    length = maze.length
    width = maze.width
    graph = maze.get_graph()

    # Create 2D grid for conflict values
    conflict_grid = np.zeros((length, width))

    # Map conflict values to grid positions
    nodes_list = list(graph.nodes())
    for idx, node in enumerate(nodes_list):
        r, c = node
        conflict_grid[r, c] = conflict_values[idx]

    # Use log scale if values span large range
    vmin = np.min(conflict_values[conflict_values > 0]) if np.any(conflict_values > 0) else 0
    vmax = np.max(conflict_values)

    # Check if log scale is appropriate
    if vmax > 0 and vmin > 0 and (vmax / vmin > 100):
        # Use log scale
        conflict_grid_plot = np.log10(conflict_grid + 1e-10)
        im = ax.imshow(conflict_grid_plot, cmap='hot', origin='upper', aspect='equal')
        cbar = plt.colorbar(im, ax=ax, label='log10(Conflict Value)')
    else:
        # Use linear scale
        im = ax.imshow(conflict_grid, cmap='hot', origin='upper', aspect='equal', vmin=0)
        cbar = plt.colorbar(im, ax=ax, label='Conflict Value')

    # Overlay maze structure (edges)
    for edge in graph.edges():
        node1, node2 = edge
        r1, c1 = node1
        r2, c2 = node2
        # Draw line between nodes (swap x and y for imshow coordinates)
        ax.plot([c1, c2], [r1, r2], color='cyan', linewidth=1.5, alpha=0.4, zorder=1)

    # Mark nodes
    for node in nodes_list:
        r, c = node
        ax.scatter(c, r, s=30, c='white', alpha=0.3, edgecolors='cyan', linewidths=0.5, zorder=2)

    # Highlight high-conflict nodes (top 10%)
    conflict_threshold = np.percentile(conflict_values, 90)
    high_conflict_indices = np.where(conflict_values >= conflict_threshold)[0]

    for idx in high_conflict_indices:
        node = nodes_list[idx]
        r, c = node
        ax.scatter(c, r, s=120, marker='*', c='yellow', edgecolors='orange',
                  linewidths=2, zorder=3, label='High Conflict' if idx == high_conflict_indices[0] else '')

    # Labels and title
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title('Conflict Map Heatmap\n(Brighter = Higher Fast-Slow Disagreement)',
                fontsize=13, fontweight='bold', pad=15)

    # Add grid
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(length))
    ax.grid(False)

    # Add legend
    if len(high_conflict_indices) > 0:
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Add statistics text box
    stats = agent.conflict_map.get_statistics()
    textstr = '\n'.join([
        f"Mean: {stats['mean_conflict']:.4f}",
        f"Std: {stats['std_conflict']:.4f}",
        f"Max: {stats['max_conflict']:.4f}",
        f"Median: {stats['median_conflict']:.4f}",
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Conflict map heatmap saved to {save_path}")


if __name__ == "__main__":
    print("Training Cognitive Agent")
    print("=" * 70)

    # Create environment
    from corridors import MazeGraph

    maze = MazeGraph(length=8, width=8, corridor=0.5, seed=60)
    env = MazeEnvironment(length=8, width=8, corridor=0.5, seed=60, control_cost=0.4)

    print(f"Environment: {env.num_nodes} nodes, {env.num_actions} actions")

    # Create agent
    # Note: num_actions = num_nodes (actions are node indices, allows teleportation)
    agent = CognitiveAgent(
        num_nodes=env.num_nodes,
        num_actions=env.num_actions,
        maze_graph=maze.get_graph(),
        control_cost=0.4
    )

    print(f"Agent created")

    # # Stage 1: Pretrain fast network
    # stage1_trainer = Stage1Trainer(env, agent, lr=3e-4)
    # stage1_metrics = stage1_trainer.train(num_episodes=500, log_interval=50)

    # # Save after stage 1
    # os.makedirs('checkpoints', exist_ok=True)
    # agent.save('checkpoints/agent_stage1.pt')
    # print("\nStage 1 checkpoint saved")

    # # Plot Stage 1 results
    # plot_stage1_curves(stage1_metrics, save_path='stage1_training.png')

    # Stage 2: Train with slow and controller
    stage2_trainer = Stage2Trainer(env, agent, lr_slow=3e-4, lr_controller=1e-3)
    stage2_metrics = stage2_trainer.train(num_episodes=5000, log_interval=50)

    # Save final agent
    agent.save('checkpoints/agent_final.pt')
    print("\nFinal checkpoint saved")

    # Plot Stage 2 results
    plot_stage2_curves(stage2_metrics, save_path='stage2_training.png')

    # Plot conflict map heatmap
    plot_conflict_map_heatmap(agent, maze, save_path='conflict_map_heatmap.png')

    # Print final statistics
    print("\n" + "=" * 70)
    print("FINAL AGENT STATISTICS")
    print("=" * 70)
    final_stats = agent.get_statistics()
    for key, value in final_stats.items():
        if key != 'conflict_map':
            print(f"  {key}: {value}")

    print("\n✓ Training complete!")
