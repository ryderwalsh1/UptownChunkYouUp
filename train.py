"""
Training Script

Two-stage training approach:
1. Stage 1: Pretrain fast network alone
2. Stage 2: Add slow network and meta-controller with RL

Implements TD(λ) with lambda-modulated eligibility traces and
objective: reward + efficiency - control cost
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from maze_env import MazeEnvironment
from agent import CognitiveAgent
from fast import FastNetworkTrainer
from slow import SlowNetworkTrainer
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

    def collect_trajectory(self, max_steps=100):
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
        self.fast_trainer = FastNetworkTrainer(agent.fast_network, lr=3e-4, gamma=gamma)
        self.slow_trainer = SlowNetworkTrainer(agent.slow_network, lr=lr_slow, gamma=gamma)
        self.controller_trainer = MetaControllerTrainer(agent.controller, lr=lr_controller, gamma=gamma)

    def collect_trajectory(self, max_steps=100, temperature=1.0):
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

            # For slow network
            'slow_states': [],
            'slow_goals': [],
            'slow_memories': [],
            'slow_actions': [],
            'slow_log_probs': [],
            'slow_values': [],
            'slow_rewards': [],
            'slow_dones': [],
            'slow_lambdas': [],

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

            # Store trajectory data based on which system was used
            if step_info['used_slow']:
                # Slow network was used
                trajectory['slow_states'].append(state_encoding.squeeze(0))
                trajectory['slow_goals'].append(goal_encoding.squeeze(0))
                trajectory['slow_memories'].append(step_info['retrieved_memory'].squeeze(0))
                trajectory['slow_actions'].append(step_info['action'])
                trajectory['slow_log_probs'].append(step_info['action_log_prob'])
                trajectory['slow_values'].append(step_info['slow_value'].squeeze())
                trajectory['slow_rewards'].append(reward)
                trajectory['slow_dones'].append(done)
                trajectory['slow_lambdas'].append(lambda_val)
            else:
                # Fast network was used
                trajectory['fast_states'].append(state_encoding.squeeze(0))
                trajectory['fast_goals'].append(goal_encoding.squeeze(0))
                trajectory['fast_actions'].append(step_info['action'])
                trajectory['fast_log_probs'].append(step_info['action_log_prob'])
                trajectory['fast_values'].append(step_info['fast_value'].squeeze())
                trajectory['fast_rewards'].append(reward)
                trajectory['fast_dones'].append(done)
                trajectory['fast_lambdas'].append(lambda_val)

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
        trajectory['next_memory'] = final_step_info['retrieved_memory'].squeeze(0)

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
                'hiddens': [None] * len(trajectory['fast_states'])
            }
            fast_loss = self.fast_trainer.train_step(fast_traj)
            loss_dict['fast'] = fast_loss

        # Train slow network if it was used
        if len(trajectory['slow_states']) > 0:
            slow_traj = {
                'states': trajectory['slow_states'],
                'goals': trajectory['slow_goals'],
                'memories': trajectory['slow_memories'],
                'actions': trajectory['slow_actions'],
                'rewards': trajectory['slow_rewards'],
                'dones': trajectory['slow_dones'],
                'values': trajectory['slow_values'],
                'next_state': trajectory['next_state'],
                'next_goal': trajectory['next_goal'],
                'next_memory': trajectory['next_memory']
            }
            slow_loss = self.slow_trainer.train_step(slow_traj)
            loss_dict['slow'] = slow_loss

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
            'mean_lambda': [],
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

            lambdas = []
            for info in step_infos:
                lam = self.agent.compute_lambda(info['conflict_value'], info['p_slow'])
                lambdas.append(lam)
            mean_lambda = np.mean(lambdas)

            # Log metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_lengths'].append(episode_length)
            metrics['success_rate'].append(1.0 if success else 0.0)
            metrics['p_slow'].append(mean_p_slow)
            metrics['mean_lambda'].append(mean_lambda)
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


def plot_training_curves(stage1_metrics, stage2_metrics, save_path='training_curves.png'):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Stage 1 reward
    ax = axes[0, 0]
    window = 50
    rewards = stage1_metrics['episode_rewards']
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(smoothed)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Stage 1: Fast Network Pretraining')
    ax.grid(alpha=0.3)

    # Stage 1 success rate
    ax = axes[0, 1]
    success = np.convolve(stage1_metrics['success_rate'], np.ones(window)/window, mode='valid')
    ax.plot(success)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Stage 1: Success Rate')
    ax.grid(alpha=0.3)

    # Stage 1 length
    ax = axes[0, 2]
    lengths = np.convolve(stage1_metrics['episode_lengths'], np.ones(window)/window, mode='valid')
    ax.plot(lengths)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length')
    ax.set_title('Stage 1: Episode Length')
    ax.grid(alpha=0.3)

    # Stage 2 reward
    ax = axes[1, 0]
    rewards = stage2_metrics['episode_rewards']
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(smoothed)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Stage 2: Full Agent Training')
    ax.grid(alpha=0.3)

    # Stage 2 p(slow)
    ax = axes[1, 1]
    p_slow = np.convolve(stage2_metrics['p_slow'], np.ones(window)/window, mode='valid')
    ax.plot(p_slow)
    ax.set_xlabel('Episode')
    ax.set_ylabel('p(slow)')
    ax.set_title('Stage 2: Slow Processing Usage')
    ax.grid(alpha=0.3)

    # Stage 2 lambda
    ax = axes[1, 2]
    lambdas = np.convolve(stage2_metrics['mean_lambda'], np.ones(window)/window, mode='valid')
    ax.plot(lambdas)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Lambda')
    ax.set_title('Stage 2: Mean Lambda')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved to {save_path}")


if __name__ == "__main__":
    print("Training Cognitive Agent")
    print("=" * 70)

    # Create environment
    from corridors import MazeGraph

    maze = MazeGraph(length=8, width=8, corridor=0.5, seed=60)
    env = MazeEnvironment(length=8, width=8, corridor=0.5, seed=60, control_cost=0.01)

    print(f"Environment: {env.num_nodes} nodes, {env.num_actions} actions")

    # Create agent
    # Note: num_actions = num_nodes (actions are node indices, allows teleportation)
    agent = CognitiveAgent(
        num_nodes=env.num_nodes,
        num_actions=env.num_actions,
        maze_graph=maze.get_graph(),
        control_cost=0.01
    )

    print(f"Agent created")

    # Stage 1: Pretrain fast network
    stage1_trainer = Stage1Trainer(env, agent, lr=3e-4)
    stage1_metrics = stage1_trainer.train(num_episodes=500, log_interval=50)

    # Save after stage 1
    os.makedirs('checkpoints', exist_ok=True)
    agent.save('checkpoints/agent_stage1.pt')
    print("\nStage 1 checkpoint saved")

    # Stage 2: Train with slow and controller
    stage2_trainer = Stage2Trainer(env, agent, lr_slow=3e-4, lr_controller=1e-3)
    stage2_metrics = stage2_trainer.train(num_episodes=1000, log_interval=50)

    # Save final agent
    agent.save('checkpoints/agent_final.pt')
    print("\nFinal checkpoint saved")

    # Plot training curves
    plot_training_curves(stage1_metrics, stage2_metrics, save_path='training_curves.png')

    # Print final statistics
    print("\n" + "=" * 70)
    print("FINAL AGENT STATISTICS")
    print("=" * 70)
    final_stats = agent.get_statistics()
    for key, value in final_stats.items():
        if key != 'conflict_map':
            print(f"  {key}: {value}")

    print("\n✓ Training complete!")
