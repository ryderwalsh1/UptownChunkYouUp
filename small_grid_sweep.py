
"""
Small grid sweep for debugging stability across (lambda, lr, teacher_coef, tau).

Adapted from the single-run debug script. It preserves the core functionality:
- builds the same maze/environment/network/memory/trainer stack
- trains one agent per configuration
- logs episode-level metrics
- saves a plot per configuration
- runs the same two final policy tests
- writes per-run JSON/NPZ outputs plus a global CSV summary

Outputs are written under: smallgridsweep/
"""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from maze_env import MazeEnvironment
from fast import FastNetwork, FastNetworkTrainer
from slow import SlowMemory
from corridors import MazeGraph


# -----------------------------
# Fixed experiment settings
# -----------------------------
SEED = 103
GRAPH_LENGTH = 8
GRAPH_WIDTH = 8
CORRIDOR_VAL = 0.0
FIXED_START_NODE = (0, 0)
GOAL_IS_DEADEND = True

NUM_EPISODES = 3000
TEST_EPISODES = 10
SMOOTH_WINDOW = 50
ROLLING_WINDOW = 10

# Sweep values
LAMBDA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LR_VALUES = [1.4e-4, 3e-4, 6e-4]
TEACHER_VALUES = [3.0, 6.5, 10.0]
TAU_VALUES = [0.2, 0.4, 0.6]

# Other trainer / mechanism defaults
GAMMA = 0.99
ENTROPY_COEF = 0.0
VALUE_COEF = 0.5
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
PROSPECTION_HEAD = False

CONSULTATION_TEMPERATURE = 0.5
HARD_TEACHER_FORCE = False
MEMORY_CONSULTATION_COST = 0.0
MEMORY_CORRECTION_COST = 0.0

OUTDIR = Path("smallgridsweep")


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def smooth(data, window=SMOOTH_WINDOW):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def rolling_mean(data, window=ROLLING_WINDOW):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def config_name(lambda_, lr, teacher_coef, tau):
    return (
        f"lam_{lambda_:0.1f}"
        f"__lr_{lr:.1e}"
        f"__teacher_{teacher_coef:g}"
        f"__tau_{tau:0.1f}"
    )


def make_plot(run_dir, env, tau, rewards_history, success_history, entropy_history,
              policy_loss_history, value_loss_history, episode_lengths,
              consultation_rate_history, correction_rate_history, memory_cost_history):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    success_rate = rolling_mean(success_history, ROLLING_WINDOW)
    avg_reward = rolling_mean(rewards_history, ROLLING_WINDOW)

    # Success rate
    axes[0, 0].plot(success_rate * 100, linewidth=2)
    axes[0, 0].set_title("Success Rate (rolling avg)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Success Rate (%)")
    axes[0, 0].grid(True, alpha=0.3)

    # Episode reward
    axes[0, 1].plot(rewards_history, alpha=0.2, color="gray", label="Raw")
    axes[0, 1].plot(avg_reward, linewidth=2, label="Rolling avg")
    axes[0, 1].set_title("Episode Reward")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Episode length
    smoothed_lengths = smooth(episode_lengths)
    axes[0, 2].plot(episode_lengths, alpha=0.2, color="gray")
    axes[0, 2].plot(
        range(SMOOTH_WINDOW // 2, SMOOTH_WINDOW // 2 + len(smoothed_lengths)),
        smoothed_lengths,
        linewidth=2,
        label="Smoothed",
    )
    axes[0, 2].axhline(y=env.max_steps, color="r", linestyle="--", label="Max steps")
    axes[0, 2].set_title("Episode Length")
    axes[0, 2].set_xlabel("Episode")
    axes[0, 2].set_ylabel("Steps")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Entropy
    smoothed_entropy = smooth(entropy_history)
    axes[1, 0].plot(entropy_history, alpha=0.2, color="gray")
    axes[1, 0].plot(
        range(SMOOTH_WINDOW // 2, SMOOTH_WINDOW // 2 + len(smoothed_entropy)),
        smoothed_entropy,
        linewidth=2,
        label="Smoothed",
    )
    axes[1, 0].axhline(y=np.log(env.num_actions), color="r", linestyle="--", label="Max entropy")
    axes[1, 0].axhline(y=tau, color="b", linestyle="--", label="Entropy threshold (tau)")
    axes[1, 0].set_title("Policy Entropy")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Entropy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Policy loss
    smoothed_policy_loss = smooth(policy_loss_history)
    axes[1, 1].plot(policy_loss_history, alpha=0.2, color="gray")
    axes[1, 1].plot(
        range(SMOOTH_WINDOW // 2, SMOOTH_WINDOW // 2 + len(smoothed_policy_loss)),
        smoothed_policy_loss,
        linewidth=2,
        label="Smoothed",
    )
    axes[1, 1].set_title("Policy Loss")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Value loss
    smoothed_value_loss = smooth(value_loss_history)
    axes[1, 2].plot(value_loss_history, alpha=0.2, color="gray")
    axes[1, 2].plot(
        range(SMOOTH_WINDOW // 2, SMOOTH_WINDOW // 2 + len(smoothed_value_loss)),
        smoothed_value_loss,
        linewidth=2,
        label="Smoothed",
    )
    axes[1, 2].set_title("Value Loss")
    axes[1, 2].set_xlabel("Episode")
    axes[1, 2].set_ylabel("Loss")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Consultation rate
    consultation_pct = np.array(consultation_rate_history) * 100
    smoothed_consult = smooth(consultation_pct)
    axes[2, 0].plot(consultation_pct, alpha=0.2, color="gray")
    axes[2, 0].plot(
        range(SMOOTH_WINDOW // 2, SMOOTH_WINDOW // 2 + len(smoothed_consult)),
        smoothed_consult,
        linewidth=2,
        label="Smoothed",
    )
    axes[2, 0].set_title("Memory Consultation Rate")
    axes[2, 0].set_xlabel("Episode")
    axes[2, 0].set_ylabel("Consultation Rate (%)")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Correction rate
    correction_pct = np.array(correction_rate_history) * 100
    smoothed_correct = smooth(correction_pct)
    axes[2, 1].plot(correction_pct, alpha=0.2, color="gray")
    axes[2, 1].plot(
        range(SMOOTH_WINDOW // 2, SMOOTH_WINDOW // 2 + len(smoothed_correct)),
        smoothed_correct,
        linewidth=2,
        label="Smoothed",
    )
    axes[2, 1].set_title("Policy Correction Rate")
    axes[2, 1].set_xlabel("Episode")
    axes[2, 1].set_ylabel("Correction Rate (%)")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # Combined consultation + correction
    axes[2, 2].plot(consultation_pct, alpha=0.1, color="blue")
    axes[2, 2].plot(
        range(SMOOTH_WINDOW // 2, SMOOTH_WINDOW // 2 + len(smoothed_consult)),
        smoothed_consult,
        linewidth=2,
        label="Consultation",
        color="blue",
    )
    axes[2, 2].plot(correction_pct, alpha=0.1, color="orange")
    axes[2, 2].plot(
        range(SMOOTH_WINDOW // 2, SMOOTH_WINDOW // 2 + len(smoothed_correct)),
        smoothed_correct,
        linewidth=2,
        label="Correction",
        color="orange",
    )
    axes[2, 2].set_title("Memory Intervention Rates")
    axes[2, 2].set_xlabel("Episode")
    axes[2, 2].set_ylabel("Rate (%)")
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = run_dir / "training_results.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def run_test_policy(env, network, slow_memory, tau, consultation_temperature, test_episodes, use_memory_consultation):
    successes = 0
    episode_records = []

    for ep in range(test_episodes):
        state = env.reset()
        network.reset_hidden(batch_size=1)

        episode_reward = 0.0
        steps_taken = 0
        success = False

        for step in range(env.max_steps):
            state_encoding = torch.tensor(state["current_encoding"], dtype=torch.float32).unsqueeze(0)
            goal_encoding = torch.tensor(state["goal_encoding"], dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action_logits, _, _, _ = network(state_encoding, goal_encoding, network.hidden)

                if use_memory_consultation:
                    policy_entropy = network.compute_entropy(action_logits).item()
                    at_goal = (state["current_pos"] == state["goal_pos"])
                    if at_goal:
                        consult_memory = True
                    else:
                        consultation_logit = (policy_entropy - tau) / consultation_temperature
                        consultation_prob = torch.sigmoid(torch.tensor(consultation_logit)).item()
                        consult_memory = np.random.random() < consultation_prob

                    memory_action_logits = slow_memory.query(state_encoding, goal_encoding)
                    memory_action_idx = memory_action_logits.argmax().item()

                    if consult_memory:
                        action = memory_action_idx
                    else:
                        action, _ = network.sample_action(action_logits)
                        action = action.item()
                else:
                    action, _ = network.sample_action(action_logits)
                    action = action.item()

            next_state, reward, done, info = env.step(action, used_slow=False)
            episode_reward += reward
            steps_taken = step + 1

            if done:
                success = bool(info["reached_goal"])
                if success:
                    successes += 1
                break

            state = next_state

        episode_records.append({
            "episode": ep,
            "reward": float(episode_reward),
            "steps": int(steps_taken),
            "success": bool(success),
        })

    return {
        "successes": int(successes),
        "test_episodes": int(test_episodes),
        "success_rate": float(successes / test_episodes),
        "episodes": episode_records,
    }


def run_single_config(lambda_, lr, teacher_coef, tau, outdir):
    run_name = config_name(lambda_, lr, teacher_coef, tau)
    run_dir = outdir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"RUN: {run_name}")
    print("=" * 100)

    set_all_seeds(SEED)

    # Create maze and environment
    maze = MazeGraph(length=GRAPH_LENGTH, width=GRAPH_WIDTH, corridor=CORRIDOR_VAL, seed=SEED)
    graph = maze.get_graph()

    env = MazeEnvironment(
        length=GRAPH_LENGTH,
        width=GRAPH_WIDTH,
        corridor=CORRIDOR_VAL,
        seed=SEED,
        control_cost=0.0,
        fixed_start_node=FIXED_START_NODE,
        goal_is_deadend=GOAL_IS_DEADEND,
    )

    print(f"Maze created | nodes={len(graph.nodes())} | edges={len(graph.edges())} | max_steps={env.max_steps}")

    # Create network + memory + trainer
    network = FastNetwork(
        num_nodes=env.num_nodes,
        num_actions=env.num_actions,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        prospection_head=PROSPECTION_HEAD,
    )

    slow_memory = SlowMemory(
        num_nodes=env.num_nodes,
        num_actions=env.num_actions,
    )
    slow_memory.initialize_memory(maze)

    trainer = FastNetworkTrainer(
        network=network,
        lr=lr,
        gamma=GAMMA,
        lambda_=lambda_,
        entropy_coef=ENTROPY_COEF,
        teacher_coef=teacher_coef,
        value_coef=VALUE_COEF,
    )

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

    for episode in range(NUM_EPISODES):
        state = env.reset()
        network.reset_hidden(batch_size=1)
        optimal_length = env.get_optimal_path_length()

        trajectory = {
            "states": [],
            "goals": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
            "values": [],
            "hiddens": [],
            "next_state": None,
            "next_goal": None,
            "used_slow": [],
            "policy_entropies": [],
            "consulted_memory": [],
            "policy_corrected": [],
        }

        episode_reward = 0.0
        episode_length = 0
        episode_memory_cost = 0.0
        success = False

        for step in range(env.max_steps):
            state_encoding = torch.tensor(state["current_encoding"], dtype=torch.float32).unsqueeze(0)
            goal_encoding = torch.tensor(state["goal_encoding"], dtype=torch.float32).unsqueeze(0)

            action_logits, _, value, hidden = network(state_encoding, goal_encoding, network.hidden)
            policy_entropy = network.compute_entropy(action_logits).item()

            consultation_logit = (policy_entropy - tau) / CONSULTATION_TEMPERATURE
            consultation_prob = torch.sigmoid(torch.tensor(consultation_logit)).item()
            consult_memory = np.random.random() < consultation_prob

            memory_action_logits = slow_memory.query(state_encoding, goal_encoding)
            memory_action_idx = memory_action_logits.argmax().item()

            teacher_force_this_step = False
            policy_corrected = False
            memory_cost = 0.0

            if consult_memory:
                action_to_take = memory_action_idx
                teacher_force_this_step = True
                consulted = True
                memory_cost = MEMORY_CONSULTATION_COST
            else:
                sampled_action, sampled_log_prob = network.sample_action(action_logits)
                sampled_action_idx = sampled_action.item()
                consulted = False

                if HARD_TEACHER_FORCE:
                    if sampled_action_idx != memory_action_idx:
                        action_to_take = memory_action_idx
                        teacher_force_this_step = True
                        policy_corrected = True
                        memory_cost = MEMORY_CORRECTION_COST
                    else:
                        action_to_take = sampled_action_idx
                        teacher_force_this_step = False
                else:
                    action_to_take = sampled_action_idx
                    teacher_force_this_step = False

            action_tensor = torch.tensor([action_to_take], dtype=torch.long)
            if teacher_force_this_step:
                log_prob = network.get_log_prob(memory_action_logits, action_tensor)
            else:
                log_prob = network.get_log_prob(action_logits, action_tensor)

            trajectory["states"].append(torch.tensor(state["current_encoding"]))
            trajectory["goals"].append(torch.tensor(state["goal_encoding"]))
            trajectory["actions"].append(action_to_take)
            trajectory["log_probs"].append(log_prob)
            trajectory["values"].append(value.squeeze())
            trajectory["hiddens"].append(hidden)
            trajectory["used_slow"].append(teacher_force_this_step)
            trajectory["policy_entropies"].append(policy_entropy)
            trajectory["consulted_memory"].append(consulted)
            trajectory["policy_corrected"].append(policy_corrected)

            next_state, reward, done, info = env.step(action_to_take, used_slow=False)
            reward_with_cost = reward - memory_cost

            trajectory["rewards"].append(reward_with_cost)
            trajectory["dones"].append(done)

            episode_reward += reward_with_cost
            episode_memory_cost += memory_cost
            episode_length += 1

            if done:
                success = bool(info["reached_goal"])
                trajectory["next_state"] = torch.tensor(next_state["current_encoding"])
                trajectory["next_goal"] = torch.tensor(next_state["goal_encoding"])
                break

            state = next_state

        if trajectory["next_state"] is None:
            trajectory["next_state"] = torch.tensor(state["current_encoding"])
            trajectory["next_goal"] = torch.tensor(state["goal_encoding"])

        loss_dict = trainer.train_step(trajectory)

        consultation_rate = (
            float(np.mean(trajectory["consulted_memory"]))
            if len(trajectory["consulted_memory"]) > 0 else 0.0
        )
        correction_rate = (
            float(np.mean(trajectory["policy_corrected"]))
            if len(trajectory["policy_corrected"]) > 0 else 0.0
        )

        rewards_history.append(float(episode_reward))
        success_history.append(1.0 if success else 0.0)
        entropy_history.append(float(loss_dict["mean_entropy"]))
        policy_loss_history.append(float(loss_dict["policy_loss"]))
        value_loss_history.append(float(loss_dict["value_loss"]))
        episode_lengths.append(int(episode_length))
        consultation_rate_history.append(consultation_rate)
        correction_rate_history.append(correction_rate)
        memory_cost_history.append(float(episode_memory_cost))

        if episode % 500 == 0:
            success_rate_10 = float(np.mean(success_history[-10:]) * 100)
            avg_reward_10 = float(np.mean(rewards_history[-10:]))
            avg_entropy_10 = float(np.mean(entropy_history[-10:]))
            avg_consult_10 = float(np.mean(consultation_rate_history[-10:]) * 100)
            avg_correct_10 = float(np.mean(correction_rate_history[-10:]) * 100)
            avg_mem_cost_10 = float(np.mean(memory_cost_history[-10:]))

            print(
                f"Ep {episode:4d} | Reward: {episode_reward:7.2f} | "
                f"Len: {episode_length:3d}/{optimal_length:2d} | Success: {success} | "
                f"SuccRate(10): {success_rate_10:5.1f}% | Entropy: {avg_entropy_10:.3f} | "
                f"Consult: {avg_consult_10:4.1f}% | Correct: {avg_correct_10:4.1f}% | "
                f"MemCost: {avg_mem_cost_10:.3f} | "
                f"PLoss: {loss_dict['policy_loss']:8.4f} | VLoss: {loss_dict['value_loss']:8.4f}"
            )

    success_rate_roll = rolling_mean(success_history, ROLLING_WINDOW)
    avg_reward_roll = rolling_mean(rewards_history, ROLLING_WINDOW)

    final_summary = {
        "final_success_rate_rolling": float(success_rate_roll[-1] if len(success_rate_roll) else np.mean(success_history)),
        "final_avg_reward_rolling": float(avg_reward_roll[-1] if len(avg_reward_roll) else np.mean(rewards_history)),
        "final_avg_episode_length": float(np.mean(episode_lengths[-ROLLING_WINDOW:])),
        "final_avg_entropy": float(np.mean(entropy_history[-ROLLING_WINDOW:])),
        "max_entropy_uniform": float(np.log(env.num_actions)),
        "final_avg_consultation_rate": float(np.mean(consultation_rate_history[-ROLLING_WINDOW:])),
        "final_avg_correction_rate": float(np.mean(correction_rate_history[-ROLLING_WINDOW:])),
        "final_avg_memory_cost": float(np.mean(memory_cost_history[-ROLLING_WINDOW:])),
        "mean_reward_overall": float(np.mean(rewards_history)),
        "mean_success_overall": float(np.mean(success_history)),
        "mean_policy_loss_overall": float(np.mean(policy_loss_history)),
        "mean_value_loss_overall": float(np.mean(value_loss_history)),
    }

    plot_path = make_plot(
        run_dir,
        env,
        tau,
        rewards_history,
        success_history,
        entropy_history,
        policy_loss_history,
        value_loss_history,
        episode_lengths,
        consultation_rate_history,
        correction_rate_history,
        memory_cost_history,
    )

    print(f"Saved plot: {plot_path}")

    memory_test = run_test_policy(
        env=env,
        network=network,
        slow_memory=slow_memory,
        tau=tau,
        consultation_temperature=CONSULTATION_TEMPERATURE,
        test_episodes=TEST_EPISODES,
        use_memory_consultation=True,
    )

    fast_only_test = run_test_policy(
        env=env,
        network=network,
        slow_memory=slow_memory,
        tau=tau,
        consultation_temperature=CONSULTATION_TEMPERATURE,
        test_episodes=TEST_EPISODES,
        use_memory_consultation=False,
    )

    run_record = {
        "config": {
            "seed": SEED,
            "graph_length": GRAPH_LENGTH,
            "graph_width": GRAPH_WIDTH,
            "corridor_val": CORRIDOR_VAL,
            "fixed_start_node": FIXED_START_NODE,
            "goal_is_deadend": GOAL_IS_DEADEND,
            "num_episodes": NUM_EPISODES,
            "test_episodes": TEST_EPISODES,
            "lambda": lambda_,
            "lr": lr,
            "teacher_coef": teacher_coef,
            "tau": tau,
            "gamma": GAMMA,
            "entropy_coef": ENTROPY_COEF,
            "value_coef": VALUE_COEF,
            "consultation_temperature": CONSULTATION_TEMPERATURE,
            "hard_teacher_force": HARD_TEACHER_FORCE,
            "memory_consultation_cost": MEMORY_CONSULTATION_COST,
            "memory_correction_cost": MEMORY_CORRECTION_COST,
        },
        "maze": {
            "num_nodes": int(len(graph.nodes())),
            "num_edges": int(len(graph.edges())),
            "max_steps": int(env.max_steps),
            "num_actions": int(env.num_actions),
            "num_nodes_env": int(env.num_nodes),
        },
        "summary": final_summary,
        "tests": {
            "memory_consultation": memory_test,
            "fast_only": fast_only_test,
        },
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(run_record, f, indent=2)

    np.savez_compressed(
        run_dir / "timeseries.npz",
        rewards_history=np.asarray(rewards_history, dtype=float),
        success_history=np.asarray(success_history, dtype=float),
        entropy_history=np.asarray(entropy_history, dtype=float),
        policy_loss_history=np.asarray(policy_loss_history, dtype=float),
        value_loss_history=np.asarray(value_loss_history, dtype=float),
        episode_lengths=np.asarray(episode_lengths, dtype=float),
        consultation_rate_history=np.asarray(consultation_rate_history, dtype=float),
        correction_rate_history=np.asarray(correction_rate_history, dtype=float),
        memory_cost_history=np.asarray(memory_cost_history, dtype=float),
    )

    return {
        "run_name": run_name,
        "lambda": lambda_,
        "lr": lr,
        "teacher_coef": teacher_coef,
        "tau": tau,
        "final_success_rate_rolling": final_summary["final_success_rate_rolling"],
        "final_avg_reward_rolling": final_summary["final_avg_reward_rolling"],
        "final_avg_episode_length": final_summary["final_avg_episode_length"],
        "final_avg_entropy": final_summary["final_avg_entropy"],
        "final_avg_consultation_rate": final_summary["final_avg_consultation_rate"],
        "final_avg_correction_rate": final_summary["final_avg_correction_rate"],
        "memory_test_success_rate": memory_test["success_rate"],
        "fast_only_test_success_rate": fast_only_test["success_rate"],
        "summary_path": str(run_dir / "summary.json"),
        "plot_path": str(run_dir / "training_results.png"),
        "timeseries_path": str(run_dir / "timeseries.npz"),
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_runs = len(LAMBDA_VALUES) * len(LR_VALUES) * len(TEACHER_VALUES) * len(TAU_VALUES)
    run_idx = 0

    for lambda_ in LAMBDA_VALUES:
        for lr in LR_VALUES:
            for teacher_coef in TEACHER_VALUES:
                for tau in TAU_VALUES:
                    run_idx += 1
                    print()
                    print("#" * 100)
                    print(f"Starting run {run_idx}/{total_runs}")
                    print("#" * 100)
                    result = run_single_config(
                        lambda_=lambda_,
                        lr=lr,
                        teacher_coef=teacher_coef,
                        tau=tau,
                        outdir=OUTDIR,
                    )
                    all_results.append(result)

                    # Update cumulative CSV after every run
                    csv_path = OUTDIR / "aggregate_results.csv"
                    fieldnames = list(all_results[0].keys())
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(all_results)

    # Also save a JSON copy
    with open(OUTDIR / "aggregate_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print()
    print("=" * 100)
    print("Sweep complete.")
    print(f"Saved aggregate CSV to: {OUTDIR / 'aggregate_results.csv'}")
    print(f"Saved aggregate JSON to: {OUTDIR / 'aggregate_results.json'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
