
"""
Plotting script for the small Cartesian hyperparameter sweep.

This script is designed for outputs produced by small_grid_sweep.py:
- smallgridsweep/aggregate_results.csv
- smallgridsweep/<run_name>/summary.json
- smallgridsweep/<run_name>/timeseries.npz

It generates a compact set of figures to help choose hyperparameters for each lambda:
1. Lambda/LR sweeps for final success, reward, episode length, and AUC
   at teacher_coef=6.5 and tau=0.4.
2. Final (entropy - tau) vs teacher_coef, with one subplot per tau and curves
   colored by LR colormap family with lambda shades.
3. Final (entropy - tau) vs LR, with one figure per tau and one subplot per lambda,
   curves colored by teacher_coef.
4. A small extra summary figure: best LR by lambda at teacher_coef=6.5, tau=0.4.

The formatting follows the general style of the user's larger plotting suite:
publication-style rcParams, clean grids, tight layouts, and multi-format saves.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


class SmallGridSweepPlotter:
    def __init__(self, results_dir="smallgridsweep", output_dir="smallgridsweep_figures", junction_filter=None):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.junction_filter = junction_filter  # None = all, or specify 10 or 20

        self.df = self._load_runs()
        self._normalize_columns()

    def _load_runs(self):
        import re

        # Find all seed subdirectories matching pattern: smallgridsweep_XXj_seedYYY
        seed_dirs = []
        pattern = re.compile(r'smallgridsweep_(\d+)j_seed(\d+)')

        for subdir in self.results_dir.iterdir():
            if subdir.is_dir():
                match = pattern.match(subdir.name)
                if match:
                    junction_count = int(match.group(1))
                    seed = int(match.group(2))
                    aggregate_csv = subdir / "aggregate_results.csv"
                    if aggregate_csv.exists():
                        seed_dirs.append({
                            'path': subdir,
                            'junction_count': junction_count,
                            'seed': seed,
                            'csv': aggregate_csv
                        })

        if not seed_dirs:
            raise FileNotFoundError(
                f"Could not find any seed subdirectories matching 'smallgridsweep_*j_seed*' in {self.results_dir}"
            )

        # Apply junction filter if specified
        if self.junction_filter is not None:
            seed_dirs = [sd for sd in seed_dirs if sd['junction_count'] == self.junction_filter]
            if not seed_dirs:
                raise FileNotFoundError(
                    f"No seed directories found with junction_count={self.junction_filter}"
                )
            print(f"Filtering for junction_count={self.junction_filter}")

        print(f"Found {len(seed_dirs)} seed directories:")
        for sd in sorted(seed_dirs, key=lambda x: (x['junction_count'], x['seed'])):
            print(f"  {sd['path'].name} (junctions={sd['junction_count']}, seed={sd['seed']})")

        # Load all runs from all seed directories
        all_rows = []

        for seed_info in seed_dirs:
            seed_dir = seed_info['path']
            junction_count = seed_info['junction_count']
            seed = seed_info['seed']

            agg = pd.read_csv(seed_info['csv'])

            for _, row in agg.iterrows():
                summary_path = Path(row["summary_path"])
                # If path is relative and doesn't exist as-is, try prepending seed_dir
                if not summary_path.is_absolute():
                    if not summary_path.exists():
                        summary_path = seed_dir / summary_path

                timeseries_path = Path(row["timeseries_path"])
                if not timeseries_path.is_absolute():
                    if not timeseries_path.exists():
                        timeseries_path = seed_dir / timeseries_path

                run_summary = {}
                if summary_path.exists():
                    with open(summary_path, "r") as f:
                        run_summary = json.load(f)

                config = run_summary.get("config", {})
                summary = run_summary.get("summary", {})
                tests = run_summary.get("tests", {})

                auc_success_rate = np.nan
                if timeseries_path.exists():
                    ts = np.load(timeseries_path, allow_pickle=True)
                    if "success_history" in ts:
                        success_history = np.asarray(ts["success_history"], dtype=float)
                        if success_history.size > 1:
                            auc_success_rate = float(np.trapz(success_history, dx=1.0) / (success_history.size - 1))
                        elif success_history.size == 1:
                            auc_success_rate = float(success_history[0])

                all_rows.append({
                    "seed": seed,
                    "junction_count": junction_count,
                    "run_name": row.get("run_name"),
                    "lambda": float(row.get("lambda", config.get("lambda"))),
                    "lr": float(row.get("lr", config.get("lr"))),
                    "teacher_coef": float(row.get("teacher_coef", config.get("teacher_coef"))),
                    "tau": float(row.get("tau", config.get("tau"))),
                    "final_success_rate": float(summary.get("final_success_rate_rolling", row.get("final_success_rate_rolling", np.nan))),
                    "final_avg_reward": float(summary.get("final_avg_reward_rolling", row.get("final_avg_reward_rolling", np.nan))),
                    "final_avg_episode_length": float(summary.get("final_avg_episode_length", row.get("final_avg_episode_length", np.nan))),
                    "final_avg_entropy": float(summary.get("final_avg_entropy", row.get("final_avg_entropy", np.nan))),
                    "final_avg_consultation_rate": float(summary.get("final_avg_consultation_rate", row.get("final_avg_consultation_rate", np.nan))),
                    "final_avg_correction_rate": float(summary.get("final_avg_correction_rate", row.get("final_avg_correction_rate", np.nan))),
                    "final_value_loss": float(summary.get("mean_value_loss_overall", np.nan)),
                    "memory_test_success_rate": float(tests.get("memory_consultation", {}).get("success_rate", row.get("memory_test_success_rate", np.nan))),
                    "fast_only_test_success_rate": float(tests.get("fast_only", {}).get("success_rate", row.get("fast_only_test_success_rate", np.nan))),
                    "auc_success_rate": auc_success_rate,
                })

        df_all = pd.DataFrame(all_rows)

        # Group by hyperparameters and junction_count, then average across seeds
        grouping_cols = ["lambda", "lr", "teacher_coef", "tau", "junction_count"]
        metric_cols = [
            "final_success_rate", "final_avg_reward", "final_avg_episode_length",
            "final_avg_entropy", "final_avg_consultation_rate", "final_avg_correction_rate",
            "final_value_loss", "memory_test_success_rate", "fast_only_test_success_rate",
            "auc_success_rate"
        ]

        # Aggregate: mean across seeds for each (lambda, lr, teacher, tau, junction) combo
        aggregated = df_all.groupby(grouping_cols, as_index=False)[metric_cols].mean()

        print(f"\nLoaded {len(all_rows)} total runs from {len(seed_dirs)} seed directories")
        print(f"Averaged to {len(aggregated)} unique hyperparameter combinations")

        return aggregated

    def _normalize_columns(self):
        self.df["lambda"] = self.df["lambda"].round(3)
        self.df["lr"] = self.df["lr"].astype(float)
        self.df["teacher_coef"] = self.df["teacher_coef"].astype(float)
        self.df["tau"] = self.df["tau"].astype(float)
        self.df["final_entropy_minus_tau"] = self.df["final_avg_entropy"] - self.df["tau"]

        self.lambda_values = sorted(self.df["lambda"].dropna().unique().tolist())
        self.lr_values = sorted(self.df["lr"].dropna().unique().tolist())
        self.teacher_values = sorted(self.df["teacher_coef"].dropna().unique().tolist())
        self.tau_values = sorted(self.df["tau"].dropna().unique().tolist())

    def _save_fig(self, fig, name):
        png_path = self.output_dir / f"{name}.png"
        pdf_path = self.output_dir / f"{name}.pdf"
        fig.savefig(png_path, bbox_inches="tight", dpi=300)
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {png_path}")
    
    def sigmoid(self, x, temp=1.0):
        return 1 / (1 + np.exp(-x / temp))

    @staticmethod
    def _format_lr(lr):
        return f"{lr:.1e}"

    @staticmethod
    def _teacher_label(val):
        return f"{val:g}"

    def _lr_color_map(self):
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(self.lr_values)))
        return {lr: colors[i] for i, lr in enumerate(self.lr_values)}

    def _teacher_color_map(self):
        colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(self.teacher_values)))
        return {t: colors[i] for i, t in enumerate(self.teacher_values)}

    def _lambda_shades_within_lr(self):
        families = {}
        if len(self.lr_values) >= 1:
            families[self.lr_values[0]] = plt.cm.Blues
        if len(self.lr_values) >= 2:
            families[self.lr_values[1]] = plt.cm.Greens
        if len(self.lr_values) >= 3:
            families[self.lr_values[2]] = plt.cm.Reds

        shade_positions = np.linspace(0.35, 0.9, len(self.lambda_values))
        color_map = {}
        for lr in self.lr_values:
            cmap = families.get(lr, plt.cm.Greys)
            for i, lam in enumerate(self.lambda_values):
                color_map[(lr, lam)] = cmap(shade_positions[i])
        return color_map

    def _plot_metric_vs_lambda(self, ax, data, metric, ylabel, title):
        lr_colors = self._lr_color_map()
        for lr in self.lr_values:
            sub = data[data["lr"] == lr].sort_values("lambda")
            ax.plot(
                sub["lambda"],
                sub[metric],
                marker="o",
                linewidth=2,
                markersize=7,
                color=lr_colors[lr],
                label=f"lr={self._format_lr(lr)}",
            )

        ax.set_xlabel("Lambda (λ)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.legend()

    def plot_lambda_lr_metrics(self):
        """Plot success rate AUC vs LR/teacher_coef pairs with mean + std dev ribbons."""
        # Create one subplot per lambda
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        # Use plasma colormap for teacher_coef values
        teacher_colors = {}
        color_positions = np.linspace(0.2, 0.85, len(self.teacher_values))
        for i, teacher in enumerate(self.teacher_values):
            teacher_colors[teacher] = plt.cm.plasma(color_positions[i])

        for ax, lam in zip(axes, self.lambda_values):
            lam_df = self.df[np.isclose(self.df["lambda"], lam)].copy()

            for teacher in self.teacher_values:
                color = teacher_colors.get(teacher, 'gray')

                # Collect data across all tau values for this teacher_coef
                lr_data = {}  # lr -> list of auc_success_rate values

                for tau in self.tau_values:
                    sub = lam_df[
                        np.isclose(lam_df["tau"], tau) &
                        np.isclose(lam_df["teacher_coef"], teacher)
                    ].sort_values("lr")

                    for _, row in sub.iterrows():
                        lr_val = row["lr"]
                        auc_val = row["auc_success_rate"]
                        # Only include finite AUC values
                        if np.isfinite(auc_val):
                            if lr_val not in lr_data:
                                lr_data[lr_val] = []
                            lr_data[lr_val].append(auc_val)

                if not lr_data:
                    continue

                # Compute mean and std dev across tau for each lr
                lrs = sorted(lr_data.keys())
                means = [np.mean(lr_data[lr]) for lr in lrs]
                stds = [np.std(lr_data[lr]) for lr in lrs]

                # Plot mean line
                ax.plot(
                    lrs,
                    means,
                    marker="o",
                    linewidth=2.5,
                    markersize=8,
                    color=color,
                    label=f"teacher={self._teacher_label(teacher)}",
                    zorder=3,
                )

                # Plot std dev ribbon
                means_array = np.array(means)
                stds_array = np.array(stds)
                ax.fill_between(
                    lrs,
                    means_array - stds_array,
                    means_array + stds_array,
                    color=color,
                    alpha=0.2,
                    zorder=2,
                )

            ax.set_xscale("log")
            ax.set_xticks(self.lr_values)
            ax.set_xticklabels([self._format_lr(lr) for lr in self.lr_values], rotation=20)
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("AUC of Success Rate")
            ax.set_title(f"λ = {lam:.1f}")
            # ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.legend(fontsize=9, ncol=1)

        fig.suptitle("AUC Success Rate vs Learning Rate (mean ± std dev across tau)", y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        self._save_fig(fig, "01_auc_success_rate")
        plt.close(fig)

    def plot_entropy_minus_tau_vs_teacher(self):
        """Plot entropy-tau delta vs LR/teacher_coef pairs with mean + std dev ribbons."""
        # Create one subplot per lambda
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        # Use plasma colormap for teacher_coef values
        teacher_colors = {}
        color_positions = np.linspace(0.2, 0.85, len(self.teacher_values))
        for i, teacher in enumerate(self.teacher_values):
            teacher_colors[teacher] = plt.cm.plasma(color_positions[i])

        for ax, lam in zip(axes, self.lambda_values):
            lam_df = self.df[np.isclose(self.df["lambda"], lam)].copy()

            for teacher in self.teacher_values:
                color = teacher_colors.get(teacher, 'gray')

                # Collect data across all tau values for this teacher_coef
                lr_data = {}  # lr -> list of sigmoid(entropy_minus_tau) values

                for tau in self.tau_values:
                    sub = lam_df[
                        np.isclose(lam_df["tau"], tau) &
                        np.isclose(lam_df["teacher_coef"], teacher)
                    ].sort_values("lr")

                    for _, row in sub.iterrows():
                        lr_val = row["lr"]
                        if lr_val not in lr_data:
                            lr_data[lr_val] = []
                        lr_data[lr_val].append(self.sigmoid(row["final_entropy_minus_tau"]))

                if not lr_data:
                    continue

                # Compute mean and std dev across tau for each lr
                lrs = sorted(lr_data.keys())
                means = [np.mean(lr_data[lr]) for lr in lrs]
                stds = [np.std(lr_data[lr]) for lr in lrs]

                # Plot mean line
                ax.plot(
                    lrs,
                    means,
                    marker="o",
                    linewidth=2.5,
                    markersize=8,
                    color=color,
                    label=f"teacher={self._teacher_label(teacher)}",
                    zorder=3,
                )

                # Plot std dev ribbon
                means_array = np.array(means)
                stds_array = np.array(stds)
                ax.fill_between(
                    lrs,
                    means_array - stds_array,
                    means_array + stds_array,
                    color=color,
                    alpha=0.2,
                    zorder=2,
                )

            ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6, zorder=1)
            ax.set_xscale("log")
            ax.set_xticks(self.lr_values)
            ax.set_xticklabels([self._format_lr(lr) for lr in self.lr_values], rotation=20)
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel(r"$\sigma(H - \tau)$ after training")
            ax.set_title(f"λ = {lam:.1f}")
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.legend(fontsize=9, ncol=1)

        fig.suptitle(r"$\sigma(H - \tau)$ vs Learning Rate (mean ± std dev across tau)", y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        self._save_fig(fig, "02_entropy_minus_tau_vs_teacher")
        plt.close(fig)


    def plot_best_lr_summary(self):
        data = self.df.copy()

        if data.empty:
            return

        # Compute composite metric using normalized weighted sum
        # Priority: (1) high AUC + low memory reliance, (2) no value loss explosion

        # AUC is already [0,1], but handle NaN values
        auc_norm = data["auc_success_rate"].fillna(0)  # Treat NaN as worst case

        # Memory independence is already [0,1] from sigmoid
        memory_independence = 1 - self.sigmoid(data["final_entropy_minus_tau"])

        # Normalize value loss to [0,1], then invert (higher stability is better)
        vl_min = data["final_value_loss"].min()
        vl_max = data["final_value_loss"].max()
        value_loss_norm = (data["final_value_loss"] - vl_min) / (vl_max - vl_min)
        stability = 1 - value_loss_norm

        # Weighted combination: 70% primary objective (AUC × memory), 30% stability
        w_primary = 0.7
        w_stability = 0.3
        data["composite_metric"] = w_primary * (auc_norm * memory_independence) + w_stability * stability

        win_rows = []
        for lam in self.lambda_values:
            sub = data[np.isclose(data["lambda"], lam)].copy()
            if sub.empty:
                continue

            # Filter for finite values in all required columns
            valid_sub = sub[
                np.isfinite(sub["composite_metric"]) &
                np.isfinite(sub["final_entropy_minus_tau"]) &
                np.isfinite(sub["final_value_loss"]) &
                (sub["final_value_loss"] > 0)  # Avoid division by zero
            ].copy()

            if valid_sub.empty:
                continue

            # Find best composite metric
            best_row = valid_sub.loc[valid_sub["composite_metric"].idxmax()]

            win_rows.append({
                "lambda": lam,
                "best_lr": float(best_row["lr"]),
                "best_teacher_coef": float(best_row["teacher_coef"]),
                "best_tau": float(best_row["tau"]),
                "best_composite_metric": float(best_row["composite_metric"]),
                "auc_success_rate": float(best_row["auc_success_rate"]) if np.isfinite(best_row["auc_success_rate"]) else 0.0,
                "entropy_minus_tau": float(best_row["final_entropy_minus_tau"]),
                "sigmoid_entropy_minus_tau": float(self.sigmoid(best_row["final_entropy_minus_tau"])),
                "final_value_loss": float(best_row["final_value_loss"]),
            })

        win_df = pd.DataFrame(win_rows)
        if win_df.empty:
            return

        lr_colors = self._lr_color_map()
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        axes = axes.flatten()

        # Top-left panel: composite metric
        ax = axes[0]
        for _, r in win_df.iterrows():
            ax.scatter(r["lambda"], r["best_composite_metric"], s=120, color=lr_colors[r["best_lr"]], zorder=3)
            label = f"t={r['best_teacher_coef']:g}\nτ={r['best_tau']:g}"
            ax.text(r["lambda"], r["best_composite_metric"] + 0.02 * (win_df["best_composite_metric"].max() - win_df["best_composite_metric"].min()),
                    label, ha="left", va="top", fontsize=7)
        ax.plot(win_df["lambda"], win_df["best_composite_metric"], color="gray", alpha=0.5, linewidth=1)
        ax.set_xlabel("Lambda (λ)")
        ax.set_ylabel("Composite Metric")
        ax.set_title("Best Composite Metric by λ")
        ax.grid(alpha=0.3, linewidth=0.5)

        # Top-right panel: AUC success rate for those same best configs
        ax = axes[1]
        for _, r in win_df.iterrows():
            ax.scatter(r["lambda"], r["auc_success_rate"], s=120, color=lr_colors[r["best_lr"]], zorder=3)
            label = f"t={r['best_teacher_coef']:g}\nτ={r['best_tau']:g}"
            ax.text(r["lambda"], r["auc_success_rate"] + 0.02, label,
                    ha="left", va="top", fontsize=7)
        ax.plot(win_df["lambda"], win_df["auc_success_rate"], color="gray", alpha=0.5, linewidth=1)
        ax.set_xlabel("Lambda (λ)")
        ax.set_ylabel("AUC Success Rate")
        ax.set_title("AUC Success Rate of Best Configs")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3, linewidth=0.5)

        # Bottom-left panel: sigmoid(H-tau) for those same best configs
        ax = axes[2]
        for _, r in win_df.iterrows():
            ax.scatter(r["lambda"], r["sigmoid_entropy_minus_tau"], s=120, color=lr_colors[r["best_lr"]], zorder=3)
            label = f"t={r['best_teacher_coef']:g}\nτ={r['best_tau']:g}"
            ax.text(r["lambda"], r["sigmoid_entropy_minus_tau"] + 0.02, label,
                    ha="left", va="top", fontsize=7)
        ax.plot(win_df["lambda"], win_df["sigmoid_entropy_minus_tau"], color="gray", alpha=0.5, linewidth=1)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("Lambda (λ)")
        ax.set_ylabel(r"$\sigma(H - \tau)$")
        ax.set_title(r"Final $\sigma(H - \tau)$ of Best Configs")
        ax.grid(alpha=0.3, linewidth=0.5)

        # Bottom-right panel: final value loss for those same best configs
        ax = axes[3]
        for _, r in win_df.iterrows():
            ax.scatter(r["lambda"], r["final_value_loss"], s=120, color=lr_colors[r["best_lr"]], zorder=3)
            label = f"t={r['best_teacher_coef']:g}\nτ={r['best_tau']:g}"
            ax.text(r["lambda"], r["final_value_loss"] + 0.02 * (win_df["final_value_loss"].max() - win_df["final_value_loss"].min()), label,
                    ha="left", va="top", fontsize=7)
        ax.plot(win_df["lambda"], win_df["final_value_loss"], color="gray", alpha=0.5, linewidth=1)
        ax.set_xlabel("Lambda (λ)")
        ax.set_ylabel("Final Value Loss")
        ax.set_title("Final Value Loss of Best Configs")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, linewidth=0.5)

        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=clr, markersize=10,
                          label=f"lr={self._format_lr(lr)}") for lr, clr in lr_colors.items()]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=True)

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        self._save_fig(fig, "04_best_lr_summary")
        plt.close(fig)

    def plot_value_loss_explosion(self):
        """Plot final value loss vs LR/teacher_coef pairs with mean + std dev ribbons."""
        # Create one subplot per lambda
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        # Use plasma colormap for teacher_coef values
        teacher_colors = {}
        color_positions = np.linspace(0.2, 0.85, len(self.teacher_values))
        for i, teacher in enumerate(self.teacher_values):
            teacher_colors[teacher] = plt.cm.plasma(color_positions[i])

        for ax, lam in zip(axes, self.lambda_values):
            lam_df = self.df[np.isclose(self.df["lambda"], lam)].copy()

            for teacher in self.teacher_values:
                color = teacher_colors.get(teacher, 'gray')

                # Collect data across all tau values for this teacher_coef
                lr_data = {}  # lr -> list of value_loss values

                for tau in self.tau_values:
                    sub = lam_df[
                        np.isclose(lam_df["tau"], tau) &
                        np.isclose(lam_df["teacher_coef"], teacher)
                    ].sort_values("lr")

                    for _, row in sub.iterrows():
                        lr_val = row["lr"]
                        if lr_val not in lr_data:
                            lr_data[lr_val] = []
                        lr_data[lr_val].append(row["final_value_loss"])

                if not lr_data:
                    continue

                # Compute mean and std dev across tau for each lr
                lrs = sorted(lr_data.keys())
                means = [np.mean(lr_data[lr]) for lr in lrs]
                stds = [np.std(lr_data[lr]) for lr in lrs]

                # Plot mean line
                ax.plot(
                    lrs,
                    means,
                    marker="o",
                    linewidth=2.5,
                    markersize=8,
                    color=color,
                    label=f"teacher={self._teacher_label(teacher)}",
                    zorder=3,
                )

                # Plot std dev ribbon
                means_array = np.array(means)
                stds_array = np.array(stds)
                ax.fill_between(
                    lrs,
                    means_array - stds_array,
                    means_array + stds_array,
                    color=color,
                    alpha=0.2,
                    zorder=2,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(self.lr_values)
            ax.set_xticklabels([self._format_lr(lr) for lr in self.lr_values], rotation=20)
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Final Value Loss")
            ax.set_title(f"λ = {lam:.1f}")
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.legend(fontsize=9, ncol=1)

        fig.suptitle("Value Loss: LR vs Teacher Coefficient (mean ± std dev across tau)", y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        self._save_fig(fig, "05_value_loss_explosion")
        plt.close(fig)

    def run_all(self):
        print(f"Loaded {len(self.df)} runs from {self.results_dir}")
        print(f"Lambdas: {self.lambda_values}")
        print(f"LRs: {self.lr_values}")
        print(f"Teacher coefs: {self.teacher_values}")
        print(f"Taus: {self.tau_values}")

        self.plot_lambda_lr_metrics()
        self.plot_entropy_minus_tau_vs_teacher()
        self.plot_best_lr_summary()
        self.plot_value_loss_explosion()

        print(f"All figures saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot small hyperparameter sweep results.")
    parser.add_argument("results_dir", nargs="?", default="smallgridsweep",
                        help="Directory containing aggregate_results.csv and per-run subdirectories.")
    parser.add_argument("--output-dir", default="smallgridsweep_figures",
                        help="Directory to save plots.")
    parser.add_argument("--junction-filter", type=int, default=None, choices=[7, 20],
                        help="Filter to only plot data from specified junction count (7 or 20).")
    args = parser.parse_args()

    plotter = SmallGridSweepPlotter(args.results_dir, args.output_dir, args.junction_filter)
    plotter.run_all()


if __name__ == "__main__":
    main()
