
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
    def __init__(self, results_dir="smallgridsweep", output_dir="smallgridsweep_figures"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.aggregate_csv = self.results_dir / "aggregate_results.csv"
        if not self.aggregate_csv.exists():
            raise FileNotFoundError(f"Could not find aggregate_results.csv in {self.results_dir}")

        self.df = self._load_runs()
        self._normalize_columns()

    def _load_runs(self):
        agg = pd.read_csv(self.aggregate_csv)
        rows = []

        for _, row in agg.iterrows():
            summary_path = Path(row["summary_path"])
            # If path is relative and doesn't exist as-is, try prepending results_dir
            if not summary_path.is_absolute():
                if not summary_path.exists():
                    summary_path = self.results_dir / summary_path

            timeseries_path = Path(row["timeseries_path"])
            if not timeseries_path.is_absolute():
                if not timeseries_path.exists():
                    timeseries_path = self.results_dir / timeseries_path

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

            rows.append({
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
                "summary_path": str(summary_path),
                "timeseries_path": str(timeseries_path),
            })

        return pd.DataFrame(rows)

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
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

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

    def plot_lambda_lr_metrics(self, teacher_coef=6.5, tau=0.4):
        data = self.df[
            np.isclose(self.df["teacher_coef"], teacher_coef) &
            np.isclose(self.df["tau"], tau)
        ].copy()

        if data.empty:
            print(f"No data found for teacher_coef={teacher_coef}, tau={tau}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        self._plot_metric_vs_lambda(
            axes[0, 0], data,
            metric="final_success_rate",
            ylabel="Final Success Rate",
            title=f"Final Success vs λ (teacher={teacher_coef:g}, tau={tau:g})",
        )
        axes[0, 0].set_ylim(-0.02, 1.02)

        self._plot_metric_vs_lambda(
            axes[0, 1], data,
            metric="final_avg_reward",
            ylabel="Final Episode Reward",
            title=f"Final Reward vs λ (teacher={teacher_coef:g}, tau={tau:g})",
        )

        self._plot_metric_vs_lambda(
            axes[1, 0], data,
            metric="final_avg_episode_length",
            ylabel="Final Episode Length",
            title=f"Final Episode Length vs λ (teacher={teacher_coef:g}, tau={tau:g})",
        )

        self._plot_metric_vs_lambda(
            axes[1, 1], data,
            metric="auc_success_rate",
            ylabel="AUC of Success Rate",
            title=f"AUC Success vs λ (teacher={teacher_coef:g}, tau={tau:g})",
        )
        axes[1, 1].set_ylim(-0.02, 1.02)

        plt.tight_layout()
        self._save_fig(fig, "01_lambda_lr_metrics_teacher6p5_tau0p4")
        plt.close(fig)

    def plot_entropy_minus_tau_vs_teacher(self):
        """Plot entropy-tau delta vs LR/teacher_coef pairs, matching value loss explosion format."""
        # Create one subplot per lambda (same layout as value loss explosion)
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        # Assign colormaps to teacher_coef values (same as value loss plot)
        teacher_colormaps = {}
        if len(self.teacher_values) >= 1:
            teacher_colormaps[self.teacher_values[0]] = plt.cm.Greens
        if len(self.teacher_values) >= 2:
            teacher_colormaps[self.teacher_values[1]] = plt.cm.Blues
        if len(self.teacher_values) >= 3:
            teacher_colormaps[self.teacher_values[2]] = plt.cm.Reds

        # Map tau to color intensity within each colormap
        tau_positions = np.linspace(0.4, 0.9, len(self.tau_values))

        for ax, lam in zip(axes, self.lambda_values):
            lam_df = self.df[np.isclose(self.df["lambda"], lam)].copy()

            for teacher in self.teacher_values:
                cmap = teacher_colormaps.get(teacher, plt.cm.Greys)

                for i, tau in enumerate(self.tau_values):
                    sub = lam_df[
                        np.isclose(lam_df["tau"], tau) &
                        np.isclose(lam_df["teacher_coef"], teacher)
                    ].sort_values("lr")

                    if sub.empty:
                        continue

                    ax.plot(
                        sub["lr"],
                        self.sigmoid(sub["final_entropy_minus_tau"]),
                        marker="o",
                        linewidth=1.8,
                        markersize=6,
                        color=cmap(tau_positions[i]),
                        label=f"t={self._teacher_label(teacher)}, τ={tau:g}",
                    )

            ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
            ax.set_xscale("log")
            ax.set_xticks(self.lr_values)
            ax.set_xticklabels([self._format_lr(lr) for lr in self.lr_values], rotation=20)
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel(r"$\sigma(H - \tau)$ after training")
            ax.set_title(f"λ = {lam:.1f}")
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.legend(fontsize=7, ncol=1)

        fig.suptitle(r"$\sigma(H - \tau)$ vs Learning Rate", y=0.995)

        # Add colormap family legend (same as value loss plot)
        teacher_handles = []
        teacher_labels = []
        for teacher in self.teacher_values:
            cmap = teacher_colormaps.get(teacher, plt.cm.Greys)
            teacher_handles.append(Line2D([0], [0], color=cmap(0.65), lw=3))
            teacher_labels.append(f"teacher={self._teacher_label(teacher)}")

        fig.legend(teacher_handles, teacher_labels, loc="upper center", ncol=len(self.teacher_values),
                  title="Teacher coef color families (darker = higher tau)", frameon=True, bbox_to_anchor=(0.5, 0.96))

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        self._save_fig(fig, "02_entropy_minus_tau_vs_teacher")
        plt.close(fig)

    def plot_entropy_minus_tau_vs_lr_by_lambda(self):
        teacher_colors = self._teacher_color_map()

        for tau in self.tau_values:
            tau_df = self.df[np.isclose(self.df["tau"], tau)].copy()
            if tau_df.empty:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
            axes = axes.flatten()

            for ax, lam in zip(axes, self.lambda_values):
                lam_df = tau_df[np.isclose(tau_df["lambda"], lam)].copy()

                for teacher in self.teacher_values:
                    sub = lam_df[np.isclose(lam_df["teacher_coef"], teacher)].sort_values("lr")
                    if sub.empty:
                        continue

                    ax.plot(
                        sub["lr"],
                        sub["final_entropy_minus_tau"],
                        marker="o",
                        linewidth=2,
                        markersize=6,
                        color=teacher_colors[teacher],
                        label=f"teacher={self._teacher_label(teacher)}",
                    )

                ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
                ax.set_xscale("log")
                ax.set_xticks(self.lr_values)
                ax.set_xticklabels([self._format_lr(lr) for lr in self.lr_values], rotation=20)
                ax.set_xlabel("Learning Rate")
                ax.set_title(f"λ = {lam:.1f}")
                ax.grid(alpha=0.3, linewidth=0.5)

            axes[0].set_ylabel("Final (Entropy - Tau)")
            axes[3].set_ylabel("Final (Entropy - Tau)")

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=len(self.teacher_values), frameon=True)
            fig.suptitle(f"Final (Entropy - Tau) vs Learning Rate, grouped by Teacher Coefficient (tau={tau:g})", y=1.02)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            tau_name = str(tau).replace(".", "p")
            self._save_fig(fig, f"03_entropy_minus_tau_vs_lr_by_lambda_tau_{tau_name}")
            plt.close(fig)

    def plot_best_lr_summary(self):
        data = self.df.copy()

        if data.empty:
            return

        # Compute composite metric: reward * sigmoid(negative entropy-tau) / value_loss
        # Higher is better: high reward AND negative (entropy - tau) AND low value loss
        data["composite_metric"] = data["final_avg_reward"] * (1 - self.sigmoid(data["final_entropy_minus_tau"])) / (data["final_value_loss"])

        win_rows = []
        for lam in self.lambda_values:
            sub = data[np.isclose(data["lambda"], lam)].copy()
            if sub.empty:
                continue

            # Filter for finite values in all required columns
            valid_sub = sub[
                np.isfinite(sub["composite_metric"]) &
                np.isfinite(sub["final_avg_reward"]) &
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
                "final_reward": float(best_row["final_avg_reward"]),
                "entropy_minus_tau": float(best_row["final_entropy_minus_tau"]),
            })

        win_df = pd.DataFrame(win_rows)
        if win_df.empty:
            return

        lr_colors = self._lr_color_map()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

        # Left panel: composite metric
        ax = axes[0]
        for _, r in win_df.iterrows():
            ax.scatter(r["lambda"], r["best_composite_metric"], s=120, color=lr_colors[r["best_lr"]], zorder=3)
            label = f"t={r['best_teacher_coef']:g}\nτ={r['best_tau']:g}"
            ax.text(r["lambda"], r["best_composite_metric"] + 0.02 * (win_df["best_composite_metric"].max() - win_df["best_composite_metric"].min()),
                    label, ha="left", va="top", fontsize=7)
        ax.plot(win_df["lambda"], win_df["best_composite_metric"], color="gray", alpha=0.5, linewidth=1)
        ax.set_xlabel("Lambda (λ)")
        ax.set_ylabel(r"Reward $\times$ $\sigma$($H-\tau$)/ValueLoss")
        ax.set_title("Best Composite Metric by λ")
        ax.grid(alpha=0.3, linewidth=0.5)

        # Right panel: final reward for those same best configs
        ax = axes[1]
        for _, r in win_df.iterrows():
            ax.scatter(r["lambda"], r["final_reward"], s=120, color=lr_colors[r["best_lr"]], zorder=3)
            label = f"t={r['best_teacher_coef']:g}\nτ={r['best_tau']:g}"
            ax.text(r["lambda"], r["final_reward"] + 0.02, label,
                    ha="left", va="top", fontsize=7)
        ax.plot(win_df["lambda"], win_df["final_reward"], color="gray", alpha=0.5, linewidth=1)
        ax.set_xlabel("Lambda (λ)")
        ax.set_ylabel("Final Reward")
        ax.set_title("Final Reward of Best Configs")
        ax.grid(alpha=0.3, linewidth=0.5)

        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=clr, markersize=10,
                          label=f"lr={self._format_lr(lr)}") for lr, clr in lr_colors.items()]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=True)

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        self._save_fig(fig, "04_best_lr_summary")
        plt.close(fig)

    def plot_value_loss_explosion(self):
        """Plot final value loss vs LR/teacher_coef pairs to show instability from high LR and low teacher_coef."""
        # Create one subplot per lambda
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        # Assign colormaps to teacher_coef values
        teacher_colormaps = {}
        if len(self.teacher_values) >= 1:
            teacher_colormaps[self.teacher_values[0]] = plt.cm.Greens
        if len(self.teacher_values) >= 2:
            teacher_colormaps[self.teacher_values[1]] = plt.cm.Blues
        if len(self.teacher_values) >= 3:
            teacher_colormaps[self.teacher_values[2]] = plt.cm.Reds

        # Map tau to color intensity within each colormap
        tau_positions = np.linspace(0.4, 0.9, len(self.tau_values))

        for ax, lam in zip(axes, self.lambda_values):
            lam_df = self.df[np.isclose(self.df["lambda"], lam)].copy()

            for teacher in self.teacher_values:
                cmap = teacher_colormaps.get(teacher, plt.cm.Greys)

                for i, tau in enumerate(self.tau_values):
                    sub = lam_df[
                        np.isclose(lam_df["tau"], tau) &
                        np.isclose(lam_df["teacher_coef"], teacher)
                    ].sort_values("lr")

                    if sub.empty:
                        continue

                    ax.plot(
                        sub["lr"],
                        sub["final_value_loss"],
                        marker="o",
                        linewidth=1.8,
                        markersize=6,
                        color=cmap(tau_positions[i]),
                        label=f"t={self._teacher_label(teacher)}, τ={tau:g}",
                    )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(self.lr_values)
            ax.set_xticklabels([self._format_lr(lr) for lr in self.lr_values], rotation=20)
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Final Value Loss")
            ax.set_title(f"λ = {lam:.1f}")
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.legend(fontsize=7, ncol=1)

        fig.suptitle("Value Loss Explosion: LR vs Teacher Coefficient", y=0.995)

        # Add colormap family legend
        teacher_handles = []
        teacher_labels = []
        for teacher in self.teacher_values:
            cmap = teacher_colormaps.get(teacher, plt.cm.Greys)
            teacher_handles.append(Line2D([0], [0], color=cmap(0.65), lw=3))
            teacher_labels.append(f"teacher={self._teacher_label(teacher)}")

        fig.legend(teacher_handles, teacher_labels, loc="upper center", ncol=len(self.teacher_values),
                  title="Teacher coef color families (darker = higher tau)", frameon=True, bbox_to_anchor=(0.5, 0.96))

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        self._save_fig(fig, "05_value_loss_explosion")
        plt.close(fig)

    def run_all(self):
        print(f"Loaded {len(self.df)} runs from {self.results_dir}")
        print(f"Lambdas: {self.lambda_values}")
        print(f"LRs: {self.lr_values}")
        print(f"Teacher coefs: {self.teacher_values}")
        print(f"Taus: {self.tau_values}")

        self.plot_lambda_lr_metrics(teacher_coef=6.5, tau=0.4)
        self.plot_entropy_minus_tau_vs_teacher()
        self.plot_entropy_minus_tau_vs_lr_by_lambda()
        self.plot_best_lr_summary()
        self.plot_value_loss_explosion()

        print(f"All figures saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot small hyperparameter sweep results.")
    parser.add_argument("results_dir", nargs="?", default="smallgridsweep",
                        help="Directory containing aggregate_results.csv and per-run subdirectories.")
    parser.add_argument("--output-dir", default="smallgridsweep_figures",
                        help="Directory to save plots.")
    args = parser.parse_args()

    plotter = SmallGridSweepPlotter(args.results_dir, args.output_dir)
    plotter.run_all()


if __name__ == "__main__":
    main()
