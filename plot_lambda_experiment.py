"""
Comprehensive Plotting Suite for Lambda Experiment

Generates publication-quality figures analyzing how optimal TD(λ) depends on maze topology.
Implements the 7-figure analysis framework from the experimental specification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from scipy import stats
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Color schemes
LAMBDA_COLORS = plt.cm.viridis(np.linspace(0, 1, 10))
TOPOLOGY_COLORS = sns.color_palette("Set2", 8)


class LambdaExperimentPlotter:
    """Main plotting class for lambda experiment analysis."""

    def __init__(self, results_dir, output_dir='figures'):
        """
        Initialize plotter with results directory.

        Parameters:
        -----------
        results_dir : str or Path
            Directory containing results.csv and aggregate metrics
        output_dir : str or Path
            Directory to save generated figures
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load all data
        print(f"Loading data from {self.results_dir}...")
        self.df = pd.read_csv(self.results_dir / 'results.csv')
        self.config = self._load_config()
        self.auc_results = self._load_if_exists('auc_results.csv')
        self.episodes_threshold = self._load_if_exists('episodes_to_threshold.csv')
        self.steps_threshold = self._load_if_exists('steps_to_threshold.csv')
        self.summary = self._load_if_exists('summary_statistics.csv')
        self.timeout_rate = self._load_if_exists('timeout_rate.csv')

        print(f"  Loaded {len(self.df)} episodes")
        print(f"  Topologies: {sorted(self.df['topology'].unique())}")
        print(f"  Lambda values: {sorted(self.df['lambda'].unique())}")
        print(f"  Seeds: {sorted(self.df['seed'].unique())}")

    def _load_config(self):
        """Load experiment config if available."""
        config_path = self.results_dir / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}

    def _load_if_exists(self, filename):
        """Load CSV if it exists, otherwise return None."""
        path = self.results_dir / filename
        if path.exists():
            return pd.read_csv(path)
        return None

    def _save_fig(self, fig, name, subdir=None):
        """Save figure in multiple formats."""
        if subdir:
            save_dir = self.output_dir / subdir
            save_dir.mkdir(exist_ok=True, parents=True)
        else:
            save_dir = self.output_dir

        # Save as PNG and PDF
        png_path = save_dir / f"{name}.png"
        pdf_path = save_dir / f"{name}.pdf"

        fig.savefig(png_path, bbox_inches='tight', dpi=300)
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"  Saved: {png_path}")

    def plot_all(self, figures='all'):
        """
        Generate all plots.

        Parameters:
        -----------
        figures : str or list
            Which figure categories to plot. Options:
            'all', 'learning', 'heatmaps', 'optimal_lambda', 'credit',
            'decision_quality', 'topology', 'diagnostics'
        """
        if figures == 'all':
            figures = ['learning', 'heatmaps', 'optimal_lambda', 'credit',
                      'decision_quality', 'topology', 'diagnostics']
        elif isinstance(figures, str):
            figures = [figures]

        print("\n" + "="*80)
        print("Generating Lambda Experiment Figures")
        print("="*80)

        if 'learning' in figures:
            print("\n[1/7] Learning Dynamics...")
            self.plot_learning_dynamics()

        if 'heatmaps' in figures:
            print("\n[2/7] Performance Heatmaps...")
            self.plot_performance_heatmaps()

        if 'optimal_lambda' in figures:
            print("\n[3/7] Optimal Lambda Analysis...")
            self.plot_optimal_lambda_analysis()

        if 'credit' in figures:
            print("\n[4/7] Credit Assignment Diagnostics...")
            self.plot_credit_diagnostics()

        if 'decision_quality' in figures:
            print("\n[5/7] Decision Quality Metrics...")
            self.plot_decision_quality()

        if 'topology' in figures:
            print("\n[6/7] Topology Descriptors...")
            self.plot_topology_descriptors()

        if 'diagnostics' in figures:
            print("\n[7/7] Detailed Diagnostics...")
            self.plot_detailed_diagnostics()

        print("\n" + "="*80)
        print(f"All figures saved to: {self.output_dir}")
        print("="*80)

    # =========================================================================
    # FIGURE 1: Learning Dynamics
    # =========================================================================

    def plot_learning_dynamics(self):
        """Plot learning curves across λ for each topology."""
        subdir = 'learning'

        topologies = sorted(self.df['topology'].unique())
        lambda_values = sorted(self.df['lambda'].unique())

        # 1. Success rate learning curves
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, topology in enumerate(topologies):
            ax = axes[idx]
            topo_data = self.df[self.df['topology'] == topology]

            for lambda_idx, lambda_val in enumerate(lambda_values):
                lambda_data = topo_data[topo_data['lambda'] == lambda_val]

                # Compute mean and std across seeds
                grouped = lambda_data.groupby('episode')['success'].agg(['mean', 'std', 'count'])
                episodes = grouped.index
                mean = grouped['mean']
                std = grouped['std']
                n = grouped['count']

                # Confidence interval
                ci = 1.96 * std / np.sqrt(n)

                color = LAMBDA_COLORS[lambda_idx % len(LAMBDA_COLORS)]
                ax.plot(episodes, mean, label=f'λ={lambda_val:.1f}',
                       color=color, linewidth=1.5, alpha=0.8)
                ax.fill_between(episodes, mean - ci, mean + ci,
                               color=color, alpha=0.15)

            ax.set_xlabel('Episode')
            ax.set_ylabel('Success Rate')
            ax.set_title(topology.replace('_', ' ').title())
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.legend(ncol=2, fontsize=8)
            ax.set_ylim(-0.05, 1.05)

        # Remove extra subplots if needed
        for idx in range(len(topologies), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        self._save_fig(fig, 'learning_curves_success', subdir)
        plt.close()

        # 2. Reward learning curves
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, topology in enumerate(topologies):
            ax = axes[idx]
            topo_data = self.df[self.df['topology'] == topology]

            for lambda_idx, lambda_val in enumerate(lambda_values):
                lambda_data = topo_data[topo_data['lambda'] == lambda_val]

                grouped = lambda_data.groupby('episode')['episode_reward'].agg(['mean', 'std', 'count'])
                episodes = grouped.index
                mean = grouped['mean']
                std = grouped['std']
                n = grouped['count']
                ci = 1.96 * std / np.sqrt(n)

                color = LAMBDA_COLORS[lambda_idx % len(LAMBDA_COLORS)]
                ax.plot(episodes, mean, label=f'λ={lambda_val:.1f}',
                       color=color, linewidth=1.5, alpha=0.8)
                ax.fill_between(episodes, mean - ci, mean + ci,
                               color=color, alpha=0.15)

            ax.set_xlabel('Episode')
            ax.set_ylabel('Episode Reward')
            ax.set_title(topology.replace('_', ' ').title())
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.legend(ncol=2, fontsize=8)

        for idx in range(len(topologies), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        self._save_fig(fig, 'learning_curves_reward', subdir)
        plt.close()

        # 3. Optimality ratio over training (for successful episodes)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, topology in enumerate(topologies):
            ax = axes[idx]
            topo_data = self.df[(self.df['topology'] == topology) & (self.df['success'] == True)]

            if len(topo_data) == 0:
                ax.text(0.5, 0.5, 'No successful episodes',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(topology.replace('_', ' ').title())
                continue

            for lambda_idx, lambda_val in enumerate(lambda_values):
                lambda_data = topo_data[topo_data['lambda'] == lambda_val]

                if len(lambda_data) == 0:
                    continue

                grouped = lambda_data.groupby('episode')['optimality_ratio'].agg(['mean', 'std', 'count'])
                episodes = grouped.index
                mean = grouped['mean']
                std = grouped['std']
                n = grouped['count']
                ci = 1.96 * std / np.sqrt(n)

                color = LAMBDA_COLORS[lambda_idx % len(LAMBDA_COLORS)]
                ax.plot(episodes, mean, label=f'λ={lambda_val:.1f}',
                       color=color, linewidth=1.5, alpha=0.8)
                ax.fill_between(episodes, mean - ci, mean + ci,
                               color=color, alpha=0.15)

            ax.set_xlabel('Episode')
            ax.set_ylabel('Optimality Ratio\n(path length / optimal)')
            ax.set_title(topology.replace('_', ' ').title())
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Optimal')
            ax.legend(ncol=2, fontsize=8)

        for idx in range(len(topologies), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        self._save_fig(fig, 'learning_curves_optimality', subdir)
        plt.close()

    # =========================================================================
    # FIGURE 2: Performance Heatmaps
    # =========================================================================

    def plot_performance_heatmaps(self):
        """Plot heatmaps of performance metrics vs (λ, topology)."""
        subdir = 'heatmaps'

        if self.auc_results is None or self.summary is None:
            print("  Warning: Missing aggregate metrics files. Skipping heatmaps.")
            return

        topologies = sorted(self.df['topology'].unique())
        lambda_values = sorted(self.df['lambda'].unique())

        # 1. AUC heatmap
        if self.auc_results is not None:
            auc_pivot = self.auc_results.groupby(['topology', 'lambda'])['auc'].mean().unstack()
            auc_pivot = auc_pivot.reindex(topologies)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(auc_pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                       cbar_kws={'label': 'AUC (Success Rate)'}, ax=ax)
            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Topology')
            ax.set_title('Area Under Learning Curve (AUC) by Topology and λ')
            plt.tight_layout()
            self._save_fig(fig, 'heatmap_auc', subdir)
            plt.close()

        # 2. Final success rate heatmap
        final_success = self.summary.groupby(['topology', 'lambda'])['success'].mean().unstack()
        final_success = final_success.reindex(topologies)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(final_success, annot=True, fmt='.2f', cmap='RdYlGn',
                   cbar_kws={'label': 'Final Success Rate'}, ax=ax,
                   vmin=0, vmax=1)
        ax.set_xlabel('Lambda (λ)')
        ax.set_ylabel('Topology')
        ax.set_title('Final Success Rate by Topology and λ')
        plt.tight_layout()
        self._save_fig(fig, 'heatmap_final_success', subdir)
        plt.close()

        # 3. Steps to threshold heatmap
        if self.steps_threshold is not None and 'steps_to_80pct' in self.steps_threshold.columns:
            # Use 80% threshold (most likely to have data)
            steps_pivot = self.steps_threshold.groupby(['topology', 'lambda'])['steps_to_80pct'].mean().unstack()
            steps_pivot = steps_pivot.reindex(topologies)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(steps_pivot, annot=True, fmt='.0f', cmap='RdYlGn_r',
                       cbar_kws={'label': 'Steps to 80% Success'}, ax=ax)
            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Topology')
            ax.set_title('Steps to Reach 80% Success Rate by Topology and λ')
            plt.tight_layout()
            self._save_fig(fig, 'heatmap_steps_to_80pct', subdir)
            plt.close()

        # 4. Combined heatmap with topology descriptors
        # Get topology metrics (use first episode of each topology)
        topo_metrics = self.df.groupby('topology').first()[
            ['topo_mean_corridor_length', 'topo_junction_density', 'topo_frac_corridors']
        ]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Final success with topology annotations
        ax = axes[0]
        sns.heatmap(final_success, annot=True, fmt='.2f', cmap='RdYlGn',
                   cbar_kws={'label': 'Final Success Rate'}, ax=ax,
                   vmin=0, vmax=1)
        ax.set_xlabel('Lambda (λ)')
        ax.set_ylabel('')
        ax.set_title('Final Success Rate')

        # Add topology descriptors as y-axis labels
        y_labels = []
        for topo in topologies:
            if topo in topo_metrics.index:
                corridor_len = topo_metrics.loc[topo, 'topo_mean_corridor_length']
                junc_dens = topo_metrics.loc[topo, 'topo_junction_density']
                label = f"{topo}\n(c={corridor_len:.1f}, j={junc_dens:.2f})"
            else:
                label = topo
            y_labels.append(label)
        ax.set_yticklabels(y_labels, rotation=0)

        # Right: Junction accuracy
        if 'junction_accuracy' in self.summary.columns:
            junction_acc = self.summary.groupby(['topology', 'lambda'])['junction_accuracy'].mean().unstack()
            junction_acc = junction_acc.reindex(topologies)

            ax = axes[1]
            sns.heatmap(junction_acc, annot=True, fmt='.2f', cmap='RdYlGn',
                       cbar_kws={'label': 'Junction Accuracy'}, ax=ax)
            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('')
            ax.set_title('Junction Decision Accuracy')
            ax.set_yticklabels([])

        plt.tight_layout()
        self._save_fig(fig, 'heatmap_combined_annotated', subdir)
        plt.close()

    # =========================================================================
    # FIGURE 3: Optimal Lambda Analysis
    # =========================================================================

    def plot_optimal_lambda_analysis(self):
        """Plot optimal λ* vs topology descriptors with regression analysis."""
        subdir = 'optimal_lambda'

        if self.auc_results is None:
            print("  Warning: No AUC results. Skipping optimal lambda analysis.")
            return

        # Compute optimal lambda for each topology (based on AUC)
        optimal_lambdas = []

        for topology in self.df['topology'].unique():
            topo_auc = self.auc_results[self.auc_results['topology'] == topology]
            avg_auc = topo_auc.groupby('lambda')['auc'].mean()
            best_lambda = avg_auc.idxmax()
            best_auc = avg_auc.max()

            # Get topology metrics
            topo_data = self.df[self.df['topology'] == topology].iloc[0]

            optimal_lambdas.append({
                'topology': topology,
                'optimal_lambda': best_lambda,
                'best_auc': best_auc,
                'mean_corridor_length': topo_data['topo_mean_corridor_length'],
                'junction_density': topo_data['topo_junction_density'],
                'frac_corridors': topo_data['topo_frac_corridors'],
                'frac_junctions': topo_data['topo_frac_junctions'],
            })

        opt_df = pd.DataFrame(optimal_lambdas)

        # Create 2x2 panel of regression plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Lambda* vs mean corridor length
        ax = axes[0, 0]
        x = opt_df['mean_corridor_length']
        y = opt_df['optimal_lambda']

        ax.scatter(x, y, s=100, alpha=0.7, c=range(len(opt_df)), cmap='tab10')

        # Add regression line
        if len(x) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'r--', linewidth=2, alpha=0.7,
                   label=f'R²={r_value**2:.3f}, p={p_value:.3f}')

        ax.set_xlabel('Mean Corridor Length')
        ax.set_ylabel('Optimal λ*')
        ax.set_title('Corridor Hypothesis: λ* vs Corridor Length')
        ax.grid(alpha=0.3)
        ax.legend()

        # Annotate points
        for idx, row in opt_df.iterrows():
            ax.annotate(row['topology'], (row['mean_corridor_length'], row['optimal_lambda']),
                       fontsize=7, alpha=0.7, xytext=(3, 3), textcoords='offset points')

        # 2. Lambda* vs junction density
        ax = axes[0, 1]
        x = opt_df['junction_density']
        y = opt_df['optimal_lambda']

        ax.scatter(x, y, s=100, alpha=0.7, c=range(len(opt_df)), cmap='tab10')

        if len(x) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'r--', linewidth=2, alpha=0.7,
                   label=f'R²={r_value**2:.3f}, p={p_value:.3f}')

        ax.set_xlabel('Junction Density')
        ax.set_ylabel('Optimal λ*')
        ax.set_title('Junction Hypothesis: λ* vs Junction Density')
        ax.grid(alpha=0.3)
        ax.legend()

        for idx, row in opt_df.iterrows():
            ax.annotate(row['topology'], (row['junction_density'], row['optimal_lambda']),
                       fontsize=7, alpha=0.7, xytext=(3, 3), textcoords='offset points')

        # 3. Lambda* vs corridor fraction
        ax = axes[1, 0]
        x = opt_df['frac_corridors']
        y = opt_df['optimal_lambda']

        ax.scatter(x, y, s=100, alpha=0.7, c=range(len(opt_df)), cmap='tab10')

        if len(x) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'r--', linewidth=2, alpha=0.7,
                   label=f'R²={r_value**2:.3f}, p={p_value:.3f}')

        ax.set_xlabel('Fraction of Corridor Nodes')
        ax.set_ylabel('Optimal λ*')
        ax.set_title('λ* vs Corridor Fraction')
        ax.grid(alpha=0.3)
        ax.legend()

        for idx, row in opt_df.iterrows():
            ax.annotate(row['topology'], (row['frac_corridors'], row['optimal_lambda']),
                       fontsize=7, alpha=0.7, xytext=(3, 3), textcoords='offset points')

        # 4. Summary bar chart of optimal lambdas
        ax = axes[1, 1]
        topologies = opt_df['topology']
        optimal_vals = opt_df['optimal_lambda']

        bars = ax.barh(range(len(topologies)), optimal_vals,
                      color=TOPOLOGY_COLORS[:len(topologies)])
        ax.set_yticks(range(len(topologies)))
        ax.set_yticklabels(topologies)
        ax.set_xlabel('Optimal λ*')
        ax.set_title('Optimal λ* by Topology')
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3, axis='x')

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, optimal_vals)):
            ax.text(val + 0.02, i, f'{val:.2f}', va='center', fontsize=9)

        plt.tight_layout()
        self._save_fig(fig, 'optimal_lambda_analysis', subdir)
        plt.close()

        # Additional plot: Performance landscape for each topology
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        topologies = sorted(self.df['topology'].unique())

        for idx, topology in enumerate(topologies):
            ax = axes[idx]
            topo_auc = self.auc_results[self.auc_results['topology'] == topology]

            # Get mean and std across seeds
            auc_stats = topo_auc.groupby('lambda')['auc'].agg(['mean', 'std', 'count'])
            lambdas = auc_stats.index
            means = auc_stats['mean']
            stds = auc_stats['std']
            ns = auc_stats['count']

            ci = 1.96 * stds / np.sqrt(ns)

            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8, color='steelblue')
            ax.fill_between(lambdas, means - ci, means + ci, alpha=0.3, color='steelblue')

            # Mark optimal
            best_idx = means.idxmax()
            ax.axvline(x=best_idx, color='red', linestyle='--', linewidth=1.5,
                      alpha=0.7, label=f'Optimal λ*={best_idx:.1f}')

            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('AUC')
            ax.set_title(topology.replace('_', ' ').title())
            ax.grid(alpha=0.3)
            ax.legend()
            ax.set_xlim(-0.05, 1.05)

        for i in range(len(topologies), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        self._save_fig(fig, 'performance_landscape_by_topology', subdir)
        plt.close()

    # =========================================================================
    # FIGURE 4: Credit Assignment Diagnostics
    # =========================================================================

    def plot_credit_diagnostics(self):
        """Plot credit assignment metrics."""
        subdir = 'credit'

        # Filter for successful episodes only (where credit diagnostics are computed)
        success_df = self.df[self.df['success'] == True].copy()

        if len(success_df) == 0:
            print("  Warning: No successful episodes. Skipping credit diagnostics.")
            return

        topologies = sorted(success_df['topology'].unique())
        lambda_values = sorted(success_df['lambda'].unique())

        # 1. Effective credit distance vs lambda
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, topology in enumerate(topologies):
            ax = axes[idx]
            topo_data = success_df[success_df['topology'] == topology]

            credit_dist_stats = topo_data.groupby('lambda')['effective_credit_distance'].agg(
                ['mean', 'std', 'count']
            )

            lambdas = credit_dist_stats.index
            means = credit_dist_stats['mean']
            stds = credit_dist_stats['std']
            ns = credit_dist_stats['count']

            ci = 1.96 * stds / np.sqrt(ns)

            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8, color='darkgreen')
            ax.fill_between(lambdas, means - ci, means + ci, alpha=0.3, color='darkgreen')

            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Effective Credit Distance')
            ax.set_title(topology.replace('_', ' ').title())
            ax.grid(alpha=0.3)

        for i in range(len(topologies), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        self._save_fig(fig, 'effective_credit_distance', subdir)
        plt.close()

        # 2. Junction/corridor credit ratio vs lambda
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, topology in enumerate(topologies):
            ax = axes[idx]
            topo_data = success_df[success_df['topology'] == topology]

            ratio_stats = topo_data.groupby('lambda')['junction_corridor_ratio'].agg(
                ['mean', 'std', 'count']
            )

            if len(ratio_stats) == 0:
                continue

            lambdas = ratio_stats.index
            means = ratio_stats['mean']
            stds = ratio_stats['std']
            ns = ratio_stats['count']

            ci = 1.96 * stds / np.sqrt(ns)

            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8, color='darkorange')
            ax.fill_between(lambdas, means - ci, means + ci, alpha=0.3, color='darkorange')

            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Junction/Corridor Credit Ratio')
            ax.set_title(topology.replace('_', ' ').title())
            ax.grid(alpha=0.3)
            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        for i in range(len(topologies), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        self._save_fig(fig, 'junction_corridor_credit_ratio', subdir)
        plt.close()

        # 3. Credit concentration by node type (stacked bars)
        # Average across final 50 episodes for each condition
        final_episodes = success_df[success_df['episode'] >= success_df['episode'].max() - 50]

        credit_by_type = final_episodes.groupby(['topology', 'lambda']).agg({
            'C_corridor': 'mean',
            'C_junction': 'mean',
            'C_dead_end': 'mean'
        }).reset_index()

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, topology in enumerate(topologies):
            ax = axes[idx]
            topo_credit = credit_by_type[credit_by_type['topology'] == topology]

            if len(topo_credit) == 0:
                continue

            lambdas = topo_credit['lambda']
            corridors = topo_credit['C_corridor']
            junctions = topo_credit['C_junction']
            dead_ends = topo_credit['C_dead_end']

            width = 0.08
            x = np.arange(len(lambdas))

            ax.bar(x, corridors, width, label='Corridor', color='skyblue')
            ax.bar(x, junctions, width, bottom=corridors, label='Junction', color='orange')
            ax.bar(x, dead_ends, width, bottom=corridors + junctions,
                  label='Dead-end', color='lightcoral')

            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Credit Magnitude')
            ax.set_title(topology.replace('_', ' ').title())
            ax.set_xticks(x)
            ax.set_xticklabels([f'{l:.1f}' for l in lambdas], rotation=45)
            ax.legend()
            ax.grid(alpha=0.3, axis='y')

        for i in range(len(topologies), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        self._save_fig(fig, 'credit_by_node_type', subdir)
        plt.close()

        # 4. Decision localization score vs lambda
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, topology in enumerate(topologies):
            ax = axes[idx]
            topo_data = success_df[success_df['topology'] == topology]

            loc_stats = topo_data.groupby('lambda')['mean_decision_localization'].agg(
                ['mean', 'std', 'count']
            )

            if len(loc_stats) == 0:
                continue

            lambdas = loc_stats.index
            means = loc_stats['mean']
            stds = loc_stats['std']
            ns = loc_stats['count']

            ci = 1.96 * stds / np.sqrt(ns)

            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8, color='purple')
            ax.fill_between(lambdas, means - ci, means + ci, alpha=0.3, color='purple')

            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Decision Localization Score')
            ax.set_title(topology.replace('_', ' ').title())
            ax.grid(alpha=0.3)

        for i in range(len(topologies), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        self._save_fig(fig, 'decision_localization', subdir)
        plt.close()

    # =========================================================================
    # FIGURE 5: Decision Quality Metrics
    # =========================================================================

    def plot_decision_quality(self):
        """Plot junction decision quality metrics."""
        subdir = 'decision_quality'

        topologies = sorted(self.df['topology'].unique())

        # 1. Junction accuracy vs lambda
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, topology in enumerate(topologies):
            ax = axes[idx]
            topo_data = self.df[self.df['topology'] == topology]

            acc_stats = topo_data.groupby('lambda')['junction_accuracy'].agg(
                ['mean', 'std', 'count']
            )

            lambdas = acc_stats.index
            means = acc_stats['mean']
            stds = acc_stats['std']
            ns = acc_stats['count']

            ci = 1.96 * stds / np.sqrt(ns)

            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8, color='darkblue')
            ax.fill_between(lambdas, means - ci, means + ci, alpha=0.3, color='darkblue')

            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Junction Accuracy')
            ax.set_title(topology.replace('_', ' ').title())
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

        for i in range(len(topologies), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        self._save_fig(fig, 'junction_accuracy', subdir)
        plt.close()

        # 2. Wrong turn rate vs lambda
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, topology in enumerate(topologies):
            ax = axes[idx]
            topo_data = self.df[self.df['topology'] == topology]

            wtr_stats = topo_data.groupby('lambda')['wrong_turn_rate'].agg(
                ['mean', 'std', 'count']
            )

            lambdas = wtr_stats.index
            means = wtr_stats['mean']
            stds = wtr_stats['std']
            ns = wtr_stats['count']

            ci = 1.96 * stds / np.sqrt(ns)

            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8, color='darkred')
            ax.fill_between(lambdas, means - ci, means + ci, alpha=0.3, color='darkred')

            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Wrong Turn Rate')
            ax.set_title(topology.replace('_', ' ').title())
            ax.grid(alpha=0.3)

        for i in range(len(topologies), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        self._save_fig(fig, 'wrong_turn_rate', subdir)
        plt.close()

        # 3. Junction vs corridor entropy comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, topology in enumerate(topologies):
            ax = axes[idx]
            topo_data = self.df[self.df['topology'] == topology]

            junc_ent = topo_data.groupby('lambda')['mean_junction_entropy'].mean()
            corr_ent = topo_data.groupby('lambda')['mean_corridor_entropy'].mean()

            lambdas = junc_ent.index

            ax.plot(lambdas, junc_ent, 'o-', linewidth=2, markersize=8,
                   color='orange', label='Junction')
            ax.plot(lambdas, corr_ent, 's-', linewidth=2, markersize=8,
                   color='skyblue', label='Corridor')

            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Policy Entropy')
            ax.set_title(topology.replace('_', ' ').title())
            ax.grid(alpha=0.3)
            ax.legend()

        for i in range(len(topologies), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        self._save_fig(fig, 'junction_vs_corridor_entropy', subdir)
        plt.close()

        # 4. Policy margin at junctions
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, topology in enumerate(topologies):
            ax = axes[idx]
            topo_data = self.df[self.df['topology'] == topology]

            margin_stats = topo_data.groupby('lambda')['mean_junction_margin'].agg(
                ['mean', 'std', 'count']
            )

            if len(margin_stats) == 0:
                continue

            lambdas = margin_stats.index
            means = margin_stats['mean']
            stds = margin_stats['std']
            ns = margin_stats['count']

            ci = 1.96 * stds / np.sqrt(ns)

            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8, color='teal')
            ax.fill_between(lambdas, means - ci, means + ci, alpha=0.3, color='teal')

            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Mean Junction Policy Margin')
            ax.set_title(topology.replace('_', ' ').title())
            ax.grid(alpha=0.3)

        for i in range(len(topologies), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        self._save_fig(fig, 'junction_policy_margin', subdir)
        plt.close()

    # =========================================================================
    # FIGURE 6: Topology Descriptors
    # =========================================================================

    def plot_topology_descriptors(self):
        """Plot topology characterization."""
        subdir = 'topology'

        # Get one sample per topology
        topo_samples = self.df.groupby('topology').first().reset_index()

        topologies = sorted(topo_samples['topology'].unique())

        # 1. Node type composition (stacked bars)
        fig, ax = plt.subplots(figsize=(12, 6))

        dead_ends = topo_samples.set_index('topology')['topo_num_dead_ends']
        corridors = topo_samples.set_index('topology')['topo_num_corridors']
        junctions = topo_samples.set_index('topology')['topo_num_junctions']

        x = np.arange(len(topologies))
        width = 0.6

        ax.bar(x, dead_ends, width, label='Dead-ends', color='lightcoral')
        ax.bar(x, corridors, width, bottom=dead_ends, label='Corridors', color='skyblue')
        ax.bar(x, junctions, width, bottom=dead_ends + corridors,
              label='Junctions', color='orange')

        ax.set_xlabel('Topology')
        ax.set_ylabel('Node Count')
        ax.set_title('Node Type Composition by Topology')
        ax.set_xticks(x)
        ax.set_xticklabels(topologies, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        self._save_fig(fig, 'node_type_composition', subdir)
        plt.close()

        # 2. Key topology metrics panel
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Mean corridor length
        ax = axes[0]
        corridor_lengths = topo_samples.set_index('topology')['topo_mean_corridor_length']
        bars = ax.barh(range(len(topologies)), corridor_lengths, color=TOPOLOGY_COLORS[:len(topologies)])
        ax.set_yticks(range(len(topologies)))
        ax.set_yticklabels(topologies)
        ax.set_xlabel('Mean Corridor Length')
        ax.set_title('Mean Corridor Length by Topology')
        ax.grid(alpha=0.3, axis='x')

        # Junction density
        ax = axes[1]
        junction_dens = topo_samples.set_index('topology')['topo_junction_density']
        bars = ax.barh(range(len(topologies)), junction_dens, color=TOPOLOGY_COLORS[:len(topologies)])
        ax.set_yticks(range(len(topologies)))
        ax.set_yticklabels([])
        ax.set_xlabel('Junction Density')
        ax.set_title('Junction Density by Topology')
        ax.grid(alpha=0.3, axis='x')

        # Corridor fraction
        ax = axes[2]
        corridor_frac = topo_samples.set_index('topology')['topo_frac_corridors']
        bars = ax.barh(range(len(topologies)), corridor_frac, color=TOPOLOGY_COLORS[:len(topologies)])
        ax.set_yticks(range(len(topologies)))
        ax.set_yticklabels([])
        ax.set_xlabel('Fraction of Corridor Nodes')
        ax.set_title('Corridor Fraction by Topology')
        ax.grid(alpha=0.3, axis='x')
        ax.set_xlim(0, 1)

        plt.tight_layout()
        self._save_fig(fig, 'topology_metrics_panel', subdir)
        plt.close()

        # 3. Correlation matrix of topology descriptors
        topo_features = topo_samples[[
            'topo_mean_corridor_length',
            'topo_junction_density',
            'topo_frac_corridors',
            'topo_frac_junctions'
        ]]

        corr_matrix = topo_features.corr()

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, vmin=-1, vmax=1, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Matrix of Topology Descriptors')

        plt.tight_layout()
        self._save_fig(fig, 'topology_correlation_matrix', subdir)
        plt.close()

    # =========================================================================
    # FIGURE 7: Detailed Diagnostics
    # =========================================================================

    def plot_detailed_diagnostics(self):
        """Plot training stability and detailed diagnostics."""
        subdir = 'diagnostics'

        topologies = sorted(self.df['topology'].unique())
        lambda_values = sorted(self.df['lambda'].unique())

        # 1. Loss components over training
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Pick one representative topology and lambda
        rep_topology = topologies[0]
        rep_lambda = lambda_values[len(lambda_values) // 2]  # Middle lambda

        rep_data = self.df[(self.df['topology'] == rep_topology) &
                          (self.df['lambda'] == rep_lambda)]

        # Total loss
        ax = axes[0, 0]
        for seed in sorted(rep_data['seed'].unique()):
            seed_data = rep_data[rep_data['seed'] == seed]
            ax.plot(seed_data['episode'], seed_data['loss'], alpha=0.6, label=f'Seed {seed}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Loss')
        ax.set_title(f'Total Loss ({rep_topology}, λ={rep_lambda})')
        ax.grid(alpha=0.3)
        ax.legend()

        # Policy loss
        ax = axes[0, 1]
        for seed in sorted(rep_data['seed'].unique()):
            seed_data = rep_data[rep_data['seed'] == seed]
            ax.plot(seed_data['episode'], seed_data['policy_loss'], alpha=0.6, label=f'Seed {seed}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Policy Loss')
        ax.set_title(f'Policy Loss ({rep_topology}, λ={rep_lambda})')
        ax.grid(alpha=0.3)
        ax.legend()

        # Value loss
        ax = axes[1, 0]
        for seed in sorted(rep_data['seed'].unique()):
            seed_data = rep_data[rep_data['seed'] == seed]
            ax.plot(seed_data['episode'], seed_data['value_loss'], alpha=0.6, label=f'Seed {seed}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value Loss')
        ax.set_title(f'Value Loss ({rep_topology}, λ={rep_lambda})')
        ax.grid(alpha=0.3)
        ax.legend()

        # Entropy
        ax = axes[1, 1]
        for seed in sorted(rep_data['seed'].unique()):
            seed_data = rep_data[rep_data['seed'] == seed]
            ax.plot(seed_data['episode'], seed_data['mean_entropy'], alpha=0.6, label=f'Seed {seed}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean Entropy')
        ax.set_title(f'Policy Entropy ({rep_topology}, λ={rep_lambda})')
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        self._save_fig(fig, 'training_stability_losses', subdir)
        plt.close()

        # 2. Value estimates over training
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, topology in enumerate(topologies):
            ax = axes[idx]
            topo_data = self.df[self.df['topology'] == topology]

            for lambda_idx, lambda_val in enumerate(lambda_values):
                lambda_data = topo_data[topo_data['lambda'] == lambda_val]

                grouped = lambda_data.groupby('episode')['mean_value'].mean()
                episodes = grouped.index
                values = grouped.values

                color = LAMBDA_COLORS[lambda_idx % len(LAMBDA_COLORS)]
                ax.plot(episodes, values, label=f'λ={lambda_val:.1f}',
                       color=color, linewidth=1.5, alpha=0.7)

            ax.set_xlabel('Episode')
            ax.set_ylabel('Mean Value Estimate')
            ax.set_title(topology.replace('_', ' ').title())
            ax.grid(alpha=0.3)
            ax.legend(ncol=2, fontsize=8)

        for i in range(len(topologies), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        self._save_fig(fig, 'value_estimates_over_training', subdir)
        plt.close()

        # 3. Per-seed variance in final performance
        if self.summary is not None:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()

            for idx, topology in enumerate(topologies):
                ax = axes[idx]
                topo_summary = self.summary[self.summary['topology'] == topology]

                for lambda_val in lambda_values:
                    lambda_summary = topo_summary[topo_summary['lambda'] == lambda_val]

                    if len(lambda_summary) == 0:
                        continue

                    # Plot individual seeds
                    seeds = lambda_summary['seed']
                    successes = lambda_summary['success']

                    # Jitter x-position slightly for visibility
                    x_pos = lambda_val + np.random.normal(0, 0.01, len(seeds))
                    ax.scatter(x_pos, successes, alpha=0.6, s=50, color='steelblue')

                # Add mean line
                mean_success = topo_summary.groupby('lambda')['success'].mean()
                ax.plot(mean_success.index, mean_success.values, 'r-',
                       linewidth=2, alpha=0.8, label='Mean')

                ax.set_xlabel('Lambda (λ)')
                ax.set_ylabel('Final Success Rate')
                ax.set_title(topology.replace('_', ' ').title())
                ax.grid(alpha=0.3)
                ax.legend()
                ax.set_ylim(-0.05, 1.05)

            for i in range(len(topologies), len(axes)):
                fig.delaxes(axes[i])

            plt.tight_layout()
            self._save_fig(fig, 'per_seed_variance', subdir)
            plt.close()

        # 4. Upstream-local credit ratio (credit smearing)
        success_df = self.df[self.df['success'] == True]

        if len(success_df) > 0 and 'mean_upstream_local_ratio' in success_df.columns:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()

            for idx, topology in enumerate(topologies):
                ax = axes[idx]
                topo_data = success_df[success_df['topology'] == topology]

                ratio_stats = topo_data.groupby('lambda')['mean_upstream_local_ratio'].agg(
                    ['mean', 'std', 'count']
                )

                if len(ratio_stats) == 0:
                    continue

                lambdas = ratio_stats.index
                means = ratio_stats['mean']
                stds = ratio_stats['std']
                ns = ratio_stats['count']

                ci = 1.96 * stds / np.sqrt(ns)

                ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8, color='brown')
                ax.fill_between(lambdas, means - ci, means + ci, alpha=0.3, color='brown')

                ax.set_xlabel('Lambda (λ)')
                ax.set_ylabel('Upstream/Local Credit Ratio')
                ax.set_title(topology.replace('_', ' ').title())
                ax.grid(alpha=0.3)
                ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

            for i in range(len(topologies), len(axes)):
                fig.delaxes(axes[i])

            plt.tight_layout()
            self._save_fig(fig, 'upstream_local_credit_ratio', subdir)
            plt.close()


def main():
    """Main entry point for plotting script."""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive plots for lambda experiment results'
    )
    parser.add_argument('results_dir', type=str,
                       help='Directory containing results.csv and aggregate metrics')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for figures (default: figures/)')
    parser.add_argument('--figures', type=str, default='all',
                       help='Comma-separated list of figure categories to plot. '
                            'Options: all, learning, heatmaps, optimal_lambda, credit, '
                            'decision_quality, topology, diagnostics')

    args = parser.parse_args()

    # Parse figure list
    if args.figures == 'all':
        figures = 'all'
    else:
        figures = [f.strip() for f in args.figures.split(',')]

    # Create plotter and generate figures
    plotter = LambdaExperimentPlotter(args.results_dir, args.output_dir)
    plotter.plot_all(figures=figures)

    print("\n✓ Plotting complete!")
    print(f"  View figures in: {plotter.output_dir}/")


if __name__ == '__main__':
    main()
