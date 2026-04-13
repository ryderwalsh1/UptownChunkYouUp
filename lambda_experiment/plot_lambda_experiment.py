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
# Use only 0.1 to 0.9 range of plasma to avoid similar colors at endpoints
LAMBDA_COLORS = plt.cm.plasma(np.linspace(0.1, 0.9, 10))
TOPOLOGY_COLORS = sns.color_palette("Set2", 8)


class LambdaExperimentPlotter:
    """Main plotting class for lambda experiment analysis."""

    def __init__(self, results_dir, output_dir='figures', learning_curve_window=200, num_bins=4):
        """
        Initialize plotter with results directory.

        Parameters:
        -----------
        results_dir : str or Path
            Directory containing results.csv and aggregate metrics
        output_dir : str or Path
            Directory to save generated figures
        learning_curve_window : int
            Window size for running-average smoothing in learning curves
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.learning_curve_window = learning_curve_window
        self.num_bins = num_bins

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

    def _compute_graph_bins(self):
        """
        Compute bins for individual graphs based on junction density.

        Returns a dataframe with columns: topology, seed, junction_density, bin_id, bin_label
        """
        # Get unique (topology, seed) combinations with their junction density
        graph_info = []
        num_bins = self.num_bins
        for (topology, seed), group in self.df.groupby(['topology', 'seed']):
            # Junction density should be constant for a given (topology, seed)
            junc_density = group.iloc[0]['topo_junction_density']
            graph_info.append({
                'topology': topology,
                'seed': seed,
                'junction_density': junc_density
            })

        graph_df = pd.DataFrame(graph_info)

        # Create bins based on junction density quantiles
        graph_df['bin_id'] = pd.qcut(graph_df['junction_density'],
                                      q=num_bins,
                                      labels=False,
                                      duplicates='drop')

        # Create readable bin labels
        bin_edges = pd.qcut(graph_df['junction_density'],
                           q=num_bins,
                           duplicates='drop',
                           retbins=True)[1]

        def make_bin_label(bin_id):
            if bin_id >= len(bin_edges) - 1:
                bin_id = len(bin_edges) - 2
            low = bin_edges[bin_id]
            high = bin_edges[bin_id + 1]
            return f'Junction Density: {low:.2f}-{high:.2f}'

        graph_df['bin_label'] = graph_df['bin_id'].apply(make_bin_label)

        return graph_df

    def _running_average(self, series, window=None):
        """Compute running average with a centered rolling window."""
        if window is None:
            window = self.learning_curve_window
        return series.rolling(window=window, min_periods=1, center=True).mean()

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
        """Plot smoothed learning curves across λ for each junction density bin."""
        subdir = 'learning'

        # Compute graph bins
        graph_bins = self._compute_graph_bins()
        num_bins = graph_bins['bin_id'].nunique()

        lambda_values = sorted(self.df['lambda'].unique())
        window = self.learning_curve_window

        # 1. Success rate learning curves
        fig, axes = plt.subplots(1, num_bins, figsize=(5*num_bins, 5))
        if num_bins == 1:
            axes = [axes]

        for bin_id in range(num_bins):
            ax = axes[bin_id]
            bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
            bin_label = bin_graphs.iloc[0]['bin_label']

            # Collect all data for graphs in this bin
            bin_data_list = []
            for _, graph_row in bin_graphs.iterrows():
                topology = graph_row['topology']
                seed = graph_row['seed']

                graph_data = self.df[
                    (self.df['topology'] == topology) &
                    (self.df['seed'] == seed)
                ]

                if len(graph_data) > 0:
                    bin_data_list.append(graph_data)

            if len(bin_data_list) == 0:
                continue

            bin_data = pd.concat(bin_data_list)

            for lambda_idx, lambda_val in enumerate(lambda_values):
                lambda_data = bin_data[bin_data['lambda'] == lambda_val]

                if len(lambda_data) == 0:
                    continue

                grouped = lambda_data.groupby('episode')['success'].agg(['mean', 'std', 'count'])
                episodes = grouped.index.to_numpy()
                mean = grouped['mean']
                std = grouped['std'].fillna(0)
                n = grouped['count']
                ci = 1.96 * std / np.sqrt(n)

                mean_smooth = self._running_average(mean, window)
                ci_smooth = self._running_average(ci, window)

                color = LAMBDA_COLORS[lambda_idx % len(LAMBDA_COLORS)]
                ax.plot(episodes, mean_smooth, label=f'λ={lambda_val:.1f}',
                        color=color, linewidth=1.5, alpha=0.8)
                ax.fill_between(episodes, mean_smooth - ci_smooth, mean_smooth + ci_smooth,
                                color=color, alpha=0.15)

            ax.set_xlabel('Episode')
            ax.set_ylabel(f'Success Rate\n({window}-ep running avg)')
            ax.set_title(bin_label)
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.legend(ncol=2, fontsize=8)
            ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        self._save_fig(fig, 'learning_curves_success', subdir)
        plt.close()

        # 2. Reward learning curves
        fig, axes = plt.subplots(1, num_bins, figsize=(5*num_bins, 5))
        if num_bins == 1:
            axes = [axes]

        for bin_id in range(num_bins):
            ax = axes[bin_id]
            bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
            bin_label = bin_graphs.iloc[0]['bin_label']

            # Collect all data for graphs in this bin
            bin_data_list = []
            for _, graph_row in bin_graphs.iterrows():
                topology = graph_row['topology']
                seed = graph_row['seed']

                graph_data = self.df[
                    (self.df['topology'] == topology) &
                    (self.df['seed'] == seed)
                ]

                if len(graph_data) > 0:
                    bin_data_list.append(graph_data)

            if len(bin_data_list) == 0:
                continue

            bin_data = pd.concat(bin_data_list)

            for lambda_idx, lambda_val in enumerate(lambda_values):
                lambda_data = bin_data[bin_data['lambda'] == lambda_val]

                if len(lambda_data) == 0:
                    continue

                grouped = lambda_data.groupby('episode')['episode_reward'].agg(['mean', 'std', 'count'])
                episodes = grouped.index.to_numpy()
                mean = grouped['mean']
                std = grouped['std'].fillna(0)
                n = grouped['count']
                ci = 1.96 * std / np.sqrt(n)

                mean_smooth = self._running_average(mean, window)
                ci_smooth = self._running_average(ci, window)

                color = LAMBDA_COLORS[lambda_idx % len(LAMBDA_COLORS)]
                ax.plot(episodes, mean_smooth, label=f'λ={lambda_val:.1f}',
                        color=color, linewidth=1.5, alpha=0.8)
                ax.fill_between(episodes, mean_smooth - ci_smooth, mean_smooth + ci_smooth,
                                color=color, alpha=0.15)

            ax.set_xlabel('Episode')
            ax.set_ylabel(f'Episode Reward\n({window}-ep running avg)')
            ax.set_title(bin_label)
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.legend(ncol=2, fontsize=8)

        plt.tight_layout()
        self._save_fig(fig, 'learning_curves_reward', subdir)
        plt.close()

        # 3. Optimality ratio over training (for successful episodes)
        fig, axes = plt.subplots(1, num_bins, figsize=(5*num_bins, 5))
        if num_bins == 1:
            axes = [axes]

        for bin_id in range(num_bins):
            ax = axes[bin_id]
            bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
            bin_label = bin_graphs.iloc[0]['bin_label']

            # Collect all data for graphs in this bin (successful episodes only)
            bin_data_list = []
            for _, graph_row in bin_graphs.iterrows():
                topology = graph_row['topology']
                seed = graph_row['seed']

                graph_data = self.df[
                    (self.df['topology'] == topology) &
                    (self.df['seed'] == seed) &
                    (self.df['success'] == True)
                ]

                if len(graph_data) > 0:
                    bin_data_list.append(graph_data)

            if len(bin_data_list) == 0:
                ax.text(0.5, 0.5, 'No successful episodes',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(bin_label)
                continue

            bin_data = pd.concat(bin_data_list)

            for lambda_idx, lambda_val in enumerate(lambda_values):
                lambda_data = bin_data[bin_data['lambda'] == lambda_val]

                if len(lambda_data) == 0:
                    continue

                grouped = lambda_data.groupby('episode')['optimality_ratio'].agg(['mean', 'std', 'count'])
                episodes = grouped.index.to_numpy()
                mean = grouped['mean']
                std = grouped['std'].fillna(0)
                n = grouped['count']
                ci = 1.96 * std / np.sqrt(n)

                mean_smooth = self._running_average(mean, window)
                ci_smooth = self._running_average(ci, window)

                color = LAMBDA_COLORS[lambda_idx % len(LAMBDA_COLORS)]
                ax.plot(episodes, mean_smooth, label=f'λ={lambda_val:.1f}',
                        color=color, linewidth=1.5, alpha=0.8)
                ax.fill_between(episodes, mean_smooth - ci_smooth, mean_smooth + ci_smooth,
                                color=color, alpha=0.15)

            ax.set_xlabel('Episode')
            ax.set_ylabel(f'Optimality Ratio\n({window}-ep running avg)')
            ax.set_title(bin_label)
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Optimal')
            ax.legend(ncol=2, fontsize=8)

        plt.tight_layout()
        self._save_fig(fig, 'learning_curves_optimality', subdir)
        plt.close()

    # =========================================================================
    # FIGURE 2: Performance Heatmaps
    # =========================================================================

    def plot_performance_heatmaps(self):
        """Plot heatmaps of performance metrics vs (λ, bin) organized by junction density."""
        subdir = 'heatmaps'

        if self.auc_results is None or self.summary is None:
            print("  Warning: Missing aggregate metrics files. Skipping heatmaps.")
            return

        # Get graph bins
        graph_bins = self._compute_graph_bins()
        num_bins = graph_bins['bin_id'].nunique()
        lambda_values = sorted(self.df['lambda'].unique())

        # Create bin labels
        bin_labels = []
        for bin_id in range(num_bins):
            bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
            bin_label = bin_graphs.iloc[0]['bin_label']
            bin_labels.append(bin_label)

        # 1. AUC heatmap
        if self.auc_results is not None:
            auc_data = []
            for bin_id in range(num_bins):
                bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
                bin_label = bin_graphs.iloc[0]['bin_label']

                for lambda_val in lambda_values:
                    # Collect all data for this bin and lambda
                    bin_auc_list = []
                    for _, graph_row in bin_graphs.iterrows():
                        topology = graph_row['topology']
                        seed = graph_row['seed']

                        graph_auc = self.auc_results[
                            (self.auc_results['topology'] == topology) &
                            (self.auc_results['seed'] == seed) &
                            (self.auc_results['lambda'] == lambda_val)
                        ]

                        if len(graph_auc) > 0:
                            bin_auc_list.append(graph_auc['auc'].mean())

                    if len(bin_auc_list) > 0:
                        auc_data.append({
                            'bin': bin_label,
                            'lambda': lambda_val,
                            'auc': np.mean(bin_auc_list)
                        })

            if len(auc_data) > 0:
                auc_df = pd.DataFrame(auc_data)
                auc_pivot = auc_df.pivot(index='bin', columns='lambda', values='auc')
                auc_pivot = auc_pivot.reindex(bin_labels)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(auc_pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                           cbar_kws={'label': 'AUC (Success Rate)'}, ax=ax)
                ax.set_xlabel('Lambda (λ)')
                ax.set_ylabel('Junction Density Bin')
                ax.set_title('Area Under Learning Curve (AUC) by Junction Density and λ')
                plt.tight_layout()
                self._save_fig(fig, 'heatmap_auc', subdir)
                plt.close()

        # 2. Final success rate heatmap
        success_data = []
        for bin_id in range(num_bins):
            bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
            bin_label = bin_graphs.iloc[0]['bin_label']

            for lambda_val in lambda_values:
                # Collect all data for this bin and lambda
                bin_success_list = []
                for _, graph_row in bin_graphs.iterrows():
                    topology = graph_row['topology']
                    seed = graph_row['seed']

                    graph_summary = self.summary[
                        (self.summary['topology'] == topology) &
                        (self.summary['seed'] == seed) &
                        (self.summary['lambda'] == lambda_val)
                    ]

                    if len(graph_summary) > 0:
                        bin_success_list.append(graph_summary['success'].mean())

                if len(bin_success_list) > 0:
                    success_data.append({
                        'bin': bin_label,
                        'lambda': lambda_val,
                        'success': np.mean(bin_success_list)
                    })

        if len(success_data) > 0:
            success_df = pd.DataFrame(success_data)
            final_success = success_df.pivot(index='bin', columns='lambda', values='success')
            final_success = final_success.reindex(bin_labels)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(final_success, annot=True, fmt='.2f', cmap='RdYlGn',
                       cbar_kws={'label': 'Final Success Rate'}, ax=ax,
                       vmin=0, vmax=1)
            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Junction Density Bin')
            ax.set_title('Final Success Rate by Junction Density and λ')
            plt.tight_layout()
            self._save_fig(fig, 'heatmap_final_success', subdir)
            plt.close()

        # 3. Steps to threshold heatmap
        if self.steps_threshold is not None and 'steps_to_80pct' in self.steps_threshold.columns:
            steps_data = []
            for bin_id in range(num_bins):
                bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
                bin_label = bin_graphs.iloc[0]['bin_label']

                for lambda_val in lambda_values:
                    # Collect all data for this bin and lambda
                    bin_steps_list = []
                    for _, graph_row in bin_graphs.iterrows():
                        topology = graph_row['topology']
                        seed = graph_row['seed']

                        graph_steps = self.steps_threshold[
                            (self.steps_threshold['topology'] == topology) &
                            (self.steps_threshold['seed'] == seed) &
                            (self.steps_threshold['lambda'] == lambda_val)
                        ]

                        if len(graph_steps) > 0:
                            bin_steps_list.append(graph_steps['steps_to_80pct'].mean())

                    if len(bin_steps_list) > 0:
                        steps_data.append({
                            'bin': bin_label,
                            'lambda': lambda_val,
                            'steps': np.mean(bin_steps_list)
                        })

            if len(steps_data) > 0:
                steps_df = pd.DataFrame(steps_data)
                steps_pivot = steps_df.pivot(index='bin', columns='lambda', values='steps')
                steps_pivot = steps_pivot.reindex(bin_labels)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(steps_pivot, annot=True, fmt='.0f', cmap='RdYlGn_r',
                           cbar_kws={'label': 'Steps to 80% Success'}, ax=ax)
                ax.set_xlabel('Lambda (λ)')
                ax.set_ylabel('Junction Density Bin')
                ax.set_title('Steps to Reach 80% Success Rate by Junction Density and λ')
                plt.tight_layout()
                self._save_fig(fig, 'heatmap_steps_to_80pct', subdir)
                plt.close()

        # 4. Combined heatmap with junction accuracy (bin-averaged)
        if 'junction_accuracy' in self.summary.columns and len(success_data) > 0:
            # Create junction accuracy data
            junc_acc_data = []
            for bin_id in range(num_bins):
                bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
                bin_label = bin_graphs.iloc[0]['bin_label']

                for lambda_val in lambda_values:
                    # Collect all data for this bin and lambda
                    bin_junc_acc_list = []
                    for _, graph_row in bin_graphs.iterrows():
                        topology = graph_row['topology']
                        seed = graph_row['seed']

                        graph_summary = self.summary[
                            (self.summary['topology'] == topology) &
                            (self.summary['seed'] == seed) &
                            (self.summary['lambda'] == lambda_val)
                        ]

                        if len(graph_summary) > 0:
                            bin_junc_acc_list.append(graph_summary['junction_accuracy'].mean())

                    if len(bin_junc_acc_list) > 0:
                        junc_acc_data.append({
                            'bin': bin_label,
                            'lambda': lambda_val,
                            'junction_accuracy': np.mean(bin_junc_acc_list)
                        })

            if len(junc_acc_data) > 0:
                junc_acc_df = pd.DataFrame(junc_acc_data)
                junction_acc = junc_acc_df.pivot(index='bin', columns='lambda', values='junction_accuracy')
                junction_acc = junction_acc.reindex(bin_labels)

                fig, axes = plt.subplots(1, 2, figsize=(18, 6))

                # Left: Final success rate
                ax = axes[0]
                sns.heatmap(final_success, annot=True, fmt='.2f', cmap='RdYlGn',
                           cbar_kws={'label': 'Final Success Rate'}, ax=ax,
                           vmin=0, vmax=1)
                ax.set_xlabel('Lambda (λ)')
                ax.set_ylabel('Junction Density Bin')
                ax.set_title('Final Success Rate')

                # Right: Junction accuracy
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

        # Compute optimal lambda for each (topology, seed) combination using composite metric
        optimal_lambdas = []

        for (topology, seed) in self.df.groupby(['topology', 'seed']).groups.keys():
            # Get per-lambda metrics for this specific graph
            graph_data = self.df[
                (self.df['topology'] == topology) &
                (self.df['seed'] == seed)
            ]

            # Get AUC results for this specific graph
            graph_auc = self.auc_results[
                (self.auc_results['topology'] == topology) &
                (self.auc_results['seed'] == seed)
            ]

            if len(graph_auc) == 0:
                continue

            # Compute composite optimal score for each lambda
            composite_scores = []
            for lambda_val in sorted(graph_data['lambda'].unique()):
                # Filter to final 20% of episodes for stable metrics
                lambda_data = graph_data[graph_data['lambda'] == lambda_val]
                num_episodes = len(lambda_data)
                final_episodes = lambda_data.iloc[int(0.8 * num_episodes):]

                # Get AUC for this lambda
                lambda_auc = graph_auc[graph_auc['lambda'] == lambda_val]
                if len(lambda_auc) == 0:
                    continue
                auc = lambda_auc['auc'].values[0]

                # Compute normalized AUC (0-1 scale relative to this graph's range)
                auc_range = graph_auc['auc'].max() - graph_auc['auc'].min()
                if auc_range > 0:
                    auc_norm = (auc - graph_auc['auc'].min()) / auc_range
                else:
                    auc_norm = 1.0

                # Get final performance metrics (mean over final 20% of training)
                junction_acc = final_episodes['junction_accuracy'].mean()
                consultation_rate = final_episodes['consultation_rate'].mean()
                wrong_turn_rate = final_episodes['wrong_turn_rate'].mean()

                # Get credit assignment metric (mean over successful episodes in final 20%)
                success_final = final_episodes[final_episodes['success'] == True]
                if len(success_final) > 0:
                    jc_ratio = success_final['junction_corridor_ratio'].mean()
                    # Normalize junction/corridor ratio: clamp extreme values
                    jc_ratio_clamped = np.clip(jc_ratio, 0, 2)
                    jc_ratio_norm = jc_ratio_clamped / 2.0
                else:
                    jc_ratio_norm = 0.0

                # Handle NaN values
                junction_acc = junction_acc if not np.isnan(junction_acc) else 0.0
                consultation_rate = consultation_rate if not np.isnan(consultation_rate) else 1.0
                wrong_turn_rate = wrong_turn_rate if not np.isnan(wrong_turn_rate) else 1.0

                # Compute composite score
                composite_score = (
                    0.30 * auc_norm +
                    0.25 * junction_acc +
                    0.20 * (1 - consultation_rate) +
                    0.15 * (1 - wrong_turn_rate) +
                    0.10 * jc_ratio_norm
                )

                composite_scores.append({
                    'lambda': lambda_val,
                    'composite_score': composite_score,
                    'auc': auc,
                    'auc_norm': auc_norm,
                    'junction_acc': junction_acc,
                    'consultation_rate': consultation_rate,
                    'wrong_turn_rate': wrong_turn_rate,
                    'jc_ratio_norm': jc_ratio_norm
                })

            if len(composite_scores) == 0:
                continue

            # Find best lambda based on composite score
            composite_df = pd.DataFrame(composite_scores)
            best_idx = composite_df['composite_score'].idxmax()
            best_row = composite_df.loc[best_idx]
            best_lambda = best_row['lambda']
            best_score = best_row['composite_score']

            # Get topology metrics for this graph
            topo_data = graph_data.iloc[0]

            optimal_lambdas.append({
                'topology': topology,
                'seed': seed,
                'optimal_lambda': best_lambda,
                'best_composite_score': best_score,
                'best_auc': best_row['auc'],
                'mean_corridor_length': topo_data['topo_mean_corridor_length'],
                'junction_density': topo_data['topo_junction_density'],
                'frac_corridors': topo_data['topo_frac_corridors'],
                'frac_junctions': topo_data['topo_frac_junctions'],
            })

        opt_df = pd.DataFrame(optimal_lambdas)

        # Extract corridor parameter from topology name (e.g., "0.3_corridor" -> 0.3)
        opt_df['corridor_param'] = opt_df['topology'].str.extract(r'([\d.]+)').astype(float)

        # Define corridor-dominated regime threshold
        CORRIDOR_THRESHOLD = 0.0
        opt_df['regime'] = opt_df['corridor_param'].apply(
            lambda x: 'corridor-dominated' if x >= CORRIDOR_THRESHOLD else 'junction-dominated'
        )

        # Split data by regime
        corridor_dom = opt_df[opt_df['regime'] == 'corridor-dominated']
        junction_dom = opt_df[opt_df['regime'] == 'junction-dominated']

        # Create 2x2 panel of regression plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Create colormap for corridor parameter
        corridor_params = opt_df['corridor_param'].values
        norm = plt.Normalize(vmin=corridor_params.min(), vmax=corridor_params.max())

        # ===== TOP ROW: Full dataset with regime highlighting =====

        # 1. Lambda* vs mean corridor length (all data + regime-specific regressions)
        ax = axes[0, 0]

        # Plot all points colored by regime
        scatter_junc = ax.scatter(junction_dom['mean_corridor_length'], junction_dom['optimal_lambda'],
                                 s=50, alpha=0.4, c='lightgray', edgecolors='red', linewidths=1.5,
                                 label='Junction-dominated (excluded)')
        scatter_corr = ax.scatter(corridor_dom['mean_corridor_length'], corridor_dom['optimal_lambda'],
                                 s=50, alpha=0.7, c=corridor_dom['corridor_param'], cmap='plasma',
                                 edgecolors='black', linewidths=0.5)

        # Add regression line for corridor-dominated regime only
        if len(corridor_dom) > 1:
            x = corridor_dom['mean_corridor_length']
            y = corridor_dom['optimal_lambda']
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'r--', linewidth=2, alpha=0.7,
                   label=f'Corridor-dom: R²={r_value**2:.3f}, p={p_value:.3f}')

        ax.set_xlabel('Mean Corridor Length')
        ax.set_ylabel('Optimal λ*')
        ax.set_title('Corridor Hypothesis: λ* vs Corridor Length')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

        # 2. Lambda* vs junction density
        ax = axes[0, 1]

        ax.scatter(junction_dom['junction_density'], junction_dom['optimal_lambda'],
                  s=50, alpha=0.4, c='lightgray', edgecolors='red', linewidths=1.5,
                  label='Junction-dominated (excluded)')
        ax.scatter(corridor_dom['junction_density'], corridor_dom['optimal_lambda'],
                  s=50, alpha=0.7, c=corridor_dom['corridor_param'], cmap='plasma',
                  edgecolors='black', linewidths=0.5)

        if len(corridor_dom) > 1:
            x = corridor_dom['junction_density']
            y = corridor_dom['optimal_lambda']
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'r--', linewidth=2, alpha=0.7,
                   label=f'Corridor-dom: R²={r_value**2:.3f}, p={p_value:.3f}')

        ax.set_xlabel('Junction Density')
        ax.set_ylabel('Optimal λ*')
        ax.set_title('Junction Hypothesis: λ* vs Junction Density')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

        # ===== BOTTOM ROW: Additional analyses =====

        # 3. Lambda* vs corridor fraction
        ax = axes[1, 0]

        ax.scatter(junction_dom['frac_corridors'], junction_dom['optimal_lambda'],
                  s=50, alpha=0.4, c='lightgray', edgecolors='red', linewidths=1.5,
                  label='Junction-dominated (excluded)')
        scatter = ax.scatter(corridor_dom['frac_corridors'], corridor_dom['optimal_lambda'],
                           s=50, alpha=0.7, c=corridor_dom['corridor_param'], cmap='plasma',
                           edgecolors='black', linewidths=0.5)

        if len(corridor_dom) > 1:
            x = corridor_dom['frac_corridors']
            y = corridor_dom['optimal_lambda']
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'r--', linewidth=2, alpha=0.7,
                   label=f'Corridor-dom: R²={r_value**2:.3f}, p={p_value:.3f}')

        ax.set_xlabel('Fraction of Corridor Nodes')
        ax.set_ylabel('Optimal λ*')
        ax.set_title('λ* vs Corridor Fraction')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

        # 4. Regime validity plot: Composite score vs corridor parameter
        ax = axes[1, 1]

        # Show optimal lambda for each corridor parameter
        regime_summary = opt_df.groupby('corridor_param').agg({
            'optimal_lambda': 'mean',
            'best_composite_score': 'mean'
        }).reset_index()

        ax2 = ax.twinx()

        # Plot optimal lambda
        line1 = ax.plot(regime_summary['corridor_param'], regime_summary['optimal_lambda'],
                       'o-', linewidth=2, markersize=8, color='steelblue', label='Optimal λ*')
        # Plot composite score
        line2 = ax2.plot(regime_summary['corridor_param'], regime_summary['best_composite_score'],
                        's--', linewidth=2, markersize=8, color='orange', alpha=0.7, label='Composite Score')

        # Mark regime threshold
        ax.axvline(x=CORRIDOR_THRESHOLD, color='red', linestyle=':', linewidth=2, alpha=0.5,
                  label=f'Regime threshold ({CORRIDOR_THRESHOLD})')

        ax.set_xlabel('Corridor Parameter')
        ax.set_ylabel('Optimal λ*', color='steelblue')
        ax2.set_ylabel('Best Composite Score', color='orange')
        ax.set_title('Regime Validity: λ* vs Topology Parameter')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax.grid(alpha=0.3)

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

        # Add colorbar
        plt.tight_layout()
        cbar = fig.colorbar(scatter, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Corridor Parameter', rotation=270, labelpad=20)

        self._save_fig(fig, 'optimal_lambda_analysis', subdir)
        plt.close()

        # Additional plot: Performance landscape binned by junction density (using composite score)
        graph_bins = self._compute_graph_bins()
        num_bins = graph_bins['bin_id'].nunique()

        fig, axes = plt.subplots(1, num_bins, figsize=(5*num_bins, 5))
        if num_bins == 1:
            axes = [axes]

        for bin_id in range(num_bins):
            ax = axes[bin_id]
            bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
            bin_label = bin_graphs.iloc[0]['bin_label']

            # Collect composite scores for all graphs in this bin
            bin_composite_scores = []
            for _, graph_row in bin_graphs.iterrows():
                topology = graph_row['topology']
                seed = graph_row['seed']

                # Find this graph's composite scores in optimal_lambdas
                graph_opt = [ol for ol in optimal_lambdas if ol['topology'] == topology and ol['seed'] == seed]
                if len(graph_opt) == 0:
                    continue

                # Recompute composite scores for all lambdas for this graph (already done above, use same logic)
                graph_data = self.df[
                    (self.df['topology'] == topology) &
                    (self.df['seed'] == seed)
                ]
                graph_auc = self.auc_results[
                    (self.auc_results['topology'] == topology) &
                    (self.auc_results['seed'] == seed)
                ]

                for lambda_val in sorted(graph_data['lambda'].unique()):
                    lambda_data = graph_data[graph_data['lambda'] == lambda_val]
                    num_episodes = len(lambda_data)
                    final_episodes = lambda_data.iloc[int(0.8 * num_episodes):]

                    lambda_auc = graph_auc[graph_auc['lambda'] == lambda_val]
                    if len(lambda_auc) == 0:
                        continue
                    auc = lambda_auc['auc'].values[0]

                    auc_range = graph_auc['auc'].max() - graph_auc['auc'].min()
                    auc_norm = (auc - graph_auc['auc'].min()) / auc_range if auc_range > 0 else 1.0

                    junction_acc = final_episodes['junction_accuracy'].mean()
                    consultation_rate = final_episodes['consultation_rate'].mean()
                    wrong_turn_rate = final_episodes['wrong_turn_rate'].mean()

                    success_final = final_episodes[final_episodes['success'] == True]
                    if len(success_final) > 0:
                        jc_ratio = success_final['junction_corridor_ratio'].mean()
                        jc_ratio_norm = np.clip(jc_ratio, 0, 2) / 2.0
                    else:
                        jc_ratio_norm = 0.0

                    junction_acc = junction_acc if not np.isnan(junction_acc) else 0.0
                    consultation_rate = consultation_rate if not np.isnan(consultation_rate) else 1.0
                    wrong_turn_rate = wrong_turn_rate if not np.isnan(wrong_turn_rate) else 1.0

                    composite_score = (
                        0.30 * auc_norm +
                        0.25 * junction_acc +
                        0.20 * (1 - consultation_rate) +
                        0.15 * (1 - wrong_turn_rate) +
                        0.10 * jc_ratio_norm
                    )

                    bin_composite_scores.append({
                        'lambda': lambda_val,
                        'composite_score': composite_score
                    })

            if len(bin_composite_scores) == 0:
                continue

            bin_composite_df = pd.DataFrame(bin_composite_scores)

            # Get mean and std across all graphs in bin
            composite_stats = bin_composite_df.groupby('lambda')['composite_score'].agg(['mean', 'std', 'count'])
            lambdas = composite_stats.index
            means = composite_stats['mean']
            stds = composite_stats['std']
            ns = composite_stats['count']

            ci = 1.96 * stds / np.sqrt(ns)

            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8, color='steelblue')
            ax.fill_between(lambdas, means - ci, means + ci, alpha=0.3, color='steelblue')

            # Mark optimal
            best_idx = means.idxmax()
            ax.axvline(x=best_idx, color='red', linestyle='--', linewidth=1.5,
                      alpha=0.7, label=f'Optimal λ*={best_idx:.1f}')

            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Composite Score')
            ax.set_title(bin_label)
            ax.grid(alpha=0.3)
            ax.legend()
            ax.set_xlim(-0.05, 1.05)

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

        # 1. Effective credit distance vs lambda (binned by junction density)
        graph_bins = self._compute_graph_bins()
        num_bins = graph_bins['bin_id'].nunique()

        # Create single plot with one curve per bin
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Create colormap for bins
        bin_colors = plt.cm.Reds(np.linspace(0.2, 1, num_bins))

        for bin_id in range(num_bins):
            bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
            bin_label = bin_graphs.iloc[0]['bin_label']

            # Collect all data for graphs in this bin
            bin_data_list = []
            for _, graph_row in bin_graphs.iterrows():
                topology = graph_row['topology']
                seed = graph_row['seed']

                graph_data = success_df[
                    (success_df['topology'] == topology) &
                    (success_df['seed'] == seed)
                ]

                if len(graph_data) > 0:
                    bin_data_list.append(graph_data)

            if len(bin_data_list) == 0:
                continue

            # Concatenate all data for this bin and compute mean across graphs
            bin_data = pd.concat(bin_data_list)
            bin_stats = bin_data.groupby('lambda')['effective_credit_distance'].agg(['mean', 'std', 'count'])

            lambdas = bin_stats.index
            means = bin_stats['mean']
            stds = bin_stats['std']
            ns = bin_stats['count']
            ci = 1.96 * stds / np.sqrt(ns)

            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8,
                   label=bin_label, color=bin_colors[bin_id], alpha=0.8)
            ax.fill_between(lambdas, means - ci, means + ci,
                           color=bin_colors[bin_id], alpha=0.2)

        ax.set_xlabel('Lambda (λ)')
        ax.set_ylabel('Effective Credit Distance')
        ax.set_title('Effective Credit Distance by Junction Density')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc='best')

        plt.tight_layout()
        self._save_fig(fig, 'effective_credit_distance', subdir)
        plt.close()

        # 2. Junction/corridor credit ratio vs lambda (binned by junction density)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for bin_id in range(num_bins):
            bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
            bin_label = bin_graphs.iloc[0]['bin_label']

            # Collect all data for graphs in this bin
            bin_data_list = []
            for _, graph_row in bin_graphs.iterrows():
                topology = graph_row['topology']
                seed = graph_row['seed']

                graph_data = success_df[
                    (success_df['topology'] == topology) &
                    (success_df['seed'] == seed)
                ]

                if len(graph_data) > 0:
                    bin_data_list.append(graph_data)

            if len(bin_data_list) == 0:
                continue

            # Concatenate all data for this bin and compute mean across graphs
            bin_data = pd.concat(bin_data_list)
            bin_stats = bin_data.groupby('lambda')['junction_corridor_ratio'].agg(['mean', 'std', 'count'])

            if len(bin_stats) == 0:
                continue

            lambdas = bin_stats.index
            means = bin_stats['mean']
            stds = bin_stats['std']
            ns = bin_stats['count']
            ci = 1.96 * stds / np.sqrt(ns)

            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8,
                   label=bin_label, color=bin_colors[bin_id], alpha=0.8)
            ax.fill_between(lambdas, means - ci, means + ci,
                           color=bin_colors[bin_id], alpha=0.2)

        ax.set_xlabel('Lambda (λ)')
        ax.set_ylabel('Junction/Corridor Credit Ratio')
        ax.set_title('Junction/Corridor Credit Ratio by Junction Density')
        ax.grid(alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend(fontsize=10, loc='best')

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

        fig, axes = plt.subplots(3, 4, figsize=(15, 8))
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

        # 4. Decision localization score vs lambda (binned by junction density)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for bin_id in range(num_bins):
            bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
            bin_label = bin_graphs.iloc[0]['bin_label']

            # Collect all data for graphs in this bin
            bin_data_list = []
            for _, graph_row in bin_graphs.iterrows():
                topology = graph_row['topology']
                seed = graph_row['seed']

                graph_data = success_df[
                    (success_df['topology'] == topology) &
                    (success_df['seed'] == seed)
                ]

                if len(graph_data) > 0:
                    bin_data_list.append(graph_data)

            if len(bin_data_list) == 0:
                continue

            # Concatenate all data for this bin and compute mean across graphs
            bin_data = pd.concat(bin_data_list)
            bin_stats = bin_data.groupby('lambda')['mean_decision_localization'].agg(['mean', 'std', 'count'])

            if len(bin_stats) == 0:
                continue

            lambdas = bin_stats.index
            means = bin_stats['mean']
            stds = bin_stats['std']
            ns = bin_stats['count']
            ci = 1.96 * stds / np.sqrt(ns)

            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8,
                   label=bin_label, color=bin_colors[bin_id], alpha=0.8)
            ax.fill_between(lambdas, means - ci, means + ci,
                           color=bin_colors[bin_id], alpha=0.2)

        ax.set_xlabel('Lambda (λ)')
        ax.set_ylabel('Decision Localization Score')
        ax.set_title('Decision Localization Score by Junction Density')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc='best')

        plt.tight_layout()
        self._save_fig(fig, 'decision_localization', subdir)
        plt.close()

    # =========================================================================
    # FIGURE 5: Decision Quality Metrics
    # =========================================================================

    def plot_decision_quality(self):
        """Plot junction decision quality metrics."""
        subdir = 'decision_quality'

        # Compute graph bins
        graph_bins = self._compute_graph_bins()
        num_bins = graph_bins['bin_id'].nunique()

        # Create colormap for bins
        bin_colors = plt.cm.Oranges(np.linspace(0.2, 1, num_bins))

        # 1. Junction accuracy vs lambda (binned by junction density)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for bin_id in range(num_bins):
            bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
            bin_label = bin_graphs.iloc[0]['bin_label']

            # Collect all data for graphs in this bin
            bin_data_list = []
            for _, graph_row in bin_graphs.iterrows():
                topology = graph_row['topology']
                seed = graph_row['seed']

                graph_data = self.df[
                    (self.df['topology'] == topology) &
                    (self.df['seed'] == seed)
                ]

                if len(graph_data) > 0:
                    bin_data_list.append(graph_data)

            if len(bin_data_list) == 0:
                continue

            # Concatenate all data for this bin and compute mean across graphs
            bin_data = pd.concat(bin_data_list)
            bin_stats = bin_data.groupby('lambda')['junction_accuracy'].agg(['mean', 'std', 'count'])

            lambdas = bin_stats.index
            means = bin_stats['mean']
            stds = bin_stats['std']
            ns = bin_stats['count']
            ci = 1.96 * stds / np.sqrt(ns)

            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8,
                   label=bin_label, color=bin_colors[bin_id], alpha=0.8)
            ax.fill_between(lambdas, means - ci, means + ci,
                           color=bin_colors[bin_id], alpha=0.2)

        ax.set_xlabel('Lambda (λ)')
        ax.set_ylabel('Junction Accuracy')
        ax.set_title('Junction Accuracy by Junction Density')
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=10, loc='best')

        plt.tight_layout()
        self._save_fig(fig, 'junction_accuracy', subdir)
        plt.close()

        # 2. Wrong turn rate vs lambda (binned by junction density)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for bin_id in range(num_bins):
            bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
            bin_label = bin_graphs.iloc[0]['bin_label']

            # Collect all data for graphs in this bin
            bin_data_list = []
            for _, graph_row in bin_graphs.iterrows():
                topology = graph_row['topology']
                seed = graph_row['seed']

                graph_data = self.df[
                    (self.df['topology'] == topology) &
                    (self.df['seed'] == seed)
                ]

                if len(graph_data) > 0:
                    bin_data_list.append(graph_data)

            if len(bin_data_list) == 0:
                continue

            # Concatenate all data for this bin and compute mean across graphs
            bin_data = pd.concat(bin_data_list)
            bin_stats = bin_data.groupby('lambda')['wrong_turn_rate'].agg(['mean', 'std', 'count'])

            lambdas = bin_stats.index
            means = bin_stats['mean']
            stds = bin_stats['std']
            ns = bin_stats['count']
            ci = 1.96 * stds / np.sqrt(ns)

            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8,
                   label=bin_label, color=bin_colors[bin_id], alpha=0.8)
            ax.fill_between(lambdas, means - ci, means + ci,
                           color=bin_colors[bin_id], alpha=0.2)

        ax.set_xlabel('Lambda (λ)')
        ax.set_ylabel('Wrong Turn Rate')
        ax.set_title('Wrong Turn Rate by Junction Density')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc='best')

        plt.tight_layout()
        self._save_fig(fig, 'wrong_turn_rate', subdir)
        plt.close()

        # 3. Junction vs corridor entropy comparison (binned by junction density)
        fig, axes = plt.subplots(1, num_bins, figsize=(5*num_bins, 5))
        if num_bins == 1:
            axes = [axes]

        for bin_id in range(num_bins):
            ax = axes[bin_id]
            bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
            bin_label = bin_graphs.iloc[0]['bin_label']

            # Collect all data for graphs in this bin
            bin_data_list = []
            for _, graph_row in bin_graphs.iterrows():
                topology = graph_row['topology']
                seed = graph_row['seed']

                graph_data = self.df[
                    (self.df['topology'] == topology) &
                    (self.df['seed'] == seed)
                ]

                if len(graph_data) > 0:
                    bin_data_list.append(graph_data)

            if len(bin_data_list) == 0:
                continue

            bin_data = pd.concat(bin_data_list)

            junc_ent = bin_data.groupby('lambda')['mean_junction_entropy'].mean()
            corr_ent = bin_data.groupby('lambda')['mean_corridor_entropy'].mean()

            lambdas = junc_ent.index

            ax.plot(lambdas, junc_ent, 'o-', linewidth=2, markersize=8,
                   color='orange', label='Junction')
            ax.plot(lambdas, corr_ent, 's-', linewidth=2, markersize=8,
                   color='skyblue', label='Corridor')

            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Policy Entropy')
            ax.set_title(bin_label)
            ax.grid(alpha=0.3)
            ax.legend()

        plt.tight_layout()
        self._save_fig(fig, 'junction_vs_corridor_entropy', subdir)
        plt.close()

        # 4. Policy margin at junctions (binned by junction density)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for bin_id in range(num_bins):
            bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
            bin_label = bin_graphs.iloc[0]['bin_label']

            # Collect all data for graphs in this bin
            bin_data_list = []
            for _, graph_row in bin_graphs.iterrows():
                topology = graph_row['topology']
                seed = graph_row['seed']

                graph_data = self.df[
                    (self.df['topology'] == topology) &
                    (self.df['seed'] == seed)
                ]

                if len(graph_data) > 0:
                    bin_data_list.append(graph_data)

            if len(bin_data_list) == 0:
                continue

            # Concatenate all data for this bin and compute mean across graphs
            bin_data = pd.concat(bin_data_list)
            bin_stats = bin_data.groupby('lambda')['mean_junction_margin'].agg(['mean', 'std', 'count'])

            if len(bin_stats) == 0:
                continue

            lambdas = bin_stats.index
            means = bin_stats['mean']
            stds = bin_stats['std']
            ns = bin_stats['count']
            ci = 1.96 * stds / np.sqrt(ns)

            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8,
                   label=bin_label, color=bin_colors[bin_id], alpha=0.8)
            ax.fill_between(lambdas, means - ci, means + ci,
                           color=bin_colors[bin_id], alpha=0.2)

        ax.set_xlabel('Lambda (λ)')
        ax.set_ylabel('Mean Junction Policy Margin')
        ax.set_title('Junction Policy Margin by Junction Density')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc='best')

        plt.tight_layout()
        self._save_fig(fig, 'junction_policy_margin', subdir)
        plt.close()

    # =========================================================================
    # FIGURE 6: Topology Descriptors
    # =========================================================================

    def plot_topology_descriptors(self):
        """Plot topology characterization."""
        subdir = 'topology'

        # Get one sample per topology and sort by junction density
        topo_samples = self.df.groupby('topology').first().reset_index()
        topo_samples = topo_samples.sort_values('topo_junction_density')

        topologies = topo_samples['topology'].tolist()

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

        # 2. Key topology metrics panel (with spatial homogeneity if available)
        has_homogeneity = 'topo_spatial_homogeneity' in topo_samples.columns
        num_panels = 4 if has_homogeneity else 3

        fig, axes = plt.subplots(1, num_panels, figsize=(5*num_panels, 5))

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

        # Spatial homogeneity (if available)
        if has_homogeneity:
            ax = axes[3]
            spatial_homog = topo_samples.set_index('topology')['topo_spatial_homogeneity']
            bars = ax.barh(range(len(topologies)), spatial_homog, color=TOPOLOGY_COLORS[:len(topologies)])
            ax.set_yticks(range(len(topologies)))
            ax.set_yticklabels([])
            ax.set_xlabel('Spatial Homogeneity')
            ax.set_title('Spatial Homogeneity by Topology')
            ax.grid(alpha=0.3, axis='x')
            ax.set_xlim(0, 1)

        plt.tight_layout()
        self._save_fig(fig, 'topology_metrics_panel', subdir)
        plt.close()

        # 3. Correlation matrix of topology descriptors
        # Build feature list, checking for availability of new metrics
        topo_feature_cols = [
            'topo_mean_corridor_length',
            'topo_junction_density',
            'topo_frac_corridors',
            'topo_frac_junctions'
        ]

        # Add new metrics if available (backwards compatibility)
        if 'topo_mean_global_corr_dec_ratio' in topo_samples.columns:
            topo_feature_cols.append('topo_mean_global_corr_dec_ratio')
        if 'topo_spatial_homogeneity' in topo_samples.columns:
            topo_feature_cols.append('topo_spatial_homogeneity')

        topo_features = topo_samples[topo_feature_cols]

        # Rename columns for better display
        rename_map = {
            'topo_mean_corridor_length': 'Mean Corridor Length',
            'topo_junction_density': 'Junction Density',
            'topo_frac_corridors': 'Corridor Fraction',
            'topo_frac_junctions': 'Junction Fraction',
            'topo_mean_global_corr_dec_ratio': 'Global Corr/Dec Ratio',
            'topo_spatial_homogeneity': 'Spatial Homogeneity'
        }

        # Only rename columns that exist
        rename_map_filtered = {k: v for k, v in rename_map.items() if k in topo_features.columns}
        topo_features_renamed = topo_features.rename(columns=rename_map_filtered)

        corr_matrix = topo_features_renamed.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, vmin=-1, vmax=1, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Matrix of Topology Descriptors')

        plt.tight_layout()
        self._save_fig(fig, 'topology_correlation_matrix', subdir)
        plt.close()

        # 4. Spatial homogeneity by junction density bins (if available)
        if 'topo_spatial_homogeneity' in self.df.columns:
            # Use the same binning logic as _compute_graph_bins()
            graph_bins = self._compute_graph_bins()

            # Merge binning info with full dataframe
            homog_data = self.df.groupby(['topology', 'seed']).first().reset_index()
            homog_data = homog_data.merge(graph_bins[['topology', 'seed', 'bin_id', 'bin_label']],
                                          on=['topology', 'seed'],
                                          how='left')

            # Extract corridor parameter for coloring
            homog_data['corridor_param'] = homog_data['topology'].str.extract(r'([\d.]+)').astype(float)

            # Compute statistics per bin
            bin_stats = homog_data.groupby('bin_label').agg({
                'topo_spatial_homogeneity': ['mean', 'std', 'count'],
                'topo_junction_density': 'mean',
                'corridor_param': 'mean'
            }).reset_index()

            # Flatten column names
            bin_stats.columns = ['bin_label', 'homog_mean', 'homog_std', 'count', 'junc_density', 'corridor_mean']

            # Sort by junction density to ensure correct ordering
            bin_stats = bin_stats.sort_values('junc_density')

            # Create figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Left: Bar plot of homogeneity by junction density bin
            ax = axes[0]
            x_pos = np.arange(len(bin_stats))

            bars = ax.bar(x_pos, bin_stats['homog_mean'], yerr=bin_stats['homog_std'],
                         capsize=5, alpha=0.7, color='steelblue', edgecolor='black')

            ax.set_xticks(x_pos)
            ax.set_xticklabels(bin_stats['bin_label'], rotation=45, ha='right')
            ax.set_xlabel('Junction Density Range')
            ax.set_ylabel('Spatial Homogeneity')
            ax.set_title('Spatial Homogeneity by Junction Density')
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3, axis='y')

            # Add count labels on bars
            for i, (bar, count) in enumerate(zip(bars, bin_stats['count'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'n={int(count)}', ha='center', va='bottom', fontsize=9)

            # Right: Scatter plot of homogeneity vs junction density
            ax = axes[1]

            # Color by corridor parameter
            scatter = ax.scatter(homog_data['topo_junction_density'],
                               homog_data['topo_spatial_homogeneity'],
                               c=homog_data['corridor_param'],
                               cmap='plasma',
                               s=50,
                               alpha=0.6,
                               edgecolors='black',
                               linewidths=0.5)

            ax.set_xlabel('Junction Density')
            ax.set_ylabel('Spatial Homogeneity')
            ax.set_title('Spatial Homogeneity vs Junction Density')
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Corridor Parameter', rotation=270, labelpad=20)

            plt.tight_layout()
            self._save_fig(fig, 'spatial_homogeneity_analysis', subdir)
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

        # 2. Value estimates over training (binned by junction density)
        graph_bins = self._compute_graph_bins()
        num_bins = graph_bins['bin_id'].nunique()

        fig, axes = plt.subplots(1, num_bins, figsize=(5*num_bins, 5))
        if num_bins == 1:
            axes = [axes]

        for bin_id in range(num_bins):
            ax = axes[bin_id]
            bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
            bin_label = bin_graphs.iloc[0]['bin_label']

            # Collect all data for graphs in this bin
            bin_data_list = []
            for _, graph_row in bin_graphs.iterrows():
                topology = graph_row['topology']
                seed = graph_row['seed']

                graph_data = self.df[
                    (self.df['topology'] == topology) &
                    (self.df['seed'] == seed)
                ]

                if len(graph_data) > 0:
                    bin_data_list.append(graph_data)

            if len(bin_data_list) == 0:
                continue

            bin_data = pd.concat(bin_data_list)

            for lambda_idx, lambda_val in enumerate(lambda_values):
                lambda_data = bin_data[bin_data['lambda'] == lambda_val]

                if len(lambda_data) == 0:
                    continue

                grouped = lambda_data.groupby('episode')['mean_value'].mean()
                episodes = grouped.index
                values = grouped.values

                color = LAMBDA_COLORS[lambda_idx % len(LAMBDA_COLORS)]
                ax.plot(episodes, values, label=f'λ={lambda_val:.1f}',
                       color=color, linewidth=1.5, alpha=0.7)

            ax.set_xlabel('Episode')
            ax.set_ylabel('Mean Value Estimate')
            ax.set_title(bin_label)
            ax.grid(alpha=0.3)
            ax.legend(ncol=2, fontsize=8)

        plt.tight_layout()
        self._save_fig(fig, 'value_estimates_over_training', subdir)
        plt.close()

        # 3. Per-seed variance in final performance (binned by junction density)
        if self.summary is not None:
            fig, axes = plt.subplots(1, num_bins, figsize=(5*num_bins, 5))
            if num_bins == 1:
                axes = [axes]

            for bin_id in range(num_bins):
                ax = axes[bin_id]
                bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
                bin_label = bin_graphs.iloc[0]['bin_label']

                # Collect summary data for graphs in this bin
                bin_summary_list = []
                for _, graph_row in bin_graphs.iterrows():
                    topology = graph_row['topology']
                    seed = graph_row['seed']

                    graph_summary = self.summary[
                        (self.summary['topology'] == topology) &
                        (self.summary['seed'] == seed)
                    ]

                    if len(graph_summary) > 0:
                        bin_summary_list.append(graph_summary)

                if len(bin_summary_list) == 0:
                    continue

                bin_summary = pd.concat(bin_summary_list)

                for lambda_val in lambda_values:
                    lambda_summary = bin_summary[bin_summary['lambda'] == lambda_val]

                    if len(lambda_summary) == 0:
                        continue

                    # Plot individual seeds
                    seeds = lambda_summary['seed']
                    successes = lambda_summary['success']

                    # Jitter x-position slightly for visibility
                    x_pos = lambda_val + np.random.normal(0, 0.01, len(seeds))
                    ax.scatter(x_pos, successes, alpha=0.6, s=50, color='steelblue')

                # Add mean line
                mean_success = bin_summary.groupby('lambda')['success'].mean()
                ax.plot(mean_success.index, mean_success.values, 'r-',
                       linewidth=2, alpha=0.8, label='Mean')

                ax.set_xlabel('Lambda (λ)')
                ax.set_ylabel('Final Success Rate')
                ax.set_title(bin_label)
                ax.grid(alpha=0.3)
                ax.legend()
                ax.set_ylim(-0.05, 1.05)

            plt.tight_layout()
            self._save_fig(fig, 'per_seed_variance', subdir)
            plt.close()

        # 4. Upstream-local credit ratio (credit smearing) - single plot with bin-averaged curves
        success_df = self.df[self.df['success'] == True]

        if len(success_df) > 0 and 'mean_upstream_local_ratio' in success_df.columns:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            # Create colormap for bins
            bin_colors = plt.cm.viridis(np.linspace(0, 1, num_bins))

            for bin_id in range(num_bins):
                bin_graphs = graph_bins[graph_bins['bin_id'] == bin_id]
                bin_label = bin_graphs.iloc[0]['bin_label']

                # Collect all data for graphs in this bin
                bin_data_list = []
                for _, graph_row in bin_graphs.iterrows():
                    topology = graph_row['topology']
                    seed = graph_row['seed']

                    graph_data = success_df[
                        (success_df['topology'] == topology) &
                        (success_df['seed'] == seed)
                    ]

                    if len(graph_data) > 0:
                        bin_data_list.append(graph_data)

                if len(bin_data_list) == 0:
                    continue

                # Concatenate all data for this bin and compute mean across graphs
                bin_data = pd.concat(bin_data_list)
                ratio_stats = bin_data.groupby('lambda')['mean_upstream_local_ratio'].agg(
                    ['mean', 'std', 'count']
                )

                if len(ratio_stats) == 0:
                    continue

                lambdas = ratio_stats.index
                means = ratio_stats['mean']
                stds = ratio_stats['std']
                ns = ratio_stats['count']
                ci = 1.96 * stds / np.sqrt(ns)

                ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8,
                       label=bin_label, color=bin_colors[bin_id], alpha=0.8)
                ax.fill_between(lambdas, means - ci, means + ci,
                               color=bin_colors[bin_id], alpha=0.2)

            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Upstream/Local Credit Ratio')
            ax.set_title('Upstream/Local Credit Ratio by Junction Density')
            ax.grid(alpha=0.3)
            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.legend(fontsize=10, loc='best')

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
    plotter = LambdaExperimentPlotter(args.results_dir, args.output_dir, num_bins=6)
    plotter.plot_all(figures=figures)

    print("\n✓ Plotting complete!")
    print(f"  View figures in: {plotter.output_dir}/")


if __name__ == '__main__':
    main()
