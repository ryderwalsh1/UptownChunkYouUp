"""
Plotting Suite for Elementary Topology Experiment

Generates publication-quality figures analyzing how TD(λ) performance varies
across elementary maze topologies with controlled structural properties.

Supports:
- Corridor topologies (varying corridor length)
- Branch topologies (varying pre/post branch lengths, with alternating/switch training schemes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
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
LAMBDA_COLORS = plt.cm.plasma(np.linspace(0.1, 0.9, 11))  # For lambda values 0.0-1.0
LENGTH_COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, 6))  # For corridor lengths


class ElementaryTopologyPlotter:
    """Main plotting class for elementary topology analysis."""

    def __init__(self, results_dir, output_dir='elementary_topology_plots',
                 learning_curve_window=200):
        """
        Initialize plotter with results directory.

        Parameters:
        -----------
        results_dir : str or Path
            Base directory containing topology-specific subdirectories
            (e.g., elementary_topology_results/)
        output_dir : str or Path
            Directory to save generated figures
        learning_curve_window : int
            Window size for running-average smoothing in learning curves
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.learning_curve_window = learning_curve_window

        # Will be populated by load methods
        self.corridor_data = None
        self.corridor_lengths = []
        self.branch_data = None
        self.branch_pre_lengths = []
        self.branch_post_lengths = []

    def _running_average(self, series, window=None):
        """Compute running average with a centered rolling window."""
        if window is None:
            window = self.learning_curve_window
        return series.rolling(window=window, min_periods=1, center=True).mean()

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

    def load_corridor_data(self):
        """
        Load data from all corridor topology experiments.

        Looks for directories matching 'corridor_L*' pattern and loads
        their results.csv files.
        """
        print("Loading corridor topology data...")

        corridor_dirs = sorted(self.results_dir.glob('corridor_L*'))
        if len(corridor_dirs) == 0:
            print(f"  Warning: No corridor topology directories found in {self.results_dir}")
            return

        all_data = []
        for corridor_dir in corridor_dirs:
            topology_name = corridor_dir.name
            results_file = corridor_dir / 'results.csv'

            if not results_file.exists():
                print(f"  Warning: No results.csv in {corridor_dir}")
                continue

            # Load data
            df = pd.read_csv(results_file)

            # Extract corridor length from topology name (e.g., "corridor_L10" -> 10)
            length = int(topology_name.split('_L')[1])
            df['corridor_length'] = length

            all_data.append(df)
            print(f"  Loaded {topology_name}: {len(df)} episodes, length={length}")

        if len(all_data) == 0:
            print("  Error: No data loaded!")
            return

        # Combine all corridor data
        self.corridor_data = pd.concat(all_data, ignore_index=True)
        self.corridor_lengths = sorted(self.corridor_data['corridor_length'].unique())

        print(f"  Total episodes: {len(self.corridor_data)}")
        print(f"  Corridor lengths: {self.corridor_lengths}")
        print(f"  Lambda values: {sorted(self.corridor_data['lambda'].unique())}")
        print(f"  Seeds: {sorted(self.corridor_data['seed'].unique())}")

    def load_branch_data(self, training_scheme):
        """
        Load data from all branch topology experiments for a specific training scheme.

        Parameters:
        -----------
        training_scheme : str
            Training scheme subdirectory name ('branch_alternating' or 'branch_switch')

        Looks for directories matching 'branch_pre*_post*/{training_scheme}' pattern
        and loads their results.csv files.
        """
        print(f"Loading branch topology data for {training_scheme}...")

        branch_dirs = sorted(self.results_dir.glob('branch_pre*_post*'))
        if len(branch_dirs) == 0:
            print(f"  Warning: No branch topology directories found in {self.results_dir}")
            return

        all_data = []
        for branch_dir in branch_dirs:
            topology_name = branch_dir.name
            results_file = branch_dir / training_scheme / 'results.csv'

            if not results_file.exists():
                print(f"  Warning: No results.csv in {branch_dir / training_scheme}")
                continue

            # Load data
            df = pd.read_csv(results_file)

            # Extract pre and post branch lengths from topology name
            # Format: "branch_pre{X}_post{Y}"
            parts = topology_name.split('_')
            pre_length = int(parts[1].replace('pre', ''))
            post_length = int(parts[2].replace('post', ''))

            df['pre_branch_length'] = pre_length
            df['post_branch_length'] = post_length

            all_data.append(df)
            print(f"  Loaded {topology_name}: {len(df)} episodes, pre={pre_length}, post={post_length}")

        if len(all_data) == 0:
            print("  Error: No data loaded!")
            return

        # Combine all branch data
        self.branch_data = pd.concat(all_data, ignore_index=True)
        self.branch_pre_lengths = sorted(self.branch_data['pre_branch_length'].unique())
        self.branch_post_lengths = sorted(self.branch_data['post_branch_length'].unique())

        print(f"  Total episodes: {len(self.branch_data)}")
        print(f"  Pre-branch lengths: {self.branch_pre_lengths}")
        print(f"  Post-branch lengths: {self.branch_post_lengths}")
        print(f"  Lambda values: {sorted(self.branch_data['lambda'].unique())}")
        print(f"  Seeds: {sorted(self.branch_data['seed'].unique())}")

    def plot_corridor_learning_curves(self):
        """
        Plot learning curves for corridor topologies.

        Creates 3 rows of subplots (one per corridor length):
        - Top row: Optimality ratio over time
        - Middle row: Reward over time
        - Bottom row: Success rate over time

        Each curve represents a different lambda value.
        """
        if self.corridor_data is None:
            print("Error: No corridor data loaded. Run load_corridor_data() first.")
            return

        subdir = 'corridor'
        lambda_values = sorted(self.corridor_data['lambda'].unique())
        window = self.learning_curve_window

        # 1. Optimality ratio learning curves
        num_lengths = len(self.corridor_lengths)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for length_idx, length in enumerate(self.corridor_lengths):
            if length_idx >= len(axes):
                break
            ax = axes[length_idx]
            length_data = self.corridor_data[self.corridor_data['corridor_length'] == length]

            # Filter to successful episodes only
            success_data = length_data[length_data['success'] == True]

            if len(success_data) == 0:
                ax.text(0.5, 0.5, 'No successful episodes',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Corridor Length {length}')
                continue

            for lambda_idx, lambda_val in enumerate(lambda_values):
                lambda_data = success_data[success_data['lambda'] == lambda_val]

                if len(lambda_data) == 0:
                    continue

                # Group by episode and compute statistics
                grouped = lambda_data.groupby('episode')['optimality_ratio'].agg(['mean', 'std', 'count'])
                episodes = grouped.index.to_numpy()
                mean = grouped['mean']
                std = grouped['std'].fillna(0)
                n = grouped['count']
                ci = 1.96 * std / np.sqrt(n)

                # Smooth
                mean_smooth = self._running_average(mean, window)
                ci_smooth = self._running_average(ci, window)

                color = LAMBDA_COLORS[lambda_idx % len(LAMBDA_COLORS)]
                ax.plot(episodes, mean_smooth, label=f'λ={lambda_val:.1f}',
                        color=color, linewidth=1.5, alpha=0.8)
                ax.fill_between(episodes, mean_smooth - ci_smooth, mean_smooth + ci_smooth,
                                color=color, alpha=0.15)

            ax.set_xlabel('Episode')
            ax.set_ylabel(f'Optimality Ratio\n({window}-ep running avg)')
            ax.set_title(f'Corridor Length {length}')
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.legend(ncol=2, fontsize=8)

        plt.tight_layout()
        self._save_fig(fig, 'learning_curves_optimality', subdir)
        plt.close()

        # 2. Reward learning curves
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for length_idx, length in enumerate(self.corridor_lengths):
            if length_idx >= len(axes):
                break
            ax = axes[length_idx]
            length_data = self.corridor_data[self.corridor_data['corridor_length'] == length]

            for lambda_idx, lambda_val in enumerate(lambda_values):
                lambda_data = length_data[length_data['lambda'] == lambda_val]

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
            ax.set_title(f'Corridor Length {length}')
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.legend(ncol=2, fontsize=8)

        plt.tight_layout()
        self._save_fig(fig, 'learning_curves_reward', subdir)
        plt.close()

        # 3. Success rate learning curves
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for length_idx, length in enumerate(self.corridor_lengths):
            if length_idx >= len(axes):
                break
            ax = axes[length_idx]
            length_data = self.corridor_data[self.corridor_data['corridor_length'] == length]

            for lambda_idx, lambda_val in enumerate(lambda_values):
                lambda_data = length_data[length_data['lambda'] == lambda_val]

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
            ax.set_title(f'Corridor Length {length}')
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.legend(ncol=2, fontsize=8)
            ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        self._save_fig(fig, 'learning_curves_success', subdir)
        plt.close()

    def plot_effective_credit_distance(self):
        """
        Plot effective credit distance vs lambda for each corridor length.

        X-axis: Lambda (0.0 to 1.0)
        Y-axis: Effective credit distance
        Each curve: Different corridor length
        """
        if self.corridor_data is None:
            print("Error: No corridor data loaded. Run load_corridor_data() first.")
            return

        subdir = 'corridor'

        # Filter to successful episodes only (credit diagnostics computed on success)
        success_data = self.corridor_data[self.corridor_data['success'] == True].copy()

        if len(success_data) == 0:
            print("Warning: No successful episodes. Skipping effective credit distance plot.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for length_idx, length in enumerate(self.corridor_lengths):
            length_data = success_data[success_data['corridor_length'] == length]

            if len(length_data) == 0:
                continue

            # Compute statistics per lambda
            stats = length_data.groupby('lambda')['effective_credit_distance'].agg(['mean', 'std', 'count'])
            lambdas = stats.index
            means = stats['mean']
            stds = stats['std'].fillna(0)
            ns = stats['count']
            ci = 1.96 * stds / np.sqrt(ns)

            color = LENGTH_COLORS[length_idx % len(LENGTH_COLORS)]
            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8,
                   label=f'Length {length}', color=color, alpha=0.8)
            ax.fill_between(lambdas, means - ci, means + ci,
                           color=color, alpha=0.2)

        ax.set_xlabel('Lambda (λ)')
        ax.set_ylabel('Effective Credit Distance')
        ax.set_title('Effective Credit Distance by Corridor Length')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc='best')

        plt.tight_layout()
        self._save_fig(fig, 'effective_credit_distance', subdir)
        plt.close()

    def plot_auc_vs_lambda(self):
        """
        Plot area under reward curve vs lambda for each corridor length.

        X-axis: Lambda (0.0 to 1.0)
        Y-axis: AUC (area under reward learning curve)
        Each curve: Different corridor length
        """
        if self.corridor_data is None:
            print("Error: No corridor data loaded. Run load_corridor_data() first.")
            return

        subdir = 'corridor'

        # Compute AUC for each (corridor_length, lambda, seed) combination
        auc_results = []

        for length in self.corridor_lengths:
            length_data = self.corridor_data[self.corridor_data['corridor_length'] == length]

            for lambda_val in sorted(length_data['lambda'].unique()):
                for seed in sorted(length_data['seed'].unique()):
                    condition_data = length_data[
                        (length_data['lambda'] == lambda_val) &
                        (length_data['seed'] == seed)
                    ]

                    if len(condition_data) == 0:
                        continue

                    # Compute AUC using trapezoidal rule
                    reward_series = condition_data.sort_values('episode')['episode_reward']
                    episodes = condition_data.sort_values('episode')['episode'].values

                    # Simple AUC: mean reward (equivalent to trapz with unit spacing)
                    auc = reward_series.mean()

                    auc_results.append({
                        'corridor_length': length,
                        'lambda': lambda_val,
                        'seed': seed,
                        'auc': auc
                    })

        if len(auc_results) == 0:
            print("Warning: No AUC results computed. Skipping AUC plot.")
            return

        auc_df = pd.DataFrame(auc_results)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for length_idx, length in enumerate(self.corridor_lengths):
            length_auc = auc_df[auc_df['corridor_length'] == length]

            if len(length_auc) == 0:
                continue

            # Compute statistics per lambda
            stats = length_auc.groupby('lambda')['auc'].agg(['mean', 'std', 'count'])
            lambdas = stats.index
            means = stats['mean']
            stds = stats['std'].fillna(0)
            ns = stats['count']
            ci = 1.96 * stds / np.sqrt(ns)

            color = LENGTH_COLORS[length_idx % len(LENGTH_COLORS)]
            ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8,
                   label=f'Length {length}', color=color, alpha=0.8)
            ax.fill_between(lambdas, means - ci, means + ci,
                           color=color, alpha=0.2)

        ax.set_xlabel('Lambda (λ)')
        ax.set_ylabel('Area Under Reward Curve')
        ax.set_title('Learning Efficiency by Corridor Length')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc='best')

        plt.tight_layout()
        self._save_fig(fig, 'auc_vs_lambda', subdir)
        plt.close()

    def plot_value_estimates_per_node(self):
        """
        Plot final value estimates for each node position.

        X-axis: Node index (0=start, max=corridor_length-1)
        Y-axis: Final value estimate
        Each curve: Different corridor length (different number of points)

        Uses final 20% of episodes for stable value estimates.
        """
        if self.corridor_data is None:
            print("Error: No corridor data loaded. Run load_corridor_data() first.")
            return

        subdir = 'corridor'

        # This plot requires per-node value data which isn't in the current results.csv
        # We need to load network checkpoints or add node-level logging to the experiment

        print("Warning: Value estimates per node plot requires network checkpoint data.")
        print("This feature requires modification to run_elementary_topology_experiment.py")
        print("to save final network states or log per-node value estimates.")
        print("Skipping this plot for now.")

        # TODO: Implement this when network checkpoints are available
        # For now, skip this plot
        return

    def plot_branch_learning_curves(self, training_scheme):
        """
        Plot learning curves for branch topologies.

        Creates 3 figures (optimality, reward, success) with 3x3 grid:
        - Rows: pre_branch_length (2, 5, 8)
        - Cols: post_branch_length (2, 5, 8)

        Each subplot shows curves for different lambda values.

        Parameters:
        -----------
        training_scheme : str
            Training scheme name for subdirectory organization
        """
        if self.branch_data is None:
            print("Error: No branch data loaded. Run load_branch_data() first.")
            return

        subdir = f'branch/{training_scheme}'
        lambda_values = sorted(self.branch_data['lambda'].unique())
        window = self.learning_curve_window

        # 1. Optimality ratio learning curves
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        for pre_idx, pre_len in enumerate(self.branch_pre_lengths):
            for post_idx, post_len in enumerate(self.branch_post_lengths):
                ax = axes[pre_idx, post_idx]

                # Filter data for this specific branch configuration
                branch_config_data = self.branch_data[
                    (self.branch_data['pre_branch_length'] == pre_len) &
                    (self.branch_data['post_branch_length'] == post_len)
                ]

                # Filter to successful episodes only
                success_data = branch_config_data[branch_config_data['success'] == True]

                if len(success_data) == 0:
                    ax.text(0.5, 0.5, 'No successful episodes',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Pre={pre_len}, Post={post_len}')
                    continue

                for lambda_idx, lambda_val in enumerate(lambda_values):
                    lambda_data = success_data[success_data['lambda'] == lambda_val]

                    if len(lambda_data) == 0:
                        continue

                    # Group by episode and compute statistics
                    grouped = lambda_data.groupby('episode')['optimality_ratio'].agg(['mean', 'std', 'count'])
                    episodes = grouped.index.to_numpy()
                    mean = grouped['mean']
                    std = grouped['std'].fillna(0)
                    n = grouped['count']
                    ci = 1.96 * std / np.sqrt(n)

                    # Smooth
                    mean_smooth = self._running_average(mean, window)
                    ci_smooth = self._running_average(ci, window)

                    color = LAMBDA_COLORS[lambda_idx % len(LAMBDA_COLORS)]
                    ax.plot(episodes, mean_smooth, label=f'λ={lambda_val:.1f}',
                            color=color, linewidth=1.5, alpha=0.8)
                    ax.fill_between(episodes, mean_smooth - ci_smooth, mean_smooth + ci_smooth,
                                    color=color, alpha=0.15)

                ax.set_xlabel('Episode')
                ax.set_ylabel(f'Optimality Ratio\n({window}-ep avg)')
                ax.set_title(f'Pre={pre_len}, Post={post_len}')
                ax.grid(alpha=0.3, linewidth=0.5)
                ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                if pre_idx == 0 and post_idx == 2:  # Top right for legend
                    ax.legend(ncol=2, fontsize=8)

        plt.tight_layout()
        self._save_fig(fig, 'learning_curves_optimality', subdir)
        plt.close()

        # 2. Reward learning curves
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        for pre_idx, pre_len in enumerate(self.branch_pre_lengths):
            for post_idx, post_len in enumerate(self.branch_post_lengths):
                ax = axes[pre_idx, post_idx]

                branch_config_data = self.branch_data[
                    (self.branch_data['pre_branch_length'] == pre_len) &
                    (self.branch_data['post_branch_length'] == post_len)
                ]

                for lambda_idx, lambda_val in enumerate(lambda_values):
                    lambda_data = branch_config_data[branch_config_data['lambda'] == lambda_val]

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
                ax.set_ylabel(f'Episode Reward\n({window}-ep avg)')
                ax.set_title(f'Pre={pre_len}, Post={post_len}')
                ax.grid(alpha=0.3, linewidth=0.5)
                if pre_idx == 0 and post_idx == 2:
                    ax.legend(ncol=2, fontsize=8)

        plt.tight_layout()
        self._save_fig(fig, 'learning_curves_reward', subdir)
        plt.close()

        # 3. Success rate learning curves
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        for pre_idx, pre_len in enumerate(self.branch_pre_lengths):
            for post_idx, post_len in enumerate(self.branch_post_lengths):
                ax = axes[pre_idx, post_idx]

                branch_config_data = self.branch_data[
                    (self.branch_data['pre_branch_length'] == pre_len) &
                    (self.branch_data['post_branch_length'] == post_len)
                ]

                for lambda_idx, lambda_val in enumerate(lambda_values):
                    lambda_data = branch_config_data[branch_config_data['lambda'] == lambda_val]

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
                ax.set_ylabel(f'Success Rate\n({window}-ep avg)')
                ax.set_title(f'Pre={pre_len}, Post={post_len}')
                ax.grid(alpha=0.3, linewidth=0.5)
                ax.set_ylim(-0.05, 1.05)
                if pre_idx == 0 and post_idx == 2:
                    ax.legend(ncol=2, fontsize=8)

        plt.tight_layout()
        self._save_fig(fig, 'learning_curves_success', subdir)
        plt.close()

    def plot_branch_effective_credit_distance(self, training_scheme):
        """
        Plot effective credit distance vs lambda for branch topologies.

        Creates 1 figure with 3 subplots (one per pre_branch_length).
        Each subplot contains 3 curves (one per post_branch_length).

        Parameters:
        -----------
        training_scheme : str
            Training scheme name for subdirectory organization
        """
        if self.branch_data is None:
            print("Error: No branch data loaded. Run load_branch_data() first.")
            return

        subdir = f'branch/{training_scheme}'

        # Filter to successful episodes only (credit diagnostics computed on success)
        success_data = self.branch_data[self.branch_data['success'] == True].copy()

        if len(success_data) == 0:
            print("Warning: No successful episodes. Skipping effective credit distance plot.")
            return

        # Create colormap for post_branch_lengths
        post_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(self.branch_post_lengths)))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for pre_idx, pre_len in enumerate(self.branch_pre_lengths):
            ax = axes[pre_idx]

            for post_idx, post_len in enumerate(self.branch_post_lengths):
                config_data = success_data[
                    (success_data['pre_branch_length'] == pre_len) &
                    (success_data['post_branch_length'] == post_len)
                ]

                if len(config_data) == 0:
                    continue

                # Compute statistics per lambda
                stats = config_data.groupby('lambda')['effective_credit_distance'].agg(['mean', 'std', 'count'])
                lambdas = stats.index
                means = stats['mean']
                stds = stats['std'].fillna(0)
                ns = stats['count']
                ci = 1.96 * stds / np.sqrt(ns)

                color = post_colors[post_idx]
                ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8,
                       label=f'Post={post_len}', color=color, alpha=0.8)
                ax.fill_between(lambdas, means - ci, means + ci,
                               color=color, alpha=0.2)

            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Effective Credit Distance')
            ax.set_title(f'Pre-branch Length = {pre_len}')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=10, loc='best')

        plt.tight_layout()
        self._save_fig(fig, 'effective_credit_distance', subdir)
        plt.close()

    def plot_branch_auc_vs_lambda(self, training_scheme):
        """
        Plot area under reward curve vs lambda for branch topologies.

        Creates 1 figure with 3 subplots (one per pre_branch_length).
        Each subplot contains 3 curves (one per post_branch_length).

        Parameters:
        -----------
        training_scheme : str
            Training scheme name for subdirectory organization
        """
        if self.branch_data is None:
            print("Error: No branch data loaded. Run load_branch_data() first.")
            return

        subdir = f'branch/{training_scheme}'

        # Compute AUC for each (pre, post, lambda, seed) combination
        auc_results = []

        for pre_len in self.branch_pre_lengths:
            for post_len in self.branch_post_lengths:
                config_data = self.branch_data[
                    (self.branch_data['pre_branch_length'] == pre_len) &
                    (self.branch_data['post_branch_length'] == post_len)
                ]

                for lambda_val in sorted(config_data['lambda'].unique()):
                    for seed in sorted(config_data['seed'].unique()):
                        condition_data = config_data[
                            (config_data['lambda'] == lambda_val) &
                            (config_data['seed'] == seed)
                        ]

                        if len(condition_data) == 0:
                            continue

                        # Compute AUC using mean reward
                        reward_series = condition_data.sort_values('episode')['episode_reward']
                        auc = reward_series.mean()

                        auc_results.append({
                            'pre_branch_length': pre_len,
                            'post_branch_length': post_len,
                            'lambda': lambda_val,
                            'seed': seed,
                            'auc': auc
                        })

        if len(auc_results) == 0:
            print("Warning: No AUC results computed. Skipping AUC plot.")
            return

        auc_df = pd.DataFrame(auc_results)

        # Create colormap for post_branch_lengths
        post_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(self.branch_post_lengths)))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for pre_idx, pre_len in enumerate(self.branch_pre_lengths):
            ax = axes[pre_idx]

            for post_idx, post_len in enumerate(self.branch_post_lengths):
                config_auc = auc_df[
                    (auc_df['pre_branch_length'] == pre_len) &
                    (auc_df['post_branch_length'] == post_len)
                ]

                if len(config_auc) == 0:
                    continue

                # Compute statistics per lambda
                stats = config_auc.groupby('lambda')['auc'].agg(['mean', 'std', 'count'])
                lambdas = stats.index
                means = stats['mean']
                stds = stats['std'].fillna(0)
                ns = stats['count']
                ci = 1.96 * stds / np.sqrt(ns)

                color = post_colors[post_idx]
                ax.plot(lambdas, means, 'o-', linewidth=2, markersize=8,
                       label=f'Post={post_len}', color=color, alpha=0.8)
                ax.fill_between(lambdas, means - ci, means + ci,
                               color=color, alpha=0.2)

            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel('Area Under Reward Curve')
            ax.set_title(f'Pre-branch Length = {pre_len}')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=10, loc='best')

        plt.tight_layout()
        self._save_fig(fig, 'auc_vs_lambda', subdir)
        plt.close()

    def plot_all_corridor(self):
        """Generate all corridor topology plots."""
        print("\n" + "="*80)
        print("Generating Elementary Topology Figures: Corridor Family")
        print("="*80)

        # Load data
        self.load_corridor_data()

        if self.corridor_data is None:
            print("Error: Failed to load corridor data. Exiting.")
            return

        # Generate plots
        print("\n[1/4] Learning Curves...")
        self.plot_corridor_learning_curves()

        print("\n[2/4] Effective Credit Distance...")
        self.plot_effective_credit_distance()

        print("\n[3/4] Area Under Success Rate Curve...")
        self.plot_auc_vs_lambda()

        print("\n[4/4] Value Estimates Per Node...")
        self.plot_value_estimates_per_node()

        print("\n" + "="*80)
        print(f"All figures saved to: {self.output_dir}/corridor/")
        print("="*80)

    def plot_all_branch(self, training_scheme):
        """
        Generate all branch topology plots for a specific training scheme.

        Parameters:
        -----------
        training_scheme : str
            Training scheme subdirectory name ('branch_alternating' or 'branch_switch')
        """
        print("\n" + "="*80)
        print(f"Generating Elementary Topology Figures: Branch Family ({training_scheme})")
        print("="*80)

        # Load data
        self.load_branch_data(training_scheme)

        if self.branch_data is None:
            print("Error: Failed to load branch data. Exiting.")
            return

        # Generate plots
        print("\n[1/3] Learning Curves...")
        self.plot_branch_learning_curves(training_scheme)

        print("\n[2/3] Effective Credit Distance...")
        self.plot_branch_effective_credit_distance(training_scheme)

        print("\n[3/3] Area Under Reward Curve...")
        self.plot_branch_auc_vs_lambda(training_scheme)

        print("\n" + "="*80)
        print(f"All figures saved to: {self.output_dir}/branch/{training_scheme}/")
        print("="*80)


def main():
    """Main entry point for plotting script."""
    parser = argparse.ArgumentParser(
        description='Generate plots for elementary topology experiment results'
    )
    parser.add_argument('results_dir', type=str, default='elementary_topology_results',
                       nargs='?',
                       help='Directory containing topology-specific subdirectories '
                            '(default: elementary_topology_results/)')
    parser.add_argument('--output-dir', type=str, default='elementary_topology_plots',
                       help='Output directory for figures (default: elementary_topology_plots/)')
    parser.add_argument('--topology-family', type=str, default='corridor',
                       help='Topology family to plot. Options: corridor, branch_alternating, branch_switch (default: corridor)')

    args = parser.parse_args()

    # Create plotter
    plotter = ElementaryTopologyPlotter(args.results_dir, args.output_dir)

    # Generate plots based on topology family
    if args.topology_family == 'corridor':
        plotter.plot_all_corridor()
        print("\n✓ Plotting complete!")
        print(f"  View figures in: {plotter.output_dir}/corridor/")
    elif args.topology_family in ['branch_alternating', 'branch_switch']:
        plotter.plot_all_branch(args.topology_family)
        print("\n✓ Plotting complete!")
        print(f"  View figures in: {plotter.output_dir}/branch/{args.topology_family}/")
    else:
        print(f"Error: Unknown topology family '{args.topology_family}'")
        print("Available families: corridor, branch_alternating, branch_switch")
        return


if __name__ == '__main__':
    main()
