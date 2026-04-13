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

    def plot_corridor_entropy_evolution(self):
        """
        Plot evolution of memory call probability during training for corridor topologies.

        Memory call probability = sigmoid((H - tau) / temperature)
        where H is the policy entropy, tau is the threshold (0.6), and temperature is 0.5.

        Creates 2x3 grid of subplots (one per corridor length).
        Each subplot shows curves for different lambda values.
        """
        if self.corridor_data is None:
            print("Error: No corridor data loaded. Run load_corridor_data() first.")
            return

        subdir = 'corridor'
        lambda_values = sorted(self.corridor_data['lambda'].unique())
        window = self.learning_curve_window

        # Check if mean_policy_entropy column exists
        if 'mean_policy_entropy' not in self.corridor_data.columns:
            print("Warning: 'mean_policy_entropy' column not found. Skipping entropy evolution plot.")
            return

        # Get tau and temperature from config (default values if not available)
        tau = 0.6
        temperature = 0.5

        # Compute memory call probability
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        # Create figure with 2x3 subplots
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

                # Compute memory call probability from entropy
                lambda_data = lambda_data.copy()
                lambda_data['memory_call_prob'] = sigmoid(
                    (lambda_data['mean_policy_entropy'] - tau) / temperature
                )

                # Group by episode and compute statistics
                grouped = lambda_data.groupby('episode')['memory_call_prob'].agg(['mean', 'std', 'count'])
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
            ax.set_ylabel(f'Memory Call Probability\n({window}-ep running avg)')
            ax.set_title(f'Corridor Length {length}')
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(ncol=2, fontsize=8)

        plt.tight_layout()
        self._save_fig(fig, 'memory_call_probability_evolution', subdir)
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

    def plot_branch_entropy_evolution(self, training_scheme):
        """
        Plot evolution of memory call probability during training for branch topologies.

        Memory call probability = sigmoid((H - tau) / temperature)
        where H is the policy entropy, tau is the threshold (0.6), and temperature is 0.5.

        Creates 3x3 grid of subplots:
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

        # Check if mean_policy_entropy column exists
        if 'mean_policy_entropy' not in self.branch_data.columns:
            print("Warning: 'mean_policy_entropy' column not found. Skipping entropy evolution plot.")
            return

        # Get tau and temperature from config (default values if not available)
        tau = 0.6
        temperature = 0.5

        # Compute memory call probability
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        # Create figure with 3x3 subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        for pre_idx, pre_len in enumerate(self.branch_pre_lengths):
            for post_idx, post_len in enumerate(self.branch_post_lengths):
                ax = axes[pre_idx, post_idx]

                # Filter data for this specific branch configuration
                branch_config_data = self.branch_data[
                    (self.branch_data['pre_branch_length'] == pre_len) &
                    (self.branch_data['post_branch_length'] == post_len)
                ]

                if len(branch_config_data) == 0:
                    ax.text(0.5, 0.5, 'No data',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Pre={pre_len}, Post={post_len}')
                    continue

                for lambda_idx, lambda_val in enumerate(lambda_values):
                    lambda_data = branch_config_data[branch_config_data['lambda'] == lambda_val]

                    if len(lambda_data) == 0:
                        continue

                    # Compute memory call probability from entropy
                    lambda_data = lambda_data.copy()
                    lambda_data['memory_call_prob'] = sigmoid(
                        (lambda_data['mean_policy_entropy'] - tau) / temperature
                    )

                    # Group by episode and compute statistics
                    grouped = lambda_data.groupby('episode')['memory_call_prob'].agg(['mean', 'std', 'count'])
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
                ax.set_ylabel(f'Memory Call Prob\n({window}-ep avg)')
                ax.set_title(f'Pre={pre_len}, Post={post_len}')
                ax.grid(alpha=0.3, linewidth=0.5)
                ax.set_ylim(-0.05, 1.05)
                if pre_idx == 0 and post_idx == 2:  # Top right for legend
                    ax.legend(ncol=2, fontsize=8)

        plt.tight_layout()
        self._save_fig(fig, 'memory_call_probability_evolution', subdir)
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
        print("\n[1/5] Learning Curves...")
        self.plot_corridor_learning_curves()

        print("\n[2/5] Effective Credit Distance...")
        self.plot_effective_credit_distance()

        print("\n[3/5] Area Under Success Rate Curve...")
        self.plot_auc_vs_lambda()

        print("\n[4/5] Memory Call Probability Evolution...")
        self.plot_corridor_entropy_evolution()

        print("\n[5/5] Value Estimates Per Node...")
        self.plot_value_estimates_per_node()

        print("\n" + "="*80)
        print(f"All figures saved to: {self.output_dir}/corridor/")
        print("="*80)

    def plot_lambda_regime_shift_comparison(self):
        """
        Create central comparison figure showing lambda regime shift between
        alternating and switching training schemes.

        This figure demonstrates that low-λ (local credit assignment) outperforms
        mid-λ in switching schemes when episodic memory is available and task
        structure changes abruptly.

        Creates a 3-panel figure:
        - Panel A: Heatmap comparison of final performance
        - Panel B: Time-series for representative topology showing regime shift
        - Panel C: Corridor length dependency of low-λ advantage
        """
        print("\n" + "="*80)
        print("Generating Lambda Regime Shift Comparison Figure")
        print("="*80)

        # Load both training schemes
        print("\nLoading alternating scheme data...")
        self.load_branch_data('branch_alternating')
        if self.branch_data is None:
            print("Error: Failed to load alternating data")
            return
        alternating_data = self.branch_data.copy()

        print("\nLoading switching scheme data...")
        self.load_branch_data('branch_switch')
        if self.branch_data is None:
            print("Error: Failed to load switching data")
            return
        switching_data = self.branch_data.copy()

        # Create figure with 3 panels
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.3)

        # === PANEL A: Heatmap Comparison ===
        print("\nGenerating Panel A: Performance heatmaps...")
        ax_heat_alt = fig.add_subplot(gs[0, 0])
        ax_heat_switch = fig.add_subplot(gs[0, 1])

        # Compute final performance (last 100 episodes average) for each scheme
        final_window = 100

        for ax, data, scheme_name in [(ax_heat_alt, alternating_data, 'Alternating'),
                                       (ax_heat_switch, switching_data, 'Switching')]:
            # Get final episodes
            max_episode = data['episode'].max()

            # For switching scheme, only use episodes AFTER the switch (second half)
            # For alternating scheme, use last 100 episodes as before
            if scheme_name == 'Switching':
                # Switch happens at midpoint, use last 100 episodes after switch
                switch_point = max_episode // 2
                final_data = data[data['episode'] >= (max_episode - final_window)]
            else:
                final_data = data[data['episode'] >= (max_episode - final_window)]

            # Compute mean reward for each (pre, post, lambda) combination
            perf_matrix = np.zeros((len(self.branch_pre_lengths),
                                   len(self.branch_post_lengths),
                                   len(sorted(data['lambda'].unique()))))

            lambda_values = sorted(data['lambda'].unique())

            for i, pre_len in enumerate(self.branch_pre_lengths):
                for j, post_len in enumerate(self.branch_post_lengths):
                    for k, lambda_val in enumerate(lambda_values):
                        subset = final_data[
                            (final_data['pre_branch_length'] == pre_len) &
                            (final_data['post_branch_length'] == post_len) &
                            (final_data['lambda'] == lambda_val)
                        ]
                        if len(subset) > 0:
                            perf_matrix[i, j, k] = subset['episode_reward'].mean()

            # For each (pre, post), find which lambda performs best
            best_lambda_idx = np.argmax(perf_matrix, axis=2)
            best_lambda_map = np.zeros((len(self.branch_pre_lengths),
                                       len(self.branch_post_lengths)))

            for i in range(len(self.branch_pre_lengths)):
                for j in range(len(self.branch_post_lengths)):
                    best_lambda_map[i, j] = lambda_values[best_lambda_idx[i, j]]

            # Plot heatmap (origin='upper' so pre-branch length increases going down)
            im = ax.imshow(best_lambda_map, cmap='plasma', aspect='auto',
                          vmin=0, vmax=1.0, origin='upper')

            # Annotate with actual lambda values
            for i in range(len(self.branch_pre_lengths)):
                for j in range(len(self.branch_post_lengths)):
                    text_color = 'white' if best_lambda_map[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{best_lambda_map[i, j]:.1f}',
                           ha='center', va='center', color=text_color,
                           fontsize=11, weight='bold')

            ax.set_xticks(range(len(self.branch_post_lengths)))
            ax.set_xticklabels([f'{x}' for x in self.branch_post_lengths])
            ax.set_yticks(range(len(self.branch_pre_lengths)))
            ax.set_yticklabels([f'{x}' for x in self.branch_pre_lengths])
            ax.set_xlabel('Post-Branch Corridor Length', fontsize=12, weight='bold')
            ax.set_ylabel('Pre-Branch Corridor Length', fontsize=12, weight='bold')
            ax.set_title(f'{scheme_name} Scheme\nOptimal λ by Topology',
                        fontsize=13, weight='bold')

        # Add panel labels
        ax_heat_alt.text(-0.15, 1.05, 'A', transform=ax_heat_alt.transAxes,
                        fontsize=18, weight='bold', va='top')
        ax_heat_switch.text(-0.15, 1.05, 'B', transform=ax_heat_switch.transAxes,
                        fontsize=18, weight='bold', va='top')

        # Add colorbar
        cbar = fig.colorbar(im, ax=[ax_heat_alt, ax_heat_switch],
                           orientation='horizontal', pad=0.08, aspect=40)
        cbar.set_label('Optimal λ Value', fontsize=11, weight='bold')

        # === PANEL B: Time-Series Comparison ===
        print("\nGenerating Panel B: Recovery dynamics...")
        ax_timeseries = fig.add_subplot(gs[1, :])

        # Select representative topology (Pre=8, Post=8 shows strongest effect)
        rep_pre, rep_post = 8, 8

        # Define lambda regimes
        lambda_regimes = {
            'Low-λ (0.0-0.2)': [0.0, 0.1, 0.2],
            'Mid-λ (0.5-0.7)': [0.5, 0.6, 0.7],
            'High-λ (0.9-1.0)': [0.9, 1.0]
        }

        regime_colors = {'Low-λ (0.0-0.2)': '#2E86AB',
                        'Mid-λ (0.5-0.7)': '#A23B72',
                        'High-λ (0.9-1.0)': '#F18F01'}

        window = self.learning_curve_window

        for regime_name, regime_lambdas in lambda_regimes.items():
            for scheme_data, scheme_name, linestyle in [(alternating_data, 'Alternating', '--'),
                                                         (switching_data, 'Switching', '-')]:
                # Filter for representative topology and lambda regime
                subset = scheme_data[
                    (scheme_data['pre_branch_length'] == rep_pre) &
                    (scheme_data['post_branch_length'] == rep_post) &
                    (scheme_data['lambda'].isin(regime_lambdas))
                ]

                if len(subset) == 0:
                    continue

                # Compute mean reward across regime lambdas and seeds
                grouped = subset.groupby('episode')['episode_reward'].agg(['mean', 'std', 'count'])
                episodes = grouped.index.to_numpy()
                mean = grouped['mean']
                std = grouped['std'].fillna(0)
                n = grouped['count']
                ci = 1.96 * std / np.sqrt(n)

                # Smooth
                mean_smooth = self._running_average(mean, window)
                ci_smooth = self._running_average(ci, window)

                color = regime_colors[regime_name]
                alpha = 0.9 if linestyle == '-' else 0.5
                linewidth = 2.5 if linestyle == '-' else 2.0

                label = f'{regime_name} ({scheme_name})'
                ax_timeseries.plot(episodes, mean_smooth, label=label,
                                  color=color, linestyle=linestyle,
                                  linewidth=linewidth, alpha=alpha)

                if linestyle == '-':  # Only shade switching scheme
                    ax_timeseries.fill_between(episodes,
                                               mean_smooth - ci_smooth,
                                               mean_smooth + ci_smooth,
                                               color=color, alpha=0.15)

        # Mark the switch point
        max_episode = switching_data['episode'].max()
        switch_point = max_episode // 2
        ax_timeseries.axvline(x=switch_point, color='red', linestyle=':',
                             linewidth=2, alpha=0.7, label='Target Switch')

        ax_timeseries.set_xlabel('Episode', fontsize=12, weight='bold')
        ax_timeseries.set_ylabel(f'Episode Reward\n({window}-ep running avg)',
                                fontsize=12, weight='bold')
        ax_timeseries.set_title(f'Recovery Dynamics (Pre={rep_pre}, Post={rep_post})',
                               fontsize=13, weight='bold')
        ax_timeseries.grid(alpha=0.3, linewidth=0.5)
        ax_timeseries.legend(ncol=3, fontsize=9, loc='lower left')

        # Add panel label
        ax_timeseries.text(-0.08, 1.05, 'C', transform=ax_timeseries.transAxes,
                          fontsize=18, weight='bold', va='top')

        # === PANEL C: Corridor Length Dependency ===
        print("\nGenerating Panel C: Corridor length dependency...")
        ax_dependency = fig.add_subplot(gs[2, :])

        # Compute performance advantage of low-λ over mid-λ in switching scheme
        # Advantage = (low-λ final reward) - (mid-λ final reward)

        final_window = 100
        max_episode = switching_data['episode'].max()
        final_switching = switching_data[switching_data['episode'] >= (max_episode - final_window)]

        low_lambdas = [0.0, 0.1, 0.2]
        mid_lambdas = [0.5, 0.6, 0.7]

        # For each (pre, post), compute advantage
        advantages = []

        for pre_len in self.branch_pre_lengths:
            for post_len in self.branch_post_lengths:
                # Low-λ performance
                low_data = final_switching[
                    (final_switching['pre_branch_length'] == pre_len) &
                    (final_switching['post_branch_length'] == post_len) &
                    (final_switching['lambda'].isin(low_lambdas))
                ]

                # Mid-λ performance
                mid_data = final_switching[
                    (final_switching['pre_branch_length'] == pre_len) &
                    (final_switching['post_branch_length'] == post_len) &
                    (final_switching['lambda'].isin(mid_lambdas))
                ]

                if len(low_data) > 0 and len(mid_data) > 0:
                    low_reward = low_data['episode_reward'].mean()
                    mid_reward = mid_data['episode_reward'].mean()
                    advantage = low_reward - mid_reward

                    advantages.append({
                        'pre': pre_len,
                        'post': post_len,
                        'advantage': advantage
                    })

        adv_df = pd.DataFrame(advantages)

        # Plot as grouped bar chart
        post_colors_dep = plt.cm.viridis(np.linspace(0.2, 0.8, len(self.branch_post_lengths)))
        x = np.arange(len(self.branch_pre_lengths))
        width = 0.25

        for i, post_len in enumerate(self.branch_post_lengths):
            post_data = adv_df[adv_df['post'] == post_len]
            if len(post_data) == 0:
                continue

            # Sort by pre length to match x positions
            post_data = post_data.sort_values('pre')
            advantages_arr = post_data['advantage'].values

            offset = width * (i - 1)  # Center the bars
            ax_dependency.bar(x + offset, advantages_arr, width,
                            label=f'Post={post_len}',
                            color=post_colors_dep[i], alpha=0.8)

        ax_dependency.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax_dependency.set_xlabel('Pre-Branch Corridor Length', fontsize=12, weight='bold')
        ax_dependency.set_ylabel('Low-λ Advantage over Mid-λ\n(Δ Reward)',
                                fontsize=12, weight='bold')
        ax_dependency.set_title('Corridor Length Modulation of Low-λ Advantage (Switching Scheme)',
                               fontsize=13, weight='bold')
        ax_dependency.set_xticks(x)
        ax_dependency.set_xticklabels([f'{pre}' for pre in self.branch_pre_lengths])
        ax_dependency.legend(fontsize=10, loc='upper left', ncol=3)
        ax_dependency.grid(alpha=0.3, linewidth=0.5, axis='y')

        # Add panel label
        ax_dependency.text(-0.08, 1.05, 'D', transform=ax_dependency.transAxes,
                          fontsize=18, weight='bold', va='top')

        # Add main title
        fig.suptitle('Lambda Regime Shift: Low-λ Outperforms Mid-λ in Switching Schemes with Episodic Memory',
                    fontsize=15, weight='bold', y=0.995)

        # Save figure
        self._save_fig(fig, 'lambda_regime_shift_comparison', 'branch')
        plt.close()

        print("\n" + "="*80)
        print(f"Central figure saved to: {self.output_dir}/branch/lambda_regime_shift_comparison.png")
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
        print("\n[1/4] Learning Curves...")
        self.plot_branch_learning_curves(training_scheme)

        print("\n[2/4] Effective Credit Distance...")
        self.plot_branch_effective_credit_distance(training_scheme)

        print("\n[3/4] Area Under Reward Curve...")
        self.plot_branch_auc_vs_lambda(training_scheme)

        print("\n[4/4] Memory Call Probability Evolution...")
        self.plot_branch_entropy_evolution(training_scheme)

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
                       help='Topology family to plot. Options: corridor, branch_alternating, branch_switch, branch_stagnant, regime_shift (default: corridor)')

    args = parser.parse_args()

    # Create plotter
    plotter = ElementaryTopologyPlotter(args.results_dir, args.output_dir)

    # Generate plots based on topology family
    if args.topology_family == 'corridor':
        plotter.plot_all_corridor()
        print("\n✓ Plotting complete!")
        print(f"  View figures in: {plotter.output_dir}/corridor/")
    elif args.topology_family in ['branch_alternating', 'branch_switch', 'branch_stagnant']:
        plotter.plot_all_branch(args.topology_family)
        print("\n✓ Plotting complete!")
        print(f"  View figures in: {plotter.output_dir}/branch/{args.topology_family}/")
    elif args.topology_family == 'regime_shift':
        plotter.plot_lambda_regime_shift_comparison()
        print("\n✓ Plotting complete!")
        print(f"  View figures in: {plotter.output_dir}/branch/")
    else:
        print(f"Error: Unknown topology family '{args.topology_family}'")
        print("Available families: corridor, branch_alternating, branch_switch, branch_stagnant, regime_shift")
        return


if __name__ == '__main__':
    main()
