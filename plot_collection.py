"""
Plot Collection Results

Generates comprehensive plots from train_collection.py outputs across 11 corridor parameters.

Plots:
1. Fast/slow usage by junction density bins (4 bins) over training
2. Fast/slow usage by node type (junction/corridor/dead-end) over training
3. Entropy/divergence/conflict by system usage (fast vs slow) over training
4. Lambda and conflict map values by node type over training
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Publication-style formatting
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


class CollectionPlotter:
    """Load and plot results from train_collection.py."""

    def __init__(self, results_dir='results/collection', output_dir='results/collection_figures',
                 smooth_window=100):
        """
        Initialize plotter.

        Parameters:
        -----------
        results_dir : str
            Directory containing corridor_X.X subdirectories
        output_dir : str
            Directory to save plots
        smooth_window : int
            Window size for smoothing trajectories
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.smooth_window = smooth_window

        # Load all metrics
        self.metrics_by_corridor = {}
        self.corridor_params = []
        self._load_all_metrics()

    def _load_all_metrics(self):
        """Load metrics.json from all corridor parameter directories."""
        print("Loading metrics from all corridor parameters...")

        for corridor in np.arange(0, 1.1, 0.1):
            corridor_dir = self.results_dir / f'corridor_{corridor:.1f}'
            metrics_path = corridor_dir / 'metrics.json'

            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    self.metrics_by_corridor[corridor] = metrics
                    self.corridor_params.append(corridor)
                    print(f"  Loaded corridor={corridor:.1f} ({len(metrics.get('episode_rewards', []))} episodes)")
            else:
                print(f"  Warning: {metrics_path} not found, skipping")

        if len(self.metrics_by_corridor) == 0:
            raise FileNotFoundError(f"No metrics found in {self.results_dir}")

        self.corridor_params = sorted(self.corridor_params)
        print(f"\nLoaded {len(self.corridor_params)} corridor parameters: {self.corridor_params}")

    def _smooth(self, data, window=None):
        """Apply uniform smoothing to 1D data."""
        if window is None:
            window = self.smooth_window

        if len(data) < window:
            return data

        # Handle NaN values with forward fill (carry last valid value forward)
        data_array = np.array(data, dtype=float)

        # Forward fill NaNs
        mask = np.isnan(data_array)
        if np.any(mask):
            # Get indices of valid values
            valid_indices = np.where(~mask)[0]

            if len(valid_indices) == 0:
                # All NaN, return as is
                return data_array

            # Forward fill: carry last valid value forward
            filled = data_array.copy()
            last_valid = valid_indices[0]
            for i in range(len(filled)):
                if not mask[i]:
                    last_valid = i
                else:
                    filled[i] = filled[last_valid]

            data_array = filled

        # Simple uniform filter using numpy convolution
        kernel = np.ones(window) / window
        # Use 'same' mode to keep same length, and handle edges by padding
        smoothed = np.convolve(data_array, kernel, mode='same')
        return smoothed

    def _get_junction_density_bins(self, num_bins=4):
        """
        Bin corridor parameters by their junction density.

        Parameters:
        -----------
        num_bins : int
            Number of bins

        Returns:
        --------
        bins : list of lists
            Each element is a list of corridor parameters in that bin
        bin_labels : list of str
            Label for each bin
        """
        # Get junction densities for all mazes
        junction_densities = []
        for corridor in self.corridor_params:
            metrics = self.metrics_by_corridor[corridor]
            # Junction density should be constant per maze
            jd = metrics['junction_density'][0]
            junction_densities.append((corridor, jd))

        # Sort by junction density
        junction_densities.sort(key=lambda x: x[1])

        # Create bins
        bins_per_bin = len(junction_densities) // num_bins
        remainder = len(junction_densities) % num_bins

        bins = []
        bin_labels = []
        start_idx = 0

        for i in range(num_bins):
            # Distribute remainder across first bins
            bin_size = bins_per_bin + (1 if i < remainder else 0)
            end_idx = start_idx + bin_size

            bin_corridors = [jd[0] for jd in junction_densities[start_idx:end_idx]]
            bin_jd_values = [jd[1] for jd in junction_densities[start_idx:end_idx]]

            bins.append(bin_corridors)
            bin_labels.append(f'JD={np.mean(bin_jd_values):.2f}')

            start_idx = end_idx

        return bins, bin_labels

    def _compute_mean_and_sem(self, metric_key, corridor_list):
        """
        Compute mean and SEM across a list of corridor parameters for a metric.

        Parameters:
        -----------
        metric_key : str
            Key in metrics dictionary
        corridor_list : list of float
            Corridor parameters to average over

        Returns:
        --------
        mean : np.ndarray
            Mean trajectory
        sem : np.ndarray
            Standard error of mean
        """
        trajectories = []

        for corridor in corridor_list:
            metrics = self.metrics_by_corridor[corridor]
            if metric_key in metrics:
                traj = np.array(metrics[metric_key], dtype=float)
                # Skip if all values are NaN or None
                if not np.all(np.isnan(traj)):
                    trajectories.append(traj)

        if len(trajectories) == 0:
            return None, None

        # Stack trajectories
        trajectories = np.array(trajectories)

        # Check if we have any valid data
        if trajectories.size == 0:
            return None, None

        # Compute mean and SEM
        with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
            mean = np.nanmean(trajectories, axis=0)
            sem = np.nanstd(trajectories, axis=0) / np.sqrt(np.sum(~np.isnan(trajectories), axis=0))

        # If mean is all NaN, return None
        if np.all(np.isnan(mean)):
            return None, None

        return mean, sem

    def plot_usage_by_junction_density(self, ax_fast, ax_slow):
        """
        Plot fast/slow usage rates binned by junction density.

        Parameters:
        -----------
        ax_fast : matplotlib.axes.Axes
            Axes for fast usage plot
        ax_slow : matplotlib.axes.Axes
            Axes for slow usage plot
        """
        bins, bin_labels = self._get_junction_density_bins(num_bins=4)

        # Color palette for bins
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        for ax, metric_key, title in [
            (ax_fast, 'p_fast', 'Fast System Usage by Junction Density'),
            (ax_slow, 'p_slow', 'Slow System Usage by Junction Density')
        ]:
            for i, (bin_corridors, label) in enumerate(zip(bins, bin_labels)):
                mean, sem = self._compute_mean_and_sem(metric_key, bin_corridors)

                if mean is not None:
                    # Smooth
                    mean_smooth = self._smooth(mean)
                    sem_smooth = self._smooth(sem)

                    episodes = np.arange(len(mean_smooth))

                    # Plot mean
                    ax.plot(episodes, mean_smooth, color=colors[i], linewidth=2, label=label)

                    # Plot confidence interval
                    ax.fill_between(episodes,
                                   mean_smooth - sem_smooth,
                                   mean_smooth + sem_smooth,
                                   color=colors[i], alpha=0.2)

            ax.set_xlabel('Episode')
            ax.set_ylabel('Usage Rate')
            ax.set_title(title, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(alpha=0.3)

    def plot_usage_by_node_type(self, ax_fast, ax_slow):
        """
        Plot fast/slow usage rates by node type (averaged across all mazes).

        Parameters:
        -----------
        ax_fast : matplotlib.axes.Axes
            Axes for fast usage plot
        ax_slow : matplotlib.axes.Axes
            Axes for slow usage plot
        """
        node_types = ['junction', 'corridor', 'dead_end']
        colors = {'junction': '#1D3557', 'corridor': '#A8DADC', 'dead_end': '#E63946'}
        labels = {'junction': 'Junction', 'corridor': 'Corridor', 'dead_end': 'Dead-end'}

        for ax, system, title in [
            (ax_fast, 'fast', 'Fast System Usage by Node Type'),
            (ax_slow, 'slow', 'Slow System Usage by Node Type')
        ]:
            for node_type in node_types:
                metric_key = f'{node_type}_use_{system}_rate'
                mean, sem = self._compute_mean_and_sem(metric_key, self.corridor_params)

                if mean is not None:
                    # Smooth
                    mean_smooth = self._smooth(mean)
                    sem_smooth = self._smooth(sem)

                    episodes = np.arange(len(mean_smooth))

                    # Plot mean
                    ax.plot(episodes, mean_smooth, color=colors[node_type],
                           linewidth=2, label=labels[node_type])

                    # Plot confidence interval
                    ax.fill_between(episodes,
                                   mean_smooth - sem_smooth,
                                   mean_smooth + sem_smooth,
                                   color=colors[node_type], alpha=0.2)

            ax.set_xlabel('Episode')
            ax.set_ylabel('Usage Rate')
            ax.set_title(title, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(alpha=0.3)

    def plot_state_properties_by_system(self, ax_entropy, ax_divergence, ax_conflict):
        """
        Plot state properties (entropy, divergence, conflict) by system usage.

        Parameters:
        -----------
        ax_entropy : matplotlib.axes.Axes
            Axes for entropy plot
        ax_divergence : matplotlib.axes.Axes
            Axes for divergence plot
        ax_conflict : matplotlib.axes.Axes
            Axes for conflict map value plot
        """
        properties = [
            ('entropy', ax_entropy, 'Entropy by System Usage'),
            ('divergence', ax_divergence, 'KL Divergence by System Usage'),
            ('conflict', ax_conflict, 'Conflict Map Value by System Usage')
        ]

        colors = {'fast': '#457B9D', 'slow': '#E63946'}
        labels = {'fast': 'Used Fast', 'slow': 'Used Slow'}

        for prop_name, ax, title in properties:
            for system in ['fast', 'slow']:
                metric_key = f'used_{system}_mean_{prop_name}'
                mean, sem = self._compute_mean_and_sem(metric_key, self.corridor_params)

                if mean is not None:
                    # Smooth
                    mean_smooth = self._smooth(mean)
                    sem_smooth = self._smooth(sem)

                    episodes = np.arange(len(mean_smooth))

                    # Plot mean
                    ax.plot(episodes, mean_smooth, color=colors[system],
                           linewidth=2, label=labels[system])

                    # Plot confidence interval
                    ax.fill_between(episodes,
                                   mean_smooth - sem_smooth,
                                   mean_smooth + sem_smooth,
                                   color=colors[system], alpha=0.2)

            ax.set_xlabel('Episode')
            ax.set_ylabel(prop_name.capitalize())
            ax.set_title(title, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(alpha=0.3)

    def plot_node_type_metrics(self, ax_lambda, ax_conflict, ax_entropy=None, ax_kl=None):
        """
        Plot lambda, conflict map, entropy, and KL divergence values by node type.

        Parameters:
        -----------
        ax_lambda : matplotlib.axes.Axes
            Axes for lambda plot
        ax_conflict : matplotlib.axes.Axes
            Axes for conflict map value plot
        ax_entropy : matplotlib.axes.Axes, optional
            Axes for entropy plot
        ax_kl : matplotlib.axes.Axes, optional
            Axes for KL divergence plot
        """
        node_types = ['junction', 'corridor', 'dead_end']
        colors = {'junction': '#1D3557', 'corridor': '#A8DADC', 'dead_end': '#E63946'}
        labels = {'junction': 'Junction', 'corridor': 'Corridor', 'dead_end': 'Dead-end'}

        # Define metrics to plot
        metrics_config = [
            (ax_lambda, 'lambda', 'Lambda (λ)', 'TD(λ) Parameter by Node Type'),
            (ax_conflict, 'conflict', 'Conflict Value', 'Conflict Map Value by Node Type')
        ]

        # Add entropy and KL if axes provided
        if ax_entropy is not None:
            metrics_config.append((ax_entropy, 'entropy', 'Entropy', 'Fast Entropy by Node Type'))
        if ax_kl is not None:
            metrics_config.append((ax_kl, 'kl', 'KL Divergence', 'KL Divergence by Node Type'))

        missing_metrics = set()
        for ax, metric_type, ylabel, title in metrics_config:
            plotted_any = False
            for node_type in node_types:
                # Handle different metric naming conventions
                if metric_type == 'entropy':
                    metric_key = f'{node_type}_mean_entropy'
                    if metric_key not in self.metrics_by_corridor[self.corridor_params[0]]:
                        missing_metrics.add(metric_key)
                        continue
                elif metric_type == 'kl':
                    metric_key = f'{node_type}_mean_divergence'
                    if metric_key not in self.metrics_by_corridor[self.corridor_params[0]]:
                        missing_metrics.add(metric_key)
                        continue
                else:
                    metric_key = f'{node_type}_mean_{metric_type}'

                mean, sem = self._compute_mean_and_sem(metric_key, self.corridor_params)

                if mean is not None:
                    plotted_any = True
                    # Smooth
                    mean_smooth = self._smooth(mean)
                    sem_smooth = self._smooth(sem)

                    episodes = np.arange(len(mean_smooth))

                    # Plot mean
                    ax.plot(episodes, mean_smooth, color=colors[node_type],
                           linewidth=2, label=labels[node_type])

                    # Plot confidence interval
                    ax.fill_between(episodes,
                                   mean_smooth - sem_smooth,
                                   mean_smooth + sem_smooth,
                                   color=colors[node_type], alpha=0.2)

            ax.set_xlabel('Episode')
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight='bold')

            # Only add legend if there are any lines plotted
            handles, labels_list = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend(loc='best')
            elif not plotted_any:
                # Add text indicating no data
                ax.text(0.5, 0.5, 'No data available',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='gray')

            ax.grid(alpha=0.3)

            # Set lambda range to [0, 1] for lambda plot
            if metric_type == 'lambda':
                ax.set_ylim([0, 1.05])

        # Warn about missing metrics if any
        if missing_metrics:
            print(f"    Warning: Missing metrics in data: {', '.join(sorted(missing_metrics))}")
            print(f"    These metrics may need to be collected by re-running train_collection.py")

    def plot_lambda_by_junction_density(self, ax_overall, ax_junction, ax_corridor, ax_dead_end):
        """
        Plot lambda values by junction density bins.

        Parameters:
        -----------
        ax_overall : matplotlib.axes.Axes
            Axes for overall mean lambda
        ax_junction : matplotlib.axes.Axes
            Axes for junction lambda
        ax_corridor : matplotlib.axes.Axes
            Axes for corridor lambda
        ax_dead_end : matplotlib.axes.Axes
            Axes for dead-end lambda
        """
        bins, bin_labels = self._get_junction_density_bins(num_bins=4)
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        # Plot overall mean lambda
        for i, (bin_corridors, label) in enumerate(zip(bins, bin_labels)):
            mean, sem = self._compute_mean_and_sem('mean_lambda', bin_corridors)

            if mean is not None:
                mean_smooth = self._smooth(mean)
                sem_smooth = self._smooth(sem)
                episodes = np.arange(len(mean_smooth))

                ax_overall.plot(episodes, mean_smooth, color=colors[i], linewidth=2, label=label)
                ax_overall.fill_between(episodes,
                                       mean_smooth - sem_smooth,
                                       mean_smooth + sem_smooth,
                                       color=colors[i], alpha=0.2)

        ax_overall.set_xlabel('Episode')
        ax_overall.set_ylabel('Lambda (λ)')
        ax_overall.set_title('Overall Mean Lambda by Junction Density', fontweight='bold')
        ax_overall.set_ylim([0, 1.05])
        ax_overall.legend(loc='best')
        ax_overall.grid(alpha=0.3)

        # Plot lambda by node type for each junction density bin
        node_configs = [
            (ax_junction, 'junction', 'Junction Lambda by Junction Density'),
            (ax_corridor, 'corridor', 'Corridor Lambda by Junction Density'),
            (ax_dead_end, 'dead_end', 'Dead-end Lambda by Junction Density')
        ]

        for ax, node_type, title in node_configs:
            for i, (bin_corridors, label) in enumerate(zip(bins, bin_labels)):
                metric_key = f'{node_type}_mean_lambda'
                mean, sem = self._compute_mean_and_sem(metric_key, bin_corridors)

                if mean is not None:
                    mean_smooth = self._smooth(mean)
                    sem_smooth = self._smooth(sem)
                    episodes = np.arange(len(mean_smooth))

                    ax.plot(episodes, mean_smooth, color=colors[i], linewidth=2, label=label)
                    ax.fill_between(episodes,
                                   mean_smooth - sem_smooth,
                                   mean_smooth + sem_smooth,
                                   color=colors[i], alpha=0.2)

            ax.set_xlabel('Episode')
            ax.set_ylabel('Lambda (λ)')
            ax.set_title(title, fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.legend(loc='best')
            ax.grid(alpha=0.3)

    def create_figure_usage_by_junction_density(self):
        """Create figure: System usage by junction density."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        self.plot_usage_by_junction_density(axes[0], axes[1])

        fig.suptitle('System Usage by Junction Density', fontsize=14, fontweight='bold')
        plt.tight_layout()

        png_path = self.output_dir / 'usage_by_junction_density.png'
        pdf_path = self.output_dir / 'usage_by_junction_density.pdf'
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')

        print(f"  Saved: {png_path.name}")
        return fig

    def create_figure_usage_by_node_type(self):
        """Create figure: System usage by node type."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        self.plot_usage_by_node_type(axes[0], axes[1])

        fig.suptitle('System Usage by Node Type', fontsize=14, fontweight='bold')
        plt.tight_layout()

        png_path = self.output_dir / 'usage_by_node_type.png'
        pdf_path = self.output_dir / 'usage_by_node_type.pdf'
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')

        print(f"  Saved: {png_path.name}")
        return fig

    def create_figure_state_properties(self):
        """Create figure: State properties by system usage."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        self.plot_state_properties_by_system(axes[0], axes[1], axes[2])

        fig.suptitle('State Properties by System Usage', fontsize=14, fontweight='bold')
        plt.tight_layout()

        png_path = self.output_dir / 'state_properties_by_system.png'
        pdf_path = self.output_dir / 'state_properties_by_system.pdf'
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')

        print(f"  Saved: {png_path.name}")
        return fig

    def create_figure_node_type_metrics(self):
        """Create figure: Lambda, conflict, entropy, and KL divergence by node type."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        self.plot_node_type_metrics(
            axes[0, 0],  # Lambda
            axes[0, 1],  # Conflict
            axes[1, 0],  # Entropy
            axes[1, 1]   # KL divergence
        )

        fig.suptitle('Metrics by Node Type', fontsize=14, fontweight='bold')
        plt.tight_layout()

        png_path = self.output_dir / 'metrics_by_node_type.png'
        pdf_path = self.output_dir / 'metrics_by_node_type.pdf'
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')

        print(f"  Saved: {png_path.name}")
        return fig

    def create_figure_lambda_by_junction_density(self):
        """Create figure: Lambda by junction density (overall + per node type)."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        self.plot_lambda_by_junction_density(
            axes[0, 0],  # Overall
            axes[0, 1],  # Junction
            axes[1, 0],  # Corridor
            axes[1, 1]   # Dead-end
        )

        fig.suptitle('Lambda by Junction Density', fontsize=14, fontweight='bold')
        plt.tight_layout()

        png_path = self.output_dir / 'lambda_by_junction_density.png'
        pdf_path = self.output_dir / 'lambda_by_junction_density.pdf'
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')

        print(f"  Saved: {png_path.name}")
        return fig

    def create_all_figures(self):
        """Create all figure families."""
        print("\nGenerating figures...")

        self.create_figure_usage_by_junction_density()
        self.create_figure_usage_by_node_type()
        self.create_figure_state_properties()
        self.create_figure_node_type_metrics()
        self.create_figure_lambda_by_junction_density()

        print("\n✓ All figures saved!")


def main():
    """Main plotting function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot results from train_collection.py'
    )
    parser.add_argument('--results_dir', type=str, default='results/collection',
                       help='Directory containing corridor_X.X subdirectories')
    parser.add_argument('--output_dir', type=str, default='results/collection_figures',
                       help='Directory to save plots')
    parser.add_argument('--smooth_window', type=int, default=100,
                       help='Window size for smoothing trajectories')

    args = parser.parse_args()

    print("=" * 70)
    print("COLLECTION PLOTTER")
    print("=" * 70)

    # Create plotter
    plotter = CollectionPlotter(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        smooth_window=args.smooth_window
    )

    # Generate all figures
    plotter.create_all_figures()

    print(f"\nAll figures saved to: {plotter.output_dir}")


if __name__ == "__main__":
    main()
