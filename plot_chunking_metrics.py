import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import os
import pickle

# Set publication-quality style (matching plot_loss.py)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 6
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Use a clean color palette
sns.set_palette("deep")


def load_stats_from_pickle(results_dir, experiment_name):
    """
    Load training stats from a pickle file.

    Args:
        results_dir: str - directory containing the results (e.g., 'results/td_n')
        experiment_name: str - experiment name

    Returns:
        dict - training statistics
    """
    pickle_path = os.path.join(results_dir, 'stats', f'{experiment_name}_stats.pkl')

    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Stats file not found: {pickle_path}")

    with open(pickle_path, 'rb') as f:
        stats = pickle.load(f)

    return stats


def load_stats_from_csv(results_dir, experiment_name):
    """
    Load training stats from CSV files.

    Args:
        results_dir: str - directory containing the results (e.g., 'results/td_n')
        experiment_name: str - experiment name

    Returns:
        dict - training statistics
    """
    metrics_dir = os.path.join(results_dir, 'metrics')

    # Load chunking indices
    chunking_path = os.path.join(metrics_dir, f'{experiment_name}_chunking_indices.csv')
    if not os.path.exists(chunking_path):
        raise FileNotFoundError(f"Chunking indices file not found: {chunking_path}")

    df_chunking = pd.read_csv(chunking_path)
    episodes = df_chunking['episode'].values
    chunking_indices = df_chunking['chunking_index'].values

    # Convert 'None' strings back to None
    chunking_indices = [None if str(ci) == 'None' else float(ci) for ci in chunking_indices]

    # For per-step entropies, we need to reconstruct from the trajectory
    # Since we don't save the full per_step_entropies_list to CSV, we'll create a minimal version
    # This will allow chunking index plotting but not per-step entropy plotting
    per_step_entropies_list = []

    stats = {
        'captured_episodes': episodes.tolist(),
        'chunking_indices': chunking_indices,
        'per_step_entropies_list': per_step_entropies_list  # Empty - won't plot Figure A
    }

    return stats


def plot_per_step_entropy(stats, experiment_name, save_dir='results/td_n', show=False):
    """
    Figure A: Plot per-step entropy over training.

    One line per decision point in the canonical trajectory.
    X-axis: training episode
    Y-axis: entropy

    Args:
        stats: dict - training statistics containing 'per_step_entropies_list'
        experiment_name: str - name for this experiment
        save_dir: str - directory to save plots
        show: bool - whether to display plot
    """
    if 'per_step_entropies_list' not in stats or len(stats['per_step_entropies_list']) == 0:
        print("Warning: No per-step entropy data found in stats. Skipping Figure A.")
        return

    # Extract data
    episodes = stats['captured_episodes']
    per_step_data_list = stats['per_step_entropies_list']

    # Determine number of decision points from the first captured episode
    first_data = per_step_data_list[0]
    num_decision_points = len(first_data['entropies'])
    states = first_data['states']

    if num_decision_points == 0:
        print("Warning: No decision points found in trajectory. Skipping Figure A.")
        return

    # Organize data: entropy_matrix[decision_point][episode]
    entropy_matrix = [[] for _ in range(num_decision_points)]

    for data in per_step_data_list:
        entropies = data['entropies']
        for i, entropy in enumerate(entropies):
            if i < num_decision_points:
                entropy_matrix[i].append(entropy)

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    # Get color palette
    colors = sns.color_palette("deep", num_decision_points)

    # Plot one line per decision point
    for i in range(num_decision_points):
        state_label = f"State {states[i]}"
        ax.plot(episodes, entropy_matrix[i],
                color=colors[i], linewidth=2.5,
                label=state_label, marker='o',
                markevery=max(1, len(episodes)//20), markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.5, markeredgecolor=colors[i])

    # Add grid for readability
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)  # Grid behind plot elements

    # Labels and title
    ax.set_xlabel('Training Episode', fontweight='normal')
    ax.set_ylabel('Entropy (bits)', fontweight='normal')
    ax.set_title('Per-Step Policy Entropy over Training',
                 fontweight='bold', pad=15)

    # Legend
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray',
              fancybox=True, shadow=False, ncol=1 if num_decision_points <= 5 else 2)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save in multiple formats
    os.makedirs(f'{save_dir}/plots', exist_ok=True)
    plt.savefig(f'{save_dir}/plots/{experiment_name}_per_step_entropy.png',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'{save_dir}/plots/{experiment_name}_per_step_entropy.pdf',
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'{save_dir}/plots/{experiment_name}_per_step_entropy.svg',
                bbox_inches='tight', facecolor='white', edgecolor='none')

    print(f"\nPer-step entropy plots saved to:")
    print(f"  - {save_dir}/plots/{experiment_name}_per_step_entropy.png")
    print(f"  - {save_dir}/plots/{experiment_name}_per_step_entropy.pdf")
    print(f"  - {save_dir}/plots/{experiment_name}_per_step_entropy.svg")

    if show:
        plt.show()
    else:
        plt.close()


def plot_chunking_index(stats, experiment_name, save_dir='results/td_n', show=False, plot_total_entropy=True):
    """
    Figure B: Plot chunking index over training.

    Single line plot showing chunking index evolution.
    Optional: overlay total_entropy on secondary y-axis.

    Args:
        stats: dict - training statistics containing 'chunking_indices' and 'per_step_entropies_list'
        experiment_name: str - name for this experiment
        save_dir: str - directory to save plots
        show: bool - whether to display plot
        plot_total_entropy: bool - whether to plot total entropy on secondary y-axis
    """
    if 'chunking_indices' not in stats or len(stats['chunking_indices']) == 0:
        print("Warning: No chunking index data found in stats. Skipping Figure B.")
        return

    # Extract data
    episodes = stats['captured_episodes']
    chunking_indices = stats['chunking_indices']

    # Filter out None values for plotting
    valid_indices = [(ep, ci) for ep, ci in zip(episodes, chunking_indices) if ci is not None]

    if len(valid_indices) == 0:
        print("Warning: All chunking indices are None. Skipping Figure B.")
        return

    valid_episodes = [ep for ep, _ in valid_indices]
    valid_chunking_indices = [ci for _, ci in valid_indices]

    # Create figure with optional secondary y-axis
    if plot_total_entropy and 'per_step_entropies_list' in stats:
        fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)
        ax2 = ax1.twinx()
    else:
        fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)
        ax2 = None

    # Plot chunking index
    color_chunking = '#2E86AB'
    ax1.plot(valid_episodes, valid_chunking_indices,
             color=color_chunking, linewidth=2.5,
             label='Chunking Index', marker='o',
             markevery=max(1, len(valid_episodes)//20), markersize=5,
             markerfacecolor='white',
             markeredgewidth=1.5, markeredgecolor=color_chunking)

    # Add horizontal reference line at 0
    ax1.axhline(y=0.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)

    # Labels for primary axis
    ax1.set_xlabel('Training Episode', fontweight='normal')
    ax1.set_ylabel('Chunking Index', fontweight='normal', color=color_chunking)
    ax1.tick_params(axis='y', labelcolor=color_chunking)

    # Add grid for readability
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)  # Grid behind plot elements

    # Plot total entropy on secondary y-axis if requested
    if ax2 is not None and 'per_step_entropies_list' in stats and len(stats['per_step_entropies_list']) > 0:
        per_step_data_list = stats['per_step_entropies_list']
        total_entropies = [data['total_entropy'] for data in per_step_data_list]

        color_entropy = '#A23B72'
        ax2.plot(episodes, total_entropies,
                 color=color_entropy, linewidth=2.5,
                 label='Total Entropy', marker='s',
                 markevery=max(1, len(episodes)//20), markersize=5,
                 markerfacecolor='white',
                 markeredgewidth=1.5, markeredgecolor=color_entropy,
                 linestyle='--', alpha=0.7)

        # Labels for secondary axis
        ax2.set_ylabel('Total Entropy (sum)', fontweight='normal', color=color_entropy)
        ax2.tick_params(axis='y', labelcolor=color_entropy)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='best', framealpha=0.95, edgecolor='gray',
                   fancybox=True, shadow=False)
    else:
        # No total entropy data available, just plot chunking index
        ax1.legend(loc='best', framealpha=0.95, edgecolor='gray',
                   fancybox=True, shadow=False)

    # Title
    ax1.set_title('Chunking Index over Training',
                  fontweight='bold', pad=15)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save in multiple formats
    os.makedirs(f'{save_dir}/plots', exist_ok=True)
    plt.savefig(f'{save_dir}/plots/{experiment_name}_chunking_index.png',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'{save_dir}/plots/{experiment_name}_chunking_index.pdf',
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'{save_dir}/plots/{experiment_name}_chunking_index.svg',
                bbox_inches='tight', facecolor='white', edgecolor='none')

    print(f"\nChunking index plots saved to:")
    print(f"  - {save_dir}/plots/{experiment_name}_chunking_index.png")
    print(f"  - {save_dir}/plots/{experiment_name}_chunking_index.pdf")
    print(f"  - {save_dir}/plots/{experiment_name}_chunking_index.svg")

    if show:
        plt.show()
    else:
        plt.close()


def plot_chunking_metrics_from_stats(stats, experiment_name, save_dir='results/td_n', show=False):
    """
    Generate both Figure A and Figure B from training stats.

    Args:
        stats: dict - training statistics
        experiment_name: str - name for this experiment
        save_dir: str - directory to save plots
        show: bool - whether to display plots
    """
    print(f"\n{'='*60}")
    print(f"Generating Chunking Metrics Plots for {experiment_name}")
    print(f"{'='*60}")

    # Figure A: Per-step entropy
    plot_per_step_entropy(stats, experiment_name, save_dir, show)

    # Figure B: Chunking index
    plot_chunking_index(stats, experiment_name, save_dir, show, plot_total_entropy=True)

    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python plot_chunking_metrics.py <experiment_name> [save_dir] [--show]")
        print("Example: python plot_chunking_metrics.py impgraph_16v_25c_tdlambda results/td_n")
        sys.exit(1)

    experiment_name = sys.argv[1]
    save_dir = sys.argv[2] if len(sys.argv) > 2 else 'results/td_n'
    show = '--show' in sys.argv

    # Try to load stats from pickle file, fallback to CSV
    try:
        stats = load_stats_from_pickle(save_dir, experiment_name)
        print(f"Loaded stats from pickle file")
    except FileNotFoundError:
        print(f"Warning: Could not find stats pickle file. Loading from CSV files...")
        try:
            stats = load_stats_from_csv(save_dir, experiment_name)
            print(f"Loaded stats from CSV files")
            print(f"Note: Per-step entropy plot (Figure A) requires the full stats object.")
            print(f"      Only chunking index plot (Figure B) will be generated.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Please ensure training has been run and CSV files exist in {save_dir}/metrics/")
            sys.exit(1)

    # Generate plots
    plot_chunking_metrics_from_stats(stats, experiment_name, save_dir, show)
