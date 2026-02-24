import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import os

# Set publication-quality style
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

# Parse command line arguments or use default
if len(sys.argv) > 1:
    experiment_name = sys.argv[1]
else:
    # Default experiment name
    experiment_name = 'impgraph_8v_10c'

# File paths
csv_path = f'results/off_policy/mlp_{experiment_name}_loss.csv'
output_dir = f'results/plots/{experiment_name}'
os.makedirs(output_dir, exist_ok=True)

# Check if file exists
if not os.path.exists(csv_path):
    print(f"Error: File not found: {csv_path}")
    print(f"Usage: python plot_loss.py [experiment_name]")
    print(f"Example: python plot_loss.py impgraph_8v_10c")
    sys.exit(1)

# Read the data
df = pd.read_csv(csv_path)

# Create figure with optimal size for publication (single column: 3.5", double column: 7")
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

# Plot the loss curve
epochs = df['epoch'].values
losses = df['loss'].values

# Main loss curve
ax.plot(epochs, losses, color='#2E86AB', linewidth=2.5, label='Training Loss', marker='o',
        markevery=max(1, len(epochs)//20), markersize=5, markerfacecolor='white',
        markeredgewidth=1.5, markeredgecolor='#2E86AB')

# Add grid for readability
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)  # Grid behind plot elements

# Labels and title
ax.set_xlabel('Epoch', fontweight='normal')
ax.set_ylabel('Mean Squared Error Loss', fontweight='normal')
ax.set_title('MLP Training Loss on Implication Graph Navigation',
             fontweight='bold', pad=15)

# Add statistics annotation
final_loss = losses[-1]
min_loss = losses.min()
min_epoch = epochs[losses.argmin()]

# Create text box with statistics
# stats_text = f'Final Loss: {final_loss:.4f}\nMin Loss: {min_loss:.4f} (epoch {min_epoch})'
# ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
#         verticalalignment='top', horizontalalignment='right',
#         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, edgecolor='gray', linewidth=1),
#         fontsize=9, family='monospace')

# Optional: Add horizontal line at minimum loss
ax.axhline(y=min_loss, color='#A23B72', linestyle=':', linewidth=1.5,
           alpha=0.6, label=f'Min Loss ({min_loss:.4f})')

# Legend
ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray',
          fancybox=True, shadow=False)

# Tight layout to prevent label cutoff
plt.tight_layout()

# Save in multiple formats
plt.savefig(f'{output_dir}/loss_curve.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f'{output_dir}/loss_curve.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f'{output_dir}/loss_curve.svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"\n{'='*60}")
print(f"Loss Curve Analysis for {experiment_name}")
print(f"{'='*60}")
print(f"Total Epochs:        {epochs[-1]}")
print(f"Initial Loss:        {losses[0]:.6f}")
print(f"Final Loss:          {final_loss:.6f}")
print(f"Minimum Loss:        {min_loss:.6f} (at epoch {min_epoch})")
print(f"Loss Reduction:      {(1 - final_loss/losses[0])*100:.2f}%")
print(f"{'='*60}")
print(f"\nPlots saved to:")
print(f"  - {output_dir}/loss_curve.png")
print(f"  - {output_dir}/loss_curve.pdf")
print(f"  - {output_dir}/loss_curve.svg")
print(f"{'='*60}\n")

# Show plot
plt.show()
