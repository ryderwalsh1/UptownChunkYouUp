"""
Visualize 2D hyperparameter interpolation across lambda and junction density.

Creates heatmaps showing how lr, teacher_coef, and tau vary with lambda and junction density
using bilinear interpolation.
"""

import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# Import LambdaExperiment
spec = importlib.util.spec_from_file_location('lambda_experiment_module', 'lambda_experiment.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
LambdaExperiment = module.LambdaExperiment


def create_interpolation_heatmaps(output_path='hyperparameter_interpolation.png'):
    """
    Create heatmaps showing 2D interpolation of hyperparameters.

    Parameters:
    -----------
    output_path : str
        Path to save the figure
    """
    # Create dense grid for interpolation
    lambda_vals = np.linspace(0.0, 1.0, 100)
    jd_vals = np.linspace(0.15, 0.34, 100)

    # Create meshgrid
    Lambda, JD = np.meshgrid(lambda_vals, jd_vals)

    # Initialize arrays for each parameter
    LR = np.zeros_like(Lambda)
    TeacherCoef = np.zeros_like(Lambda)
    Tau = np.zeros_like(Lambda)

    # Compute interpolated values at each grid point
    print("Computing interpolated hyperparameters...")
    for i in range(len(jd_vals)):
        for j in range(len(lambda_vals)):
            params = LambdaExperiment.get_optimal_hyperparameters_2d(
                lambda_vals[j], jd_vals[i]
            )
            LR[i, j] = params['lr']
            TeacherCoef[i, j] = params['teacher_coef']
            Tau[i, j] = params['tau']

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plotting parameters
    cmap = 'plasma'

    # Plot 1: Learning Rate
    ax = axes[0]
    im1 = ax.contourf(Lambda, JD, LR, levels=20, cmap=cmap)
    ax.contour(Lambda, JD, LR, levels=10, colors='white', alpha=0.3, linewidths=0.5)

    # Mark empirical data points
    empirical_data = LambdaExperiment.OPTIMAL_HYPERPARAMETERS_2D
    for jd in empirical_data.keys():
        for lam in empirical_data[jd].keys():
            ax.plot(lam, jd, 'ko', markersize=6, markeredgewidth=1.5,
                   markerfacecolor='white', markeredgecolor='black')

    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Junction Density', fontsize=12)
    ax.set_title('Learning Rate', fontsize=13, fontweight='bold')
    cbar1 = fig.colorbar(im1, ax=ax)
    cbar1.set_label('Learning Rate', fontsize=10)
    ax.grid(alpha=0.2, linewidth=0.5)

    # Plot 2: Teacher Coefficient
    ax = axes[1]
    im2 = ax.contourf(Lambda, JD, TeacherCoef, levels=20, cmap=cmap)
    ax.contour(Lambda, JD, TeacherCoef, levels=10, colors='white', alpha=0.3, linewidths=0.5)

    # Mark empirical data points
    for jd in empirical_data.keys():
        for lam in empirical_data[jd].keys():
            ax.plot(lam, jd, 'ko', markersize=6, markeredgewidth=1.5,
                   markerfacecolor='white', markeredgecolor='black')

    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Junction Density', fontsize=12)
    ax.set_title('Teacher Coefficient', fontsize=13, fontweight='bold')
    cbar2 = fig.colorbar(im2, ax=ax)
    cbar2.set_label('Teacher Coefficient', fontsize=10)
    ax.grid(alpha=0.2, linewidth=0.5)

    # Plot 3: Tau
    ax = axes[2]
    im3 = ax.contourf(Lambda, JD, Tau, levels=20, cmap=cmap)
    ax.contour(Lambda, JD, Tau, levels=10, colors='white', alpha=0.3, linewidths=0.5)

    # Mark empirical data points
    for jd in empirical_data.keys():
        for lam in empirical_data[jd].keys():
            ax.plot(lam, jd, 'ko', markersize=6, markeredgewidth=1.5,
                   markerfacecolor='white', markeredgecolor='black')

    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Junction Density', fontsize=12)
    ax.set_title('Tau (τ)', fontsize=13, fontweight='bold')
    cbar3 = fig.colorbar(im3, ax=ax)
    cbar3.set_label('Tau', fontsize=10)
    ax.grid(alpha=0.2, linewidth=0.5)

    # Overall title
    fig.suptitle('2D Hyperparameter Interpolation: Lambda × Junction Density',
                 fontsize=15, fontweight='bold', y=1.02)

    # Add text annotation
    fig.text(0.5, -0.02,
             'Black-white circles indicate empirical measurement points',
             ha='center', fontsize=9, style='italic')

    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")

    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved figure to: {pdf_path}")

    plt.close()


if __name__ == '__main__':
    # Set publication-quality plotting parameters
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9

    create_interpolation_heatmaps()
    print("\nVisualization complete!")
