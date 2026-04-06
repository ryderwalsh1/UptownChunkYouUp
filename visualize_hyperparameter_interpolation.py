"""
Visualize 2D hyperparameter interpolation across lambda and junction density.

Creates heatmaps showing how lr, teacher_coef, and tau vary with lambda and junction density
using bilinear interpolation.
"""

import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import os

# Import LambdaExperiment
spec = importlib.util.spec_from_file_location('lambda_experiment_module', 'lambda_experiment.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
LambdaExperiment = module.LambdaExperiment


def create_interpolation_heatmaps(output_dir='hyperparameter_interpolation'):
    """
    Create heatmaps showing 2D interpolation of hyperparameters.

    Parameters:
    -----------
    output_dir : str
        Directory to save the figures
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    # Create dense grid for interpolation
    lambda_vals = np.linspace(0.0, 1.0, 100)
    jd_vals = np.linspace(0.1, 0.3, 100)

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

    # Plotting parameters
    cmap = 'plasma'
    empirical_data = LambdaExperiment.OPTIMAL_HYPERPARAMETERS_2D

    # Plot 1: Learning Rate
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(Lambda, JD, LR, levels=20, cmap=cmap)
    ax.contour(Lambda, JD, LR, levels=10, colors='white', alpha=0.3, linewidths=0.5)

    # Mark empirical data points
    for jd in empirical_data.keys():
        for lam in empirical_data[jd].keys():
            ax.plot(lam, jd, 'ko', markersize=6, markeredgewidth=1.5,
                   markerfacecolor='white', markeredgecolor='black')

    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Junction Density', fontsize=12)
    ax.set_title('Learning Rate', fontsize=13, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Learning Rate', fontsize=10)
    ax.grid(alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'learning_rate.pdf'), bbox_inches='tight')
    print(f"Saved learning rate to: {output_dir}/learning_rate.png")
    plt.close()

    # Plot 2: Teacher Coefficient
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(Lambda, JD, TeacherCoef, levels=20, cmap=cmap)
    ax.contour(Lambda, JD, TeacherCoef, levels=10, colors='white', alpha=0.3, linewidths=0.5)

    # Mark empirical data points
    for jd in empirical_data.keys():
        for lam in empirical_data[jd].keys():
            ax.plot(lam, jd, 'ko', markersize=6, markeredgewidth=1.5,
                   markerfacecolor='white', markeredgecolor='black')

    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Junction Density', fontsize=12)
    ax.set_title('Teacher Coefficient', fontsize=13, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Teacher Coefficient', fontsize=10)
    ax.grid(alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'teacher_coefficient.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'teacher_coefficient.pdf'), bbox_inches='tight')
    print(f"Saved teacher coefficient to: {output_dir}/teacher_coefficient.png")
    plt.close()

    # Plot 3: Tau
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(Lambda, JD, Tau, levels=20, cmap=cmap)
    ax.contour(Lambda, JD, Tau, levels=10, colors='white', alpha=0.3, linewidths=0.5)

    # Mark empirical data points
    for jd in empirical_data.keys():
        for lam in empirical_data[jd].keys():
            ax.plot(lam, jd, 'ko', markersize=6, markeredgewidth=1.5,
                   markerfacecolor='white', markeredgecolor='black')

    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Junction Density', fontsize=12)
    ax.set_title('Tau (τ)', fontsize=13, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Tau', fontsize=10)
    ax.grid(alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tau.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'tau.pdf'), bbox_inches='tight')
    print(f"Saved tau to: {output_dir}/tau.png")
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
