#!/usr/bin/env python3
"""
Quick test script to verify run_lambda_experiment.py parallelization works.
Tests with a minimal configuration.

IMPORTANT: This script MUST be run with if __name__ == '__main__' protection
because macOS/Windows use 'spawn' for multiprocessing, which re-imports the module.
"""

if __name__ == '__main__':
    from run_lambda_experiment import LambdaExperiment

    # Minimal test configuration
    config = {
        'lambda_values': [0.0, 0.5, 1.0],  # Just 3 lambda values
        'seeds': [60, 61],  # Just 2 seeds
        'num_episodes': 100,  # Very short training
        'topologies': [
            '0.5 corridor',  # Just one topology
        ],
        'lr': 3e-4,  # Fixed learning rate (not adaptive)
        'gamma': 0.99,
        'entropy_coef': 0.01,
        'teacher_coef': 10.0,
        'value_coef': 0.5,
        'tau': 0.6,
        'consultation_temperature': 0.5,
        'hard_teacher_force': False,
        'memory_consultation_cost': 0.0,
        'memory_correction_cost': 0.0,
        'output_dir': 'test_parallel_results',
    }

    print("Testing parallelized run_lambda_experiment.py...")
    print(f"This will run {len(config['topologies']) * len(config['lambda_values']) * len(config['seeds'])} configurations in parallel")
    print(f"Each configuration will train for {config['num_episodes']} episodes")
    print()

    # Create and run experiment
    experiment = LambdaExperiment(config)
    experiment.run()

    print("\nTest complete! Check test_parallel_results/ for outputs.")
