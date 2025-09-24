#!/usr/bin/env python3
"""
Robust Regression Comparison Example

This example demonstrates the difference between least squares
and least absolute deviations regression on data with outliers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from ml_regression import (
    LinearRegression, 
    load_data, 
    visualize_model_comparison,
    print_model_performance,
    MSE,
    MAD
)


def main():
    """Run robust regression comparison example."""
    print("Robust Regression Comparison")
    print("=" * 40)
    
    # Load data with outliers
    try:
        x, y = load_data('data/regression_outliers.csv')
        print(f"Loaded outlier data with {x.shape[1]} samples")
    except FileNotFoundError:
        print("Creating sample data with outliers...")
        # Create sample data with outliers
        np.random.seed(42)
        x = np.random.uniform(0, 10, (1, 20))
        y = 2 * x + 1 + np.random.normal(0, 0.5, x.shape)
        
        # Add outliers
        x_outliers = np.array([[15.0, 20.0]])
        y_outliers = np.array([[5.0, 3.0]])
        x = np.concatenate([x, x_outliers], axis=1)
        y = np.concatenate([y, y_outliers], axis=1)
    
    # Train two models
    print("\nTraining models...")
    
    # Least Squares model
    ls_model = LinearRegression(cost_function="least_squares")
    ls_weight_history, ls_cost_history = ls_model.fit(
        x.flatten(), y.flatten(),
        w_init=np.array([1.0, 1.0]),
        alpha=0.1,
        max_iterations=100
    )
    
    # Least Absolute Deviations model
    lad_model = LinearRegression(cost_function="least_absolute_deviations")
    lad_weight_history, lad_cost_history = lad_model.fit(
        x.flatten(), y.flatten(),
        w_init=np.array([1.0, 1.0]),
        alpha=0.1,
        max_iterations=100
    )
    
    # Compare cost histories
    plt.figure(figsize=(10, 6))
    plt.plot(ls_cost_history, 'b-', label='Least Squares')
    plt.plot(lad_cost_history, 'r-', label='Least Absolute Deviations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost History Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Compare fitted lines
    print("\nComparing fitted regression lines...")
    models = {
        'Least Squares': ls_model,
        'Least Absolute Deviations': lad_model
    }
    visualize_model_comparison(
        x, y, models,
        "Regression Models Comparison",
        "x", "y"
    )
    
    # Evaluate performance
    print("\nModel Performance Evaluation:")
    print_model_performance(models, x.flatten(), y.flatten())
    
    # Print model weights
    print(f"\nModel Weights:")
    print(f"Least Squares: {ls_model.weights}")
    print(f"Least Absolute Deviations: {lad_model.weights}")
    
    print("\nAnalysis:")
    print("- Least Absolute Deviations is more robust to outliers")
    print("- Least Squares is more sensitive to extreme values")
    print("- Choose LAD when data contains outliers")


if __name__ == "__main__":
    main()
