#!/usr/bin/env python3
"""
Basic Linear Regression Example

This example demonstrates how to use the ml_regression package
to perform linear regression on sample data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from ml_regression import LinearRegression, load_data, visualize_scatter, visualize_cost_history, visualize_regression_line


def main():
    """Run basic linear regression example."""
    print("Linear Regression Example")
    print("=" * 40)
    
    # Load sample data
    try:
        x, y = load_data('data/kleibers_law_data.csv')
        print(f"Loaded data with {x.shape[1]} samples")
    except FileNotFoundError:
        print("Creating sample data...")
        # Create sample data if file doesn't exist
        np.random.seed(42)
        x = np.random.uniform(0, 10, (1, 50))
        y = 2 * x + 1 + np.random.normal(0, 0.5, x.shape)
    
    # Visualize the data
    print("\nVisualizing data...")
    visualize_scatter(x, y, "Sample Data", "x", "y")
    
    # Create and train linear regression model
    print("\nTraining linear regression model...")
    model = LinearRegression(cost_function="least_squares")
    
    # Train the model
    weight_history, cost_history = model.fit(
        x.flatten(), 
        y.flatten(),
        alpha=0.01,
        max_iterations=1000
    )
    
    # Visualize cost history
    print("Plotting cost history...")
    visualize_cost_history(cost_history, "Cost History Over Iterations")
    
    # Visualize fitted line
    print("Plotting fitted regression line...")
    visualize_regression_line(
        x, y, model.weights,
        "Linear Regression Fit",
        "x", "y"
    )
    
    # Print results
    print(f"\nModel Results:")
    print(f"Final weights: {model.weights}")
    print(f"Final cost: {cost_history[-1]:.6f}")
    print(f"Cost reduction: {cost_history[0] - cost_history[-1]:.6f}")
    
    # Make predictions
    x_test = np.array([[5.0], [7.5], [10.0]])
    y_pred = model.predict(x_test.flatten())
    
    print(f"\nPredictions:")
    for i, (x_val, y_val) in enumerate(zip(x_test.flatten(), y_pred)):
        print(f"x={x_val:.1f} -> y={y_val:.3f}")


if __name__ == "__main__":
    main()
