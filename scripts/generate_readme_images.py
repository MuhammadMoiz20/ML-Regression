#!/usr/bin/env python3
"""
Generate visualizations for README

This script creates the visualizations from the original notebook
and saves them as images for the README.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ml_regression import (
    LinearRegression, 
    load_data, 
    visualize_scatter, 
    visualize_cost_history,
    visualize_regression_line,
    visualize_model_comparison
)


def create_kleibers_law_plot():
    """Create Kleiber's Law scatter plot."""
    # Load data
    x, y = load_data('data/kleibers_law_data.csv')
    x_log = np.log(x)
    y_log = np.log(y)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_log.flatten(), y_log.flatten(), alpha=0.6, color='blue')
    plt.xlabel('log(x)')
    plt.ylabel('log(y)')
    plt.title("Scatter Plot of Kleiber's Law Data")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/kleibers_law_data.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_cost_history_plot():
    """Create cost history plot."""
    # Create sample data
    np.random.seed(42)
    x = np.random.uniform(0, 10, 50)
    y = 2 * x + 1 + np.random.normal(0, 0.5, 50)
    
    # Train model
    model = LinearRegression(cost_function="least_squares")
    weight_history, cost_history = model.fit(x, y, alpha=0.01, max_iterations=1000)
    
    plt.figure(figsize=(8, 6))
    plt.plot(cost_history, color='blue', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost History Over Iterations')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/cost_history.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_regression_line_plot():
    """Create regression line plot."""
    # Create sample data
    np.random.seed(42)
    x = np.random.uniform(0, 10, 50)
    y = 2 * x + 1 + np.random.normal(0, 0.5, 50)
    
    # Train model
    model = LinearRegression(cost_function="least_squares")
    weight_history, cost_history = model.fit(x, y, alpha=0.01, max_iterations=1000)
    
    # Generate line points
    x_line = np.linspace(np.min(x), np.max(x), 100)
    y_line = model.predict(x_line)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, color='blue', label='Data Points')
    plt.plot(x_line, y_line, 'r-', linewidth=2, label='Fitted Line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/regression_line.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_outliers_plot():
    """Create outliers scatter plot."""
    # Create data with outliers
    np.random.seed(42)
    x = np.random.uniform(0, 10, 20)
    y = 2 * x + 1 + np.random.normal(0, 0.3, 20)
    
    # Add outliers
    x_outliers = np.array([15.0, 20.0])
    y_outliers = np.array([5.0, 3.0])
    x = np.concatenate([x, x_outliers])
    y = np.concatenate([y, y_outliers])
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='green', alpha=0.7, s=50)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data with Outliers')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/outliers_data.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_model_comparison_plot():
    """Create model comparison plot."""
    # Create data with outliers
    np.random.seed(42)
    x = np.random.uniform(0, 10, 20)
    y = 2 * x + 1 + np.random.normal(0, 0.3, 20)
    
    # Add outliers
    x_outliers = np.array([15.0, 20.0])
    y_outliers = np.array([5.0, 3.0])
    x = np.concatenate([x, x_outliers])
    y = np.concatenate([y, y_outliers])
    
    # Train both models
    ls_model = LinearRegression(cost_function="least_squares")
    lad_model = LinearRegression(cost_function="least_absolute_deviations")
    
    ls_model.fit(x, y, w_init=np.array([1.0, 1.0]), alpha=0.1, max_iterations=100)
    lad_model.fit(x, y, w_init=np.array([1.0, 1.0]), alpha=0.1, max_iterations=100)
    
    # Generate line points
    x_line = np.linspace(np.min(x), np.max(x), 100)
    ls_y_line = ls_model.predict(x_line)
    lad_y_line = lad_model.predict(x_line)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, color='gray', label='Data Points')
    plt.plot(x_line, ls_y_line, 'b-', linewidth=2, label='Least Squares')
    plt.plot(x_line, lad_y_line, 'r-', linewidth=2, label='Least Absolute Deviations')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Robust Regression Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_cost_comparison_plot():
    """Create cost history comparison plot."""
    # Create data with outliers
    np.random.seed(42)
    x = np.random.uniform(0, 10, 20)
    y = 2 * x + 1 + np.random.normal(0, 0.3, 20)
    
    # Add outliers
    x_outliers = np.array([15.0, 20.0])
    y_outliers = np.array([5.0, 3.0])
    x = np.concatenate([x, x_outliers])
    y = np.concatenate([y, y_outliers])
    
    # Train both models
    ls_model = LinearRegression(cost_function="least_squares")
    lad_model = LinearRegression(cost_function="least_absolute_deviations")
    
    ls_weight_history, ls_cost_history = ls_model.fit(
        x, y, w_init=np.array([1.0, 1.0]), alpha=0.1, max_iterations=100
    )
    lad_weight_history, lad_cost_history = lad_model.fit(
        x, y, w_init=np.array([1.0, 1.0]), alpha=0.1, max_iterations=100
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(ls_cost_history, 'b-', linewidth=2, label='Least Squares')
    plt.plot(lad_cost_history, 'r-', linewidth=2, label='Least Absolute Deviations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost History Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/cost_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Generate all visualizations."""
    print("Generating visualizations for README...")
    
    # Create docs directory if it doesn't exist
    os.makedirs('docs', exist_ok=True)
    
    # Generate all plots
    create_kleibers_law_plot()
    print("✓ Created Kleiber's Law data plot")
    
    create_cost_history_plot()
    print("✓ Created cost history plot")
    
    create_regression_line_plot()
    print("✓ Created regression line plot")
    
    create_outliers_plot()
    print("✓ Created outliers data plot")
    
    create_model_comparison_plot()
    print("✓ Created model comparison plot")
    
    create_cost_comparison_plot()
    print("✓ Created cost comparison plot")
    
    print("\nAll visualizations generated successfully!")
    print("Images saved in docs/ directory")


if __name__ == "__main__":
    main()
