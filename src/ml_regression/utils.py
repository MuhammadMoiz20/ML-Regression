"""
Utility Functions Module

This module contains utility functions for data loading, visualization,
and other helper functions.
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import os


def load_data(filename: str, delimiter: str = ',') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV file.
    
    Args:
        filename: Path to CSV file
        delimiter: CSV delimiter
        
    Returns:
        Tuple of (features, targets)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file {filename} not found")
    
    data = np.loadtxt(filename, delimiter=delimiter)
    x = data[:-1, :]
    y = data[-1:, :]
    
    return x, y


def load_weather_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load weather dataset with proper column handling.
    
    Args:
        filename: Path to weather CSV file
        
    Returns:
        Tuple of (features, targets)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Weather data file {filename} not found")
    
    weather_data = np.genfromtxt(filename, delimiter=',', names=True, dtype=None, encoding='utf-8')
    
    # Extract temperature as target
    y = weather_data['Temperature']
    
    # Extract features: Apparent_Temperature, Humidity, Wind_Speed, Wind_Bearing, Visibility, Pressure
    X = np.column_stack((
        weather_data['Apparent_Temperature'],
        weather_data['Humidity'],
        weather_data['Wind_Speed'],
        weather_data['Wind_Bearing'],
        weather_data['Visibility'],
        weather_data['Pressure']
    ))
    
    return X, y


def visualize_scatter(x: np.ndarray, y: np.ndarray, 
                    title: str = "Scatter Plot", 
                    xlabel: str = "x", 
                    ylabel: str = "y",
                    figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Create a scatter plot of the data.
    
    Args:
        x: X-axis data
        y: Y-axis data
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.scatter(x.flatten(), y.flatten(), alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()


def visualize_cost_history(cost_history: List[float], 
                          title: str = "Cost History",
                          figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot cost history over iterations.
    
    Args:
        cost_history: List of cost values
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(title)
    plt.grid(True)
    plt.show()


def visualize_regression_line(x: np.ndarray, y: np.ndarray, 
                            weights: np.ndarray,
                            title: str = "Regression Line",
                            xlabel: str = "x",
                            ylabel: str = "y",
                            figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot data points with fitted regression line.
    
    Args:
        x: Input features
        y: Target values
        weights: Fitted model weights
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
    """
    from .models import LinearRegression
    
    # Generate line points
    x_line = np.linspace(np.min(x), np.max(x), 100)
    linear_model = LinearRegression()
    y_line = linear_model.model(x_line, weights)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.scatter(x.flatten(), y.flatten(), alpha=0.5, label='Data')
    plt.plot(x_line, y_line, 'r-', label='Fitted Line')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_model_comparison(x: np.ndarray, y: np.ndarray,
                             models: dict,
                             title: str = "Model Comparison",
                             xlabel: str = "x",
                             ylabel: str = "y",
                             figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Compare multiple models on the same plot.
    
    Args:
        x: Input features
        y: Target values
        models: Dictionary of model names and fitted models
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.scatter(x.flatten(), y.flatten(), alpha=0.5, label='Data')
    
    x_line = np.linspace(np.min(x), np.max(x), 100)
    colors = ['b-', 'r-', 'g-', 'm-', 'c-']
    
    for i, (name, model) in enumerate(models.items()):
        if hasattr(model, 'weights') and model.weights is not None:
            from .models import LinearRegression
            linear_model = LinearRegression()
            y_line = linear_model.model(x_line, model.weights)
            plt.plot(x_line, y_line, colors[i % len(colors)], label=name)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def print_model_performance(models: dict, x_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Print performance metrics for multiple models.
    
    Args:
        models: Dictionary of model names and fitted models
        x_test: Test features
        y_test: Test targets
    """
    from .metrics import ModelEvaluator
    
    evaluator = ModelEvaluator()
    results = evaluator.compare_models(models, x_test, y_test)
    
    print("\nModel Performance Comparison:")
    print("=" * 50)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name.upper()}: {value:.6f}")


def create_demo_plots():
    """
    Create demonstration plots showcasing the package capabilities.
    """
    # This would be used to generate plots for the README
    pass
