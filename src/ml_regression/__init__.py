"""
Machine Learning Regression Package

A comprehensive Python package implementing various regression algorithms
including linear regression, ridge regression, lasso regression, and
gradient descent optimization methods.

This package demonstrates fundamental machine learning concepts including:
- Linear regression models
- Gradient descent optimization
- Regularization techniques (L1/L2)
- Model evaluation metrics
- Robust regression methods
"""

__version__ = "1.0.0"
__author__ = "Muhammad Moiz"
__email__ = "moiz@example.com"

from .models import LinearRegression, RidgeRegression, LassoRegression
from .optimizers import GradientDescent
from .metrics import MSE, MAD, L1Regularizer, L2Regularizer
from .utils import load_data, load_weather_data, visualize_scatter, visualize_cost_history, visualize_regression_line, visualize_model_comparison, print_model_performance

__all__ = [
    "LinearRegression",
    "RidgeRegression", 
    "LassoRegression",
    "GradientDescent",
    "MSE",
    "MAD",
    "L1Regularizer",
    "L2Regularizer",
    "load_data",
    "load_weather_data",
    "visualize_scatter",
    "visualize_cost_history",
    "visualize_regression_line",
    "visualize_model_comparison",
    "print_model_performance"
]
