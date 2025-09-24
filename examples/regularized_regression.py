#!/usr/bin/env python3
"""
Regularized Regression Example

This example demonstrates Ridge and Lasso regression
with different regularization parameters.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from ml_regression import (
    RidgeRegression, 
    LassoRegression,
    load_weather_data,
    print_model_performance
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SklearnLinearRegression, Ridge as SklearnRidge, Lasso as SklearnLasso
from sklearn.metrics import mean_squared_error


def main():
    """Run regularized regression example."""
    print("Regularized Regression Example")
    print("=" * 40)
    
    # Load weather data
    try:
        X, y = load_weather_data('data/weatherHistory.csv')
        print(f"Loaded weather data with {X.shape[0]} samples and {X.shape[1]} features")
    except FileNotFoundError:
        print("Creating sample multivariate data...")
        # Create sample multivariate data
        np.random.seed(42)
        n_samples, n_features = 100, 6
        X = np.random.randn(n_samples, n_features)
        true_weights = np.array([0.5, -0.3, 0.8, 0.1, -0.2, 0.4])
        y = X @ true_weights + np.random.normal(0, 0.1, n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models using sklearn for comparison
    print("\nTraining models...")
    
    # Sklearn models
    sklearn_models = {
        'Sklearn Linear': SklearnLinearRegression(),
        'Sklearn Ridge': SklearnRidge(alpha=1.0),
        'Sklearn Lasso': SklearnLasso(alpha=1.0)
    }
    
    for name, model in sklearn_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{name} MSE: {mse:.4f}")
        
        if hasattr(model, 'coef_'):
            print(f"{name} coefficients: {model.coef_}")
    
    # Compare regularization effects
    print("\nComparing regularization parameters...")
    
    lambda_values = [0.01, 0.1, 1.0, 10.0]
    
    plt.figure(figsize=(15, 5))
    
    # Ridge regression
    plt.subplot(1, 3, 1)
    ridge_mse = []
    for lmbda in lambda_values:
        ridge_model = SklearnRidge(alpha=lmbda)
        ridge_model.fit(X_train, y_train)
        y_pred = ridge_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        ridge_mse.append(mse)
    
    plt.plot(lambda_values, ridge_mse, 'b-o')
    plt.xlabel('Lambda (α)')
    plt.ylabel('MSE')
    plt.title('Ridge Regression')
    plt.xscale('log')
    plt.grid(True)
    
    # Lasso regression
    plt.subplot(1, 3, 2)
    lasso_mse = []
    for lmbda in lambda_values:
        lasso_model = SklearnLasso(alpha=lmbda)
        lasso_model.fit(X_train, y_train)
        y_pred = lasso_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        lasso_mse.append(mse)
    
    plt.plot(lambda_values, lasso_mse, 'r-o')
    plt.xlabel('Lambda (α)')
    plt.ylabel('MSE')
    plt.title('Lasso Regression')
    plt.xscale('log')
    plt.grid(True)
    
    # Coefficient comparison
    plt.subplot(1, 3, 3)
    feature_names = ['Apparent_Temp', 'Humidity', 'Wind_Speed', 'Wind_Bearing', 'Visibility', 'Pressure']
    
    # Get coefficients for different lambda values
    ridge_coefs = []
    lasso_coefs = []
    
    for lmbda in lambda_values:
        ridge_model = SklearnRidge(alpha=lmbda)
        ridge_model.fit(X_train, y_train)
        ridge_coefs.append(ridge_model.coef_)
        
        lasso_model = SklearnLasso(alpha=lmbda)
        lasso_model.fit(X_train, y_train)
        lasso_coefs.append(lasso_model.coef_)
    
    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)
    
    for i in range(len(feature_names)):
        plt.plot(lambda_values, ridge_coefs[:, i], 'b-', alpha=0.7, label='Ridge' if i == 0 else "")
        plt.plot(lambda_values, lasso_coefs[:, i], 'r-', alpha=0.7, label='Lasso' if i == 0 else "")
    
    plt.xlabel('Lambda (α)')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficient Shrinkage')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nAnalysis:")
    print("- Ridge regression shrinks coefficients but doesn't eliminate them")
    print("- Lasso regression can eliminate features (coefficients = 0)")
    print("- Higher lambda values lead to more regularization")
    print("- Choose Ridge for keeping all features, Lasso for feature selection")


if __name__ == "__main__":
    main()
