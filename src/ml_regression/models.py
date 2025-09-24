"""
Regression Models Module

This module contains implementations of various regression models including
linear regression, ridge regression, and lasso regression.
"""

import autograd.numpy as np
from typing import Tuple, List, Optional


class LinearRegression:
    """
    Linear Regression Model
    
    Implements a simple linear regression model using gradient descent optimization.
    Supports both least squares and least absolute deviations cost functions.
    """
    
    def __init__(self, cost_function: str = "least_squares"):
        """
        Initialize the Linear Regression model.
        
        Args:
            cost_function: Type of cost function to use ("least_squares" or "least_absolute_deviations")
        """
        self.cost_function = cost_function
        self.weights = None
        self.weight_history = []
        self.cost_history = []
        
    def model(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Linear model function: y = w[0] + w[1] * x
        
        Args:
            x: Input features
            w: Weight vector [bias, slope]
            
        Returns:
            Predicted values
        """
        return w[0] + w[1] * x
    
    def least_squares(self, w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """
        Least squares cost function.
        
        Args:
            w: Weight vector
            x: Input features
            y: Target values
            
        Returns:
            Mean squared error
        """
        y_pred = self.model(x, w)
        return np.mean((y - y_pred) ** 2)
    
    def least_absolute_deviations(self, w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """
        Least absolute deviations cost function.
        
        Args:
            w: Weight vector
            x: Input features
            y: Target values
            
        Returns:
            Sum of absolute errors
        """
        y_pred = self.model(x, w)
        return np.sum(np.abs(y - y_pred))
    
    def fit(self, x: np.ndarray, y: np.ndarray, 
            w_init: Optional[np.ndarray] = None,
            alpha: float = 0.01,
            max_iterations: int = 1000) -> Tuple[List[np.ndarray], List[float]]:
        """
        Fit the linear regression model using gradient descent.
        
        Args:
            x: Training features
            y: Training targets
            w_init: Initial weights (if None, random initialization)
            alpha: Learning rate
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (weight_history, cost_history)
        """
        if w_init is None:
            np.random.seed(42)
            w_init = np.random.uniform(-0.1, 0.1, 2)
        
        # Select cost function
        if self.cost_function == "least_squares":
            cost_func = self.least_squares
        else:
            cost_func = self.least_absolute_deviations
        
        # Gradient descent optimization
        from .optimizers import GradientDescent
        optimizer = GradientDescent()
        self.weight_history, self.cost_history = optimizer.optimize(
            cost_func, alpha, max_iterations, w_init, x, y
        )
        
        # Store best weights
        best_idx = np.argmin(self.cost_history)
        self.weights = self.weight_history[best_idx]
        
        return self.weight_history, self.cost_history
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            x: Input features
            
        Returns:
            Predicted values
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.model(x, self.weights)


class RidgeRegression:
    """
    Ridge Regression Model
    
    Implements ridge regression with L2 regularization.
    """
    
    def __init__(self, lambda_reg: float = 1.0):
        """
        Initialize Ridge Regression model.
        
        Args:
            lambda_reg: Regularization parameter (lambda)
        """
        self.lambda_reg = lambda_reg
        self.weights = None
        
    def ridge_cost(self, w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """
        Ridge regression cost function with L2 regularization.
        
        Args:
            w: Weight vector
            x: Input features
            y: Target values
            
        Returns:
            Ridge cost (MSE + L2 regularization)
        """
        from .metrics import L2Regularizer
        from .models import LinearRegression
        
        linear_model = LinearRegression()
        mse = linear_model.least_squares(w, x, y)
        l2_penalty = L2Regularizer()(w)
        
        return mse + self.lambda_reg * l2_penalty
    
    def fit(self, x: np.ndarray, y: np.ndarray, 
            w_init: Optional[np.ndarray] = None,
            alpha: float = 0.01,
            max_iterations: int = 1000) -> Tuple[List[np.ndarray], List[float]]:
        """
        Fit the ridge regression model.
        
        Args:
            x: Training features
            y: Training targets
            w_init: Initial weights
            alpha: Learning rate
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (weight_history, cost_history)
        """
        if w_init is None:
            np.random.seed(42)
            w_init = np.random.uniform(-0.1, 0.1, 2)
        
        from .optimizers import GradientDescent
        optimizer = GradientDescent()
        weight_history, cost_history = optimizer.optimize(
            self.ridge_cost, alpha, max_iterations, w_init, x, y
        )
        
        # Store best weights
        best_idx = np.argmin(cost_history)
        self.weights = weight_history[best_idx]
        
        return weight_history, cost_history
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            x: Input features
            
        Returns:
            Predicted values
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        linear_model = LinearRegression()
        return linear_model.model(x, self.weights)


class LassoRegression:
    """
    Lasso Regression Model
    
    Implements lasso regression with L1 regularization.
    """
    
    def __init__(self, lambda_reg: float = 1.0):
        """
        Initialize Lasso Regression model.
        
        Args:
            lambda_reg: Regularization parameter (lambda)
        """
        self.lambda_reg = lambda_reg
        self.weights = None
        
    def lasso_cost(self, w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """
        Lasso regression cost function with L1 regularization.
        
        Args:
            w: Weight vector
            x: Input features
            y: Target values
            
        Returns:
            Lasso cost (MSE + L1 regularization)
        """
        from .metrics import L1Regularizer
        from .models import LinearRegression
        
        linear_model = LinearRegression()
        mse = linear_model.least_squares(w, x, y)
        l1_penalty = L1Regularizer()(w)
        
        return mse + self.lambda_reg * l1_penalty
    
    def fit(self, x: np.ndarray, y: np.ndarray, 
            w_init: Optional[np.ndarray] = None,
            alpha: float = 0.01,
            max_iterations: int = 1000) -> Tuple[List[np.ndarray], List[float]]:
        """
        Fit the lasso regression model.
        
        Args:
            x: Training features
            y: Training targets
            w_init: Initial weights
            alpha: Learning rate
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (weight_history, cost_history)
        """
        if w_init is None:
            np.random.seed(42)
            w_init = np.random.uniform(-0.1, 0.1, 2)
        
        from .optimizers import GradientDescent
        optimizer = GradientDescent()
        weight_history, cost_history = optimizer.optimize(
            self.lasso_cost, alpha, max_iterations, w_init, x, y
        )
        
        # Store best weights
        best_idx = np.argmin(cost_history)
        self.weights = weight_history[best_idx]
        
        return weight_history, cost_history
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            x: Input features
            
        Returns:
            Predicted values
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        linear_model = LinearRegression()
        return linear_model.model(x, self.weights)
