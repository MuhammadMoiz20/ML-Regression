"""
Metrics and Regularization Module

This module contains evaluation metrics and regularization functions
for machine learning models.
"""

import autograd.numpy as np
from typing import Union


class MSE:
    """
    Mean Squared Error metric for regression evaluation.
    """
    
    @staticmethod
    def __call__(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Error.
        
        Args:
            y_actual: Actual target values
            y_pred: Predicted values
            
        Returns:
            Mean squared error
        """
        return np.mean((y_actual - y_pred) ** 2)
    
    def __str__(self) -> str:
        return "Mean Squared Error"


class MAD:
    """
    Mean Absolute Deviation metric for regression evaluation.
    """
    
    @staticmethod
    def __call__(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Deviation.
        
        Args:
            y_actual: Actual target values
            y_pred: Predicted values
            
        Returns:
            Mean absolute deviation
        """
        return np.mean(np.abs(y_actual - y_pred))
    
    def __str__(self) -> str:
        return "Mean Absolute Deviation"


class L1Regularizer:
    """
    L1 Regularization (Lasso) penalty function.
    """
    
    @staticmethod
    def __call__(w: np.ndarray) -> float:
        """
        Calculate L1 regularization penalty.
        
        Args:
            w: Weight vector
            
        Returns:
            L1 penalty (sum of absolute values)
        """
        return np.sum(np.abs(w))
    
    def __str__(self) -> str:
        return "L1 Regularization"


class L2Regularizer:
    """
    L2 Regularization (Ridge) penalty function.
    """
    
    @staticmethod
    def __call__(w: np.ndarray) -> float:
        """
        Calculate L2 regularization penalty.
        
        Args:
            w: Weight vector
            
        Returns:
            L2 penalty (sum of squared values)
        """
        return np.sum(w ** 2)
    
    def __str__(self) -> str:
        return "L2 Regularization"


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.metrics = {
            'mse': MSE(),
            'mad': MAD()
        }
    
    def evaluate(self, y_actual: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Evaluate model predictions using multiple metrics.
        
        Args:
            y_actual: Actual target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metric scores
        """
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(y_actual, y_pred)
        return results
    
    def compare_models(self, models: dict, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Compare multiple models on test data.
        
        Args:
            models: Dictionary of model names and fitted models
            x_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation results for each model
        """
        results = {}
        for name, model in models.items():
            y_pred = model.predict(x_test)
            results[name] = self.evaluate(y_test, y_pred)
        return results
