"""
Optimization Module

This module contains optimization algorithms including gradient descent
for training machine learning models.
"""

import autograd.numpy as np
from autograd import grad
from typing import Callable, List, Tuple, Optional


class GradientDescent:
    """
    Gradient Descent Optimizer
    
    Implements gradient descent optimization algorithm for minimizing
    cost functions in machine learning models.
    """
    
    def __init__(self):
        """Initialize the Gradient Descent optimizer."""
        self.weight_history = []
        self.cost_history = []
    
    def optimize(self, cost_function: Callable, 
                 alpha: float, 
                 max_iterations: int, 
                 w_init: np.ndarray,
                 x: np.ndarray, 
                 y: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Optimize using gradient descent.
        
        Args:
            cost_function: Cost function to minimize
            alpha: Learning rate
            max_iterations: Maximum number of iterations
            w_init: Initial weight vector
            x: Input features
            y: Target values
            
        Returns:
            Tuple of (weight_history, cost_history)
        """
        gradient = grad(cost_function)
        w = np.array(w_init, dtype=float)
        
        self.weight_history = []
        self.cost_history = []
        
        for iteration in range(max_iterations):
            # Store current state
            self.weight_history.append(w.copy())
            current_cost = cost_function(w, x, y)
            self.cost_history.append(current_cost)
            
            # Update weights
            w -= alpha * gradient(w, x, y)
        
        return self.weight_history, self.cost_history
    
    def get_best_weights(self) -> np.ndarray:
        """
        Get the weights with the lowest cost.
        
        Returns:
            Best weight vector
        """
        if not self.cost_history:
            raise ValueError("No optimization history available")
        
        best_idx = np.argmin(self.cost_history)
        return self.weight_history[best_idx]
    
    def get_convergence_info(self) -> dict:
        """
        Get information about the optimization convergence.
        
        Returns:
            Dictionary with convergence metrics
        """
        if not self.cost_history:
            return {}
        
        return {
            "initial_cost": self.cost_history[0],
            "final_cost": self.cost_history[-1],
            "cost_reduction": self.cost_history[0] - self.cost_history[-1],
            "iterations": len(self.cost_history),
            "converged": len(self.cost_history) > 1 and abs(self.cost_history[-1] - self.cost_history[-2]) < 1e-6
        }
