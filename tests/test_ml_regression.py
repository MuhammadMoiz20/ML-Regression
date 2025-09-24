#!/usr/bin/env python3
"""
Test suite for ml_regression package.

This module contains comprehensive unit tests for all components
of the ml_regression package.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_regression.models import LinearRegression, RidgeRegression, LassoRegression
from ml_regression.optimizers import GradientDescent
from ml_regression.metrics import MSE, MAD, L1Regularizer, L2Regularizer, ModelEvaluator
from ml_regression.utils import load_data, load_weather_data


class TestLinearRegression:
    """Test cases for LinearRegression class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = LinearRegression()
        np.random.seed(42)
        self.x = np.random.uniform(0, 10, 50)
        self.y = 2 * self.x + 1 + np.random.normal(0, 0.1, 50)
    
    def test_model_function(self):
        """Test the linear model function."""
        w = np.array([1.0, 2.0])
        x_test = np.array([1.0, 2.0, 3.0])
        y_pred = self.model.model(x_test, w)
        
        expected = np.array([3.0, 5.0, 7.0])
        np.testing.assert_array_almost_equal(y_pred, expected)
    
    def test_least_squares_cost(self):
        """Test least squares cost function."""
        w = np.array([1.0, 2.0])
        cost = self.model.least_squares(w, self.x, self.y)
        
        assert isinstance(cost, float)
        assert cost >= 0
    
    def test_least_absolute_deviations_cost(self):
        """Test least absolute deviations cost function."""
        w = np.array([1.0, 2.0])
        cost = self.model.least_absolute_deviations(w, self.x, self.y)
        
        assert isinstance(cost, float)
        assert cost >= 0
    
    def test_fit_least_squares(self):
        """Test fitting with least squares."""
        model = LinearRegression(cost_function="least_squares")
        weight_history, cost_history = model.fit(
            self.x, self.y, alpha=0.01, max_iterations=100
        )
        
        assert len(weight_history) == 100
        assert len(cost_history) == 100
        assert model.weights is not None
        assert len(model.weights) == 2
    
    def test_fit_least_absolute_deviations(self):
        """Test fitting with least absolute deviations."""
        model = LinearRegression(cost_function="least_absolute_deviations")
        weight_history, cost_history = model.fit(
            self.x, self.y, alpha=0.01, max_iterations=100
        )
        
        assert len(weight_history) == 100
        assert len(cost_history) == 100
        assert model.weights is not None
    
    def test_predict(self):
        """Test prediction functionality."""
        model = LinearRegression()
        model.fit(self.x, self.y, alpha=0.01, max_iterations=100)
        
        x_test = np.array([1.0, 2.0, 3.0])
        y_pred = model.predict(x_test)
        
        assert len(y_pred) == 3
        assert all(isinstance(val, (int, float)) for val in y_pred)
    
    def test_predict_without_fit(self):
        """Test that prediction fails without fitting."""
        model = LinearRegression()
        x_test = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(x_test)


class TestRidgeRegression:
    """Test cases for RidgeRegression class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = RidgeRegression(lambda_reg=1.0)
        np.random.seed(42)
        self.x = np.random.uniform(0, 10, 50)
        self.y = 2 * self.x + 1 + np.random.normal(0, 0.1, 50)
    
    def test_ridge_cost(self):
        """Test ridge cost function."""
        w = np.array([1.0, 2.0])
        cost = self.model.ridge_cost(w, self.x, self.y)
        
        assert isinstance(cost, float)
        assert cost >= 0
    
    def test_fit(self):
        """Test ridge regression fitting."""
        weight_history, cost_history = self.model.fit(
            self.x, self.y, alpha=0.01, max_iterations=100
        )
        
        assert len(weight_history) == 100
        assert len(cost_history) == 100
        assert self.model.weights is not None
    
    def test_predict(self):
        """Test ridge regression prediction."""
        self.model.fit(self.x, self.y, alpha=0.01, max_iterations=100)
        
        x_test = np.array([1.0, 2.0, 3.0])
        y_pred = self.model.predict(x_test)
        
        assert len(y_pred) == 3


class TestLassoRegression:
    """Test cases for LassoRegression class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = LassoRegression(lambda_reg=1.0)
        np.random.seed(42)
        self.x = np.random.uniform(0, 10, 50)
        self.y = 2 * self.x + 1 + np.random.normal(0, 0.1, 50)
    
    def test_lasso_cost(self):
        """Test lasso cost function."""
        w = np.array([1.0, 2.0])
        cost = self.model.lasso_cost(w, self.x, self.y)
        
        assert isinstance(cost, float)
        assert cost >= 0
    
    def test_fit(self):
        """Test lasso regression fitting."""
        weight_history, cost_history = self.model.fit(
            self.x, self.y, alpha=0.01, max_iterations=100
        )
        
        assert len(weight_history) == 100
        assert len(cost_history) == 100
        assert self.model.weights is not None
    
    def test_predict(self):
        """Test lasso regression prediction."""
        self.model.fit(self.x, self.y, alpha=0.01, max_iterations=100)
        
        x_test = np.array([1.0, 2.0, 3.0])
        y_pred = self.model.predict(x_test)
        
        assert len(y_pred) == 3


class TestGradientDescent:
    """Test cases for GradientDescent class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = GradientDescent()
        
        def simple_cost(w, x, y):
            return np.sum((w[0] + w[1] * x - y) ** 2)
        
        self.cost_function = simple_cost
        np.random.seed(42)
        self.x = np.random.uniform(0, 10, 50)
        self.y = 2 * self.x + 1 + np.random.normal(0, 0.1, 50)
    
    def test_optimize(self):
        """Test gradient descent optimization."""
        w_init = np.array([0.0, 0.0])
        weight_history, cost_history = self.optimizer.optimize(
            self.cost_function, alpha=0.01, max_iterations=100,
            w_init=w_init, x=self.x, y=self.y
        )
        
        assert len(weight_history) == 100
        assert len(cost_history) == 100
        assert len(weight_history[0]) == 2
    
    def test_get_best_weights(self):
        """Test getting best weights."""
        w_init = np.array([0.0, 0.0])
        self.optimizer.optimize(
            self.cost_function, alpha=0.01, max_iterations=100,
            w_init=w_init, x=self.x, y=self.y
        )
        
        best_weights = self.optimizer.get_best_weights()
        assert len(best_weights) == 2
    
    def test_get_convergence_info(self):
        """Test convergence information."""
        w_init = np.array([0.0, 0.0])
        self.optimizer.optimize(
            self.cost_function, alpha=0.01, max_iterations=100,
            w_init=w_init, x=self.x, y=self.y
        )
        
        info = self.optimizer.get_convergence_info()
        assert 'initial_cost' in info
        assert 'final_cost' in info
        assert 'cost_reduction' in info
        assert 'iterations' in info
        assert 'converged' in info


class TestMetrics:
    """Test cases for metrics classes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.y_actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        self.w = np.array([1.0, -0.5, 2.0])
    
    def test_mse(self):
        """Test Mean Squared Error metric."""
        mse = MSE()
        error = mse(self.y_actual, self.y_pred)
        
        assert isinstance(error, float)
        assert error >= 0
    
    def test_mad(self):
        """Test Mean Absolute Deviation metric."""
        mad = MAD()
        error = mad(self.y_actual, self.y_pred)
        
        assert isinstance(error, float)
        assert error >= 0
    
    def test_l1_regularizer(self):
        """Test L1 regularizer."""
        l1_reg = L1Regularizer()
        penalty = l1_reg(self.w)
        
        assert isinstance(penalty, float)
        assert penalty >= 0
        assert penalty == np.sum(np.abs(self.w))
    
    def test_l2_regularizer(self):
        """Test L2 regularizer."""
        l2_reg = L2Regularizer()
        penalty = l2_reg(self.w)
        
        assert isinstance(penalty, float)
        assert penalty >= 0
        assert penalty == np.sum(self.w ** 2)
    
    def test_model_evaluator(self):
        """Test ModelEvaluator class."""
        evaluator = ModelEvaluator()
        results = evaluator.evaluate(self.y_actual, self.y_pred)
        
        assert 'mse' in results
        assert 'mad' in results
        assert isinstance(results['mse'], float)
        assert isinstance(results['mad'], float)


class TestUtils:
    """Test cases for utility functions."""
    
    def test_load_data(self):
        """Test data loading function."""
        # Create temporary test data
        test_data = np.array([[1, 2, 3], [4, 5, 6]])
        np.savetxt('test_data.csv', test_data, delimiter=',')
        
        try:
            x, y = load_data('test_data.csv')
            assert x.shape == (1, 3)
            assert y.shape == (1, 3)
        finally:
            # Clean up
            import os
            if os.path.exists('test_data.csv'):
                os.remove('test_data.csv')
    
    def test_load_weather_data(self):
        """Test weather data loading function."""
        # Create temporary weather data
        weather_data = np.array([
            ['2016-01-01', 'Clear', 0.0, 10.0, 9.0, 0.8, 5.0, 180, 10.0, 0, 1013.25, 'Clear'],
            ['2016-01-02', 'Cloudy', 0.1, 12.0, 11.0, 0.7, 8.0, 200, 8.0, 0, 1015.0, 'Cloudy']
        ])
        
        header = 'Date,Summary,Precip,Temperature,Apparent_Temperature,Humidity,Wind_Speed,Wind_Bearing,Visibility,Loud_Cover,Pressure,Daily_Summary'
        np.savetxt('test_weather.csv', weather_data, delimiter=',', fmt='%s', header=header, comments='')
        
        try:
            X, y = load_weather_data('test_weather.csv')
            assert X.shape[0] == 2  # 2 samples
            assert X.shape[1] == 6  # 6 features
            assert len(y) == 2  # 2 targets
        finally:
            # Clean up
            import os
            if os.path.exists('test_weather.csv'):
                os.remove('test_weather.csv')


if __name__ == "__main__":
    pytest.main([__file__])
