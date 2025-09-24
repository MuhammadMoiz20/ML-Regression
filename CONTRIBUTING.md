# Contributing to ML Regression

Thank you for your interest in contributing to the ML Regression package! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/ml-regression.git
   cd ml-regression
   ```

2. **Install Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## 📝 How to Contribute

### Reporting Issues

Before creating an issue, please:
- Check existing issues to avoid duplicates
- Use the issue templates provided
- Include Python version, OS, and package version
- Provide minimal reproducible examples

### Suggesting Enhancements

- Use the "Enhancement" issue template
- Clearly describe the proposed feature
- Explain the use case and benefits
- Consider implementation complexity

### Code Contributions

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   pytest
   pytest --cov=src/ml_regression
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new regression algorithm"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 🎯 Coding Standards

### Code Style

- **Formatting**: Use Black with line length 88
- **Linting**: Follow flake8 guidelines
- **Type Hints**: All public functions must have type annotations
- **Docstrings**: Use Google-style docstrings

### Example Code Style

```python
def linear_regression(
    x: np.ndarray, 
    y: np.ndarray, 
    alpha: float = 0.01
) -> Tuple[np.ndarray, List[float]]:
    """
    Perform linear regression using gradient descent.
    
    Args:
        x: Input features of shape (n_samples,)
        y: Target values of shape (n_samples,)
        alpha: Learning rate for gradient descent
        
    Returns:
        Tuple of (final_weights, cost_history)
        
    Raises:
        ValueError: If input shapes don't match
    """
    if len(x) != len(y):
        raise ValueError("Input shapes must match")
    
    # Implementation here
    return weights, cost_history
```

### Testing Requirements

- **Coverage**: Minimum 80% test coverage
- **Test Types**: Unit tests, integration tests, and edge cases
- **Test Names**: Descriptive test names starting with `test_`
- **Fixtures**: Use pytest fixtures for common setup

### Example Test

```python
def test_linear_regression_convergence():
    """Test that linear regression converges to correct solution."""
    # Arrange
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])  # Perfect linear relationship
    
    # Act
    weights, cost_history = linear_regression(x, y, alpha=0.1)
    
    # Assert
    assert len(weights) == 2
    assert cost_history[-1] < cost_history[0]  # Cost decreases
    assert abs(weights[1] - 2.0) < 0.1  # Slope close to 2
```

## 📚 Documentation Standards

### Docstrings

All public functions, classes, and methods must have docstrings:

```python
class LinearRegression:
    """
    Linear regression model with gradient descent optimization.
    
    This class implements linear regression using gradient descent
    with support for different cost functions and regularization.
    
    Attributes:
        weights: Fitted model weights
        cost_history: Training cost over iterations
    """
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the linear regression model.
        
        Args:
            x: Training features
            y: Training targets
            
        Raises:
            ValueError: If input data is invalid
        """
        pass
```

### README Updates

When adding new features:
- Update the Quick Start section
- Add new examples to the Examples section
- Update the API Documentation
- Include performance benchmarks if applicable

## 🔄 Pull Request Process

### Before Submitting

1. **Run Tests**
   ```bash
   pytest
   pytest --cov=src/ml_regression
   ```

2. **Check Code Quality**
   ```bash
   black --check src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

3. **Update Documentation**
   - Update docstrings
   - Add examples if needed
   - Update README if applicable

### PR Requirements

- **Title**: Use conventional commit format
- **Description**: Clear description of changes
- **Tests**: All tests must pass
- **Coverage**: Maintain or improve test coverage
- **Documentation**: Update relevant documentation

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and linting
2. **Code Review**: At least one maintainer reviews the code
3. **Testing**: Manual testing of new features
4. **Documentation**: Review of documentation updates

## 🏷️ Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Update Version**: Update version in `pyproject.toml`
2. **Update Changelog**: Add changes to `CHANGELOG.md`
3. **Create Release**: Use GitHub release workflow
4. **Publish**: Package automatically published to PyPI

## 🐛 Bug Reports

When reporting bugs, please include:

- **Environment**: Python version, OS, package version
- **Steps to Reproduce**: Minimal code example
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Error Messages**: Full traceback if applicable

## 💡 Feature Requests

For feature requests:

- **Use Case**: Explain why this feature is needed
- **Proposed Solution**: Describe your proposed implementation
- **Alternatives**: Consider other approaches
- **Impact**: Discuss potential impact on existing code

## 📞 Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Create issues for bugs and feature requests
- **Email**: Contact maintainers for sensitive issues

## 🎉 Recognition

Contributors will be:
- Listed in the README
- Mentioned in release notes
- Credited in documentation

Thank you for contributing to ML Regression! 🚀
