# Contributing to Manify

Thank you for your interest in contributing to Manify! We welcome contributions of all kinds.

## Getting Started

1. **Fork and clone** the repository
2. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```
3. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Code Quality Standards

### Type Annotations
- Use type annotations for all functions and methods
- Use `jaxtyping` for tensor shape annotations:
  ```python
  from jaxtyping import Float
  import torch
  
  def process_embeddings(x: Float[torch.Tensor, "batch dim"]) -> Float[torch.Tensor, "batch output_dim"]:
      ...
  ```

### Testing
- Write unit tests for all new functionality
- **Coverage requirement**: 80%+ for new code
- Run tests with beartype enabled (as in CI):
  ```bash
  pytest tests --cov
  ```
- Tests should cover edge cases and error conditions

### Code Style
- We use **Ruff** for linting and formatting
- Check your code before committing:
  ```bash
  ruff check manify/
  ruff format manify/
  ```
- Type check with MyPy:
  ```bash
  mypy manify/
  ```

## Documentation

We especially welcome documentation contributions! Areas where help is needed:

- **Mathematical details**: The [paper](https://arxiv.org/abs/2503.09576) contains rich mathematical content that could be integrated into the docs
- **Tutorials**: More examples and tutorials are always appreciated
- **API documentation**: Improving docstrings and examples
- **Use case guides**: Real-world applications and workflows

Documentation uses Google-style docstrings:
```python
def my_function(param1: int, param2: str) -> bool:
    """Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something goes wrong
    """
```

## How to Contribute

To set up your environment and preview changes locally, follow these steps:

### Installation
```bash
pip install -r docs/requirements.txt
```
Building and Serving

```bash
mkdocs build --clean  # Remove previous builds
mkdocs serve          # Start development server
```

Access the live preview at:

```bash
http://127.0.0.1:8000
```
The preview auto-refreshes when you modify files in `docs/` and Changes to Markdown content appear instantly in your browser


## Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** following the standards above
3. **Add tests** with good coverage
4. **Update documentation** as needed
5. **Ensure CI passes** (tests, linting, type checking)
6. **Submit a pull request** with a clear description

## Questions?

- Open an [issue](https://github.com/pchlenski/manify/issues) for bugs or feature requests
- Start a [discussion](https://github.com/pchlenski/manify/discussions) for questions

We appreciate your contributions to making non-Euclidean machine learning more accessible!
