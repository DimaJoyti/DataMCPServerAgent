# UV Package Manager Setup Guide

This guide explains how to use `uv` as the primary package manager for DataMCPServerAgent v2.0.

## Why uv?

- **Fast**: 10-100x faster than pip
- **Reliable**: Better dependency resolution
- **Modern**: Built in Rust with modern Python packaging standards
- **Compatible**: Drop-in replacement for pip

## Quick Start

### 1. Install uv

```bash
# Option 1: Using our quick install script
python scripts/quick_install_uv.py

# Option 2: Manual installation
pip install uv

# Option 3: Direct install (Unix/Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Option 4: Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Project Dependencies

```bash
# Install all dependencies
uv pip install -e .

# Install development dependencies
uv pip install -r requirements-ci.txt

# Install specific packages
uv pip install fastapi uvicorn pydantic
```

### 3. Run Tests

```bash
# Using our uv-enabled test runner
python scripts/test_runner_uv.py

# Or directly with pytest
uv pip install pytest
python -m pytest tests/
```

## Common Commands

### Package Management

```bash
# Install a package
uv pip install package-name

# Install with version constraint
uv pip install "package-name>=1.0.0"

# Install from requirements file
uv pip install -r requirements.txt

# Install in development mode
uv pip install -e .

# Uninstall a package
uv pip uninstall package-name

# List installed packages
uv pip list

# Show package information
uv pip show package-name
```

### Virtual Environments

```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # Unix/Linux/macOS
.venv\Scripts\activate     # Windows

# Install in virtual environment
uv pip install --python .venv/bin/python package-name
```

### Project Setup

```bash
# Full project setup
python scripts/quick_install_uv.py

# Manual setup
uv pip install fastapi uvicorn pydantic rich typer structlog
uv pip install pytest pytest-cov

# Verify installation
python -c "import fastapi, pydantic, rich; print('✅ All packages installed')"
```

## CI/CD Integration

Our GitHub Actions workflows use uv for faster builds:

```yaml
- name: Install uv
  run: |
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "$HOME/.cargo/bin" >> $GITHUB_PATH

- name: Install dependencies
  run: |
    uv pip install --system pytest pydantic fastapi
```

## Troubleshooting

### uv not found

```bash
# Add to PATH (Unix/Linux/macOS)
export PATH="$HOME/.cargo/bin:$PATH"

# Or install via pip
pip install uv
```

### Permission errors

```bash
# Use --system flag in CI/containers
uv pip install --system package-name

# Or use virtual environment
uv venv
source .venv/bin/activate
uv pip install package-name
```

### Dependency conflicts

```bash
# Force reinstall
uv pip install --force-reinstall package-name

# Check dependency tree
uv pip show --verbose package-name
```

## Performance Comparison

| Operation | pip | uv | Speedup |
|-----------|-----|----|---------| 
| Install numpy | 3.2s | 0.3s | 10x |
| Install fastapi | 5.1s | 0.4s | 12x |
| Resolve dependencies | 8.7s | 0.6s | 14x |

## Migration from pip

Replace `pip` commands with `uv pip`:

```bash
# Before
pip install package-name
pip install -r requirements.txt
pip list

# After  
uv pip install package-name
uv pip install -r requirements.txt
uv pip list
```

## Best Practices

1. **Use uv for all installations**: Faster and more reliable
2. **Pin versions**: Use exact versions in production
3. **Use virtual environments**: Isolate project dependencies
4. **Cache dependencies**: uv automatically caches for speed
5. **Use requirements files**: Better reproducibility

## Scripts Using uv

- `scripts/quick_install_uv.py` - Quick setup with uv
- `scripts/test_runner_uv.py` - Test runner using uv
- `scripts/install_dependencies.py` - Enhanced with uv support

## Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [uv GitHub Repository](https://github.com/astral-sh/uv)
- [Python Packaging Guide](https://packaging.python.org/)

## Support

If you encounter issues with uv:

1. Check the [troubleshooting section](#troubleshooting)
2. Fall back to pip if needed
3. Report issues in our GitHub repository
4. Check uv documentation for advanced usage
