# Contributing Guide

Thank you for your interest in contributing to the DataMCPServerAgent project! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your contribution
4. Make your changes
5. Push your changes to your fork
6. Submit a pull request

## Development Environment

1. Install the package in development mode:
   ```bash
   pip install -e .
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. Create a `.env` file from the template:
   ```bash
   cp .env.template .env
   ```
   Then edit the `.env` file with your credentials.

## Code Style

This project follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. Please ensure your code adheres to this style guide.

You can use tools like `flake8` and `black` to check and format your code:

```bash
# Check code style
flake8 src tests

# Format code
black src tests
```

## Testing

Please write tests for your code. This project uses `unittest` for testing.

To run the tests:

```bash
python run_tests.py
```

To run specific tests:

```bash
python run_tests.py test_pattern
```

## Documentation

Please update the documentation when making changes to the code. This project uses Markdown for documentation.

Documentation files are located in the `docs/` directory.

## Pull Request Process

1. Ensure your code follows the code style guidelines
2. Update the documentation if necessary
3. Add tests for your changes
4. Ensure all tests pass
5. Submit a pull request

## Commit Messages

Please use descriptive commit messages that explain the changes you've made. Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages.

Examples:
- `feat: Add multi-agent learning system`
- `fix: Fix memory persistence bug`
- `docs: Update installation instructions`
- `test: Add tests for memory system`
- `refactor: Refactor tool selection algorithm`

## Issue Reporting

If you find a bug or have a feature request, please create an issue on GitHub. Please include as much information as possible, including:

- A clear and descriptive title
- A detailed description of the issue or feature request
- Steps to reproduce the issue (if applicable)
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Environment information (OS, Python version, etc.)

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We strive to create a welcoming and inclusive environment for all contributors.

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.
