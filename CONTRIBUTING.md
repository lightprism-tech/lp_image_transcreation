# Contributing to Image Transcreation Pipeline

Thank you for your interest in contributing! We welcome all contributions, from bug fixes to new features.

## Code of Conduct

Please be professional and respectful. We are committed to providing a friendly, safe, and welcoming environment for all, regardless of level of experience, gender identity and expression, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, religion, nationality, or other similar characteristic.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork**:
    ```bash
    git clone https://github.com/your-username/image-transcreation-pipeline.git
    cd image-transcreation-pipeline
    ```
3.  **Set up the environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    pip install -e .
    ```

## Development Workflow

1.  **Create a branch** for your feature or fix:
    ```bash
    git checkout -b feature/amazing-feature
    ```
2.  **Make your changes**.
3.  **Run tests** to ensure nothing is broken:
    ```bash
    pytest
    ```
4.  **Commit your changes** using descriptive commit messages.
5.  **Push to your fork**:
    ```bash
    git push origin feature/amazing-feature
    ```
6.  **Open a Pull Request** against the `main` branch.

## Coding Standards

We follow strict code quality standards to maintain a robust and maintainable codebase.

### Style Guide
- **Code Formatter**: We use [Black](https://github.com/psf/black).
- **Linter**: We use [Flake8](https://flake8.pycqa.org/en/latest/).
- **Type Checking**: We use [MyPy](http://mypy-lang.org/).

Before committing, run:
```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Documentation
- Use **docstrings** for all modules, classes, and functions (Google style recommended).
- Update `README.md` and other documentation files if your changes affect usage or setup.

## Testing

- All new features **must** be accompanied by unit tests.
- We use `pytest` for testing.
- Place tests in the `tests/` directory matching the source structure.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub. Include:
- Description of the issue
- Steps to reproduce
- Expected behavior
- Screenshots (if applicable)

Thank you for contributing!
