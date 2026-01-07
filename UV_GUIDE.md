# Package Management with uv

This project **enforces** the use of `uv` for Python package management.

## Why uv?

- ⚡ **Speed**: 10-100x faster than pip
- 🔒 **Reliability**: Deterministic dependency resolution
- 🎯 **Modern**: Python's next-generation package manager
- 🔧 **Compatible**: Works with existing pip, requirements.txt, and pyproject.toml

## Installation

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Via pip (if you already have Python)
pip install uv

# Verify installation
uv --version
```

## Usage

### Basic Setup

```bash
# Install core dependencies (numpy, tqdm)
uv pip install -r requirements.txt

# Or use pyproject.toml (recommended)
uv sync
```

### With Optional Dependencies

```bash
# Install with ML dependencies (torch, pandas, matplotlib, etc.)
uv pip install -e ".[ml]"

# Install with development tools (pytest, black, ruff)
uv pip install -e ".[dev]"

# Install with deployment tools (onnx, onnxruntime)
uv pip install -e ".[deploy]"

# Install everything
uv pip install -e ".[ml,dev,deploy]"
```

### Project-Specific Commands

```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install dependencies in one command
uv pip install -r requirements.txt

# Sync from pyproject.toml (creates venv if needed)
uv sync

# Add a new dependency
uv pip install some-package

# Run a script with uv (auto-creates venv)
uv run python scripts/generate_dataset.py --games 10
```

## Project Dependencies

### Core (Required)
- `numpy>=1.24.0` - Numerical computing
- `tqdm>=4.65.0` - Progress bars

### Optional: ML Component
- `torch>=2.0.0` - Neural networks
- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.3.0` - ML utilities
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization
- `tensorboard>=2.13.0` - Training monitoring

### Optional: Development
- `jupyter>=1.0.0` - Interactive notebooks
- `plotly>=5.15.0` - Interactive plots
- `pytest>=7.4.0` - Testing
- `black>=23.0.0` - Code formatting
- `ruff>=0.1.0` - Fast linting

### Optional: Deployment
- `onnx>=1.14.0` - Model export
- `onnxruntime>=1.15.0` - ONNX inference

## Quick Start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repository
cd play2048

# 3. Install dependencies
uv sync

# 4. Run dataset generator
uv run python scripts/generate_dataset.py --games 10 --verbose
```

## Common Tasks

### Generate Dataset
```bash
uv run python scripts/generate_dataset.py --games 500
```

### Run Tests (when implemented)
```bash
uv run pytest
```

### Format Code (with dev dependencies)
```bash
uv pip install -e ".[dev]"
uv run black .
uv run ruff check .
```

### Jupyter Notebook (with dev dependencies)
```bash
uv pip install -e ".[dev]"
uv run jupyter notebook
```

## Troubleshooting

### uv not found
```bash
# Make sure uv is in your PATH
export PATH="$HOME/.cargo/bin:$PATH"  # Add to ~/.bashrc or ~/.zshrc
```

### Permission denied
```bash
# Use --user flag or virtual environment
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Slow first run
```bash
# uv caches packages, first run downloads
# Subsequent runs are much faster
```

## Migration from pip

If you were using pip before:

```bash
# Old way (pip)
pip install -r requirements.txt

# New way (uv - faster!)
uv pip install -r requirements.txt

# Even better (with pyproject.toml)
uv sync
```

Your existing `requirements.txt` still works with uv!

## Why This Project Enforces uv

1. **Consistency**: All contributors use same package manager
2. **Speed**: Faster CI/CD and local development
3. **Reliability**: Deterministic builds across environments
4. **Modern**: Following Python packaging best practices
5. **Education**: Demonstrating spec-driven approach includes tooling choices

## References

- **uv Documentation**: https://github.com/astral-sh/uv
- **uv Installation**: https://astral.sh/uv
- **pyproject.toml**: Modern Python project configuration

---

**Note**: If you absolutely must use pip, you can, but uv is strongly recommended and all documentation assumes uv usage.
