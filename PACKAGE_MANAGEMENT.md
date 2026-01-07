# Package Management with uv (Enforced)

**Policy**: This project enforces the use of `uv` for all Python package management.

## Quick Start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Run setup script
./setup.sh

# 3. Start using the project
uv run python scripts/generate_dataset.py --games 10
```

## Why uv is Enforced

### Technical Reasons
1. **Speed**: 10-100x faster than pip
2. **Determinism**: Guaranteed reproducible builds
3. **Modern**: Uses latest Python packaging standards
4. **Reliability**: Better dependency resolution

### Project Reasons
1. **Spec-Driven**: Tool choice is part of the specification
2. **Consistency**: All contributors use same tooling
3. **CI/CD**: Faster automated testing and deployment
4. **Education**: Demonstrates modern Python practices

## What This Means

### ✅ Allowed
```bash
uv pip install <package>
uv sync
uv run python <script>
```

### ❌ Not Allowed (in official documentation)
```bash
pip install <package>
python <script>  # Without uv run or activated venv
```

## Files Managed by uv

| File | Purpose | Notes |
|------|---------|-------|
| `pyproject.toml` | Primary dependency specification | Modern standard |
| `requirements.txt` | Legacy compatibility | Still works with uv |
| `uv.lock` | Locked dependencies | Git-ignored, auto-generated |
| `.venv/` | Virtual environment | Git-ignored, managed by uv |

## Common Commands

### Setup
```bash
# Create venv and install dependencies
uv sync

# Install specific dependency groups
uv pip install -e ".[ml]"      # Machine learning deps
uv pip install -e ".[dev]"     # Development tools
uv pip install -e ".[deploy]"  # Deployment tools
```

### Development
```bash
# Run a script with uv (auto-handles venv)
uv run python scripts/generate_dataset.py --games 10

# Run tests
uv run pytest

# Format code
uv run black .
uv run ruff check .
```

### Dependency Management
```bash
# Add a dependency
uv pip install numpy

# Update dependencies
uv pip install --upgrade numpy

# Show installed packages
uv pip list

# Generate requirements.txt from pyproject.toml
uv pip compile pyproject.toml -o requirements.txt
```

## Integration with IDEs

### VS Code
```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```

### PyCharm
1. Settings → Project → Python Interpreter
2. Add Interpreter → Existing environment
3. Select `.venv/bin/python`

## CI/CD Integration

### GitHub Actions
```yaml
steps:
  - name: Install uv
    run: curl -LsSf https://astral.sh/uv/install.sh | sh
    
  - name: Install dependencies
    run: uv sync
    
  - name: Run tests
    run: uv run pytest
```

### GitLab CI
```yaml
before_script:
  - curl -LsSf https://astral.sh/uv/install.sh | sh
  - uv sync

test:
  script:
    - uv run pytest
```

## Migration from pip

If you have existing pip workflows:

| Old (pip) | New (uv) |
|-----------|----------|
| `pip install -r requirements.txt` | `uv pip install -r requirements.txt` |
| `pip install package` | `uv pip install package` |
| `python -m venv .venv` | `uv venv` |
| `pip freeze > requirements.txt` | `uv pip freeze > requirements.txt` |
| `python script.py` | `uv run python script.py` |

## Troubleshooting

### "uv: command not found"
```bash
# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.cargo/bin:$PATH"

# Or reinstall
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### "No such file or directory: .venv"
```bash
# Create virtual environment
uv venv .venv
```

### "Package not found"
```bash
# Update uv
uv self update

# Clear cache
rm -rf ~/.cache/uv
```

### "Permission denied"
```bash
# Use virtual environment (recommended)
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Enforcement Checklist

- [x] `pyproject.toml` created with dependency specifications
- [x] `requirements.txt` updated with uv usage note
- [x] All documentation uses `uv` commands
- [x] `setup.sh` script uses uv
- [x] README.md enforces uv usage
- [x] UV_GUIDE.md provides detailed instructions
- [x] `.gitignore` includes uv-specific patterns

## Exceptions

**None.** This project enforces uv usage. If you cannot use uv:

1. Install uv anyway (it works everywhere Python works)
2. Or manually install dependencies listed in `pyproject.toml`

But all official documentation assumes uv.

## References

- **uv Documentation**: https://github.com/astral-sh/uv
- **Installation**: https://astral.sh/uv
- **PyPI**: https://pypi.org/project/uv/

---

**Policy Status**: ✅ Enforced  
**Last Updated**: 2026-01-07  
**Rationale**: Spec-driven development includes tooling specifications
