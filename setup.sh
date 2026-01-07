#!/bin/bash
# Quick setup script for 2048 Spec-Driven Development project
# This script uses uv for package management (enforced)

set -e

echo "🎮 2048 Spec-Driven Development - Setup"
echo "========================================"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found!"
    echo ""
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "Or visit: https://github.com/astral-sh/uv"
    exit 1
fi

echo "✓ uv found: $(uv --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
uv venv .venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment (show instructions)
echo "📝 To activate the virtual environment:"
echo "  source .venv/bin/activate    # macOS/Linux"
echo "  .venv\\Scripts\\activate       # Windows"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
uv pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed data/augmented
mkdir -p checkpoints
mkdir -p results/training_curves results/game_simulations
echo "✓ Directories created"
echo ""

# Success message
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source .venv/bin/activate"
echo "  2. Play the game: open src/game/index.html"
echo "  3. Generate dataset: uv run python scripts/generate_dataset.py --games 10"
echo "  4. Read specs: cat specs/spec.md"
echo ""
echo "For more info, see README.md or UV_GUIDE.md"
