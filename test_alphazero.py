"""
Quick test of AlphaZero training pipeline.
Runs 3 iterations with minimal settings to verify implementation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from training.train_alphazero import train_alphazero
import torch


def main():
    print("="*60)
    print("AlphaZero Pipeline Test")
    print("="*60)
    print("\nThis will run 3 iterations with minimal settings to test:")
    print("  - Self-play data generation")
    print("  - Network training")
    print("  - Model evaluation")
    print("  - Checkpoint saving")
    print("\nExpected time: ~3-5 minutes")
    print("="*60)
    
    # Run minimal training
    train_alphazero(
        iterations=3,
        games_per_iteration=5,
        mcts_simulations=20,
        training_epochs=3,
        batch_size=64,
        eval_interval=2,
        eval_games=10,
        save_interval=2,
        device='auto',
        output_dir='checkpoints/test_alphazero'
    )
    
    print("\n" + "="*60)
    print("✅ AlphaZero pipeline test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
