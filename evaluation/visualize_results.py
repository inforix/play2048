"""
Visualization tools for training results and evaluation metrics.

Plots:
- Training curves (loss, accuracy over epochs)
- Learning rate schedule
- Comparison of train vs validation metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_training_history(history_path):
    """Load training history from JSON file."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Normalize key names (handle both old and new formats)
    key_mapping = {
        'train_accuracy': 'train_acc',
        'val_accuracy': 'val_acc',
        'train_top2_accuracy': 'train_top2',
        'val_top2_accuracy': 'val_top2'
    }
    
    normalized_history = {}
    for key, value in history.items():
        # Use mapped key if available, otherwise use original
        normalized_key = key_mapping.get(key, key)
        normalized_history[normalized_key] = value
    
    return normalized_history


def plot_loss_curves(history, save_path):
    """Plot training and validation loss over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    
    # Mark best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    ax.plot(best_epoch, best_val_loss, 'g*', markersize=15, label=f'Best Val Loss: {best_val_loss:.4f}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Loss curves saved to: {save_path}")


def plot_accuracy_curves(history, save_path):
    """Plot training and validation accuracy over epochs."""
    epochs = range(1, len(history['train_acc']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Exact accuracy
    ax1.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax1.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    
    best_epoch = np.argmax(history['val_acc']) + 1
    best_val_acc = max(history['val_acc'])
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax1.plot(best_epoch, best_val_acc, 'g*', markersize=15, label=f'Best: {best_val_acc:.2f}%')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Exact Match Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top-2 accuracy
    ax2.plot(epochs, history['train_top2'], 'b-', label='Train Top-2', linewidth=2)
    ax2.plot(epochs, history['val_top2'], 'r-', label='Val Top-2', linewidth=2)
    
    best_epoch_top2 = np.argmax(history['val_top2']) + 1
    best_val_top2 = max(history['val_top2'])
    ax2.axvline(x=best_epoch_top2, color='g', linestyle='--', alpha=0.5)
    ax2.plot(best_epoch_top2, best_val_top2, 'g*', markersize=15, label=f'Best: {best_val_top2:.2f}%')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Top-2 Accuracy (%)')
    ax2.set_title('Top-2 Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Accuracy curves saved to: {save_path}")


def plot_learning_rate(history, save_path):
    """Plot learning rate schedule."""
    epochs = range(1, len(history['learning_rate']) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(epochs, history['learning_rate'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Learning rate schedule saved to: {save_path}")


def plot_combined_metrics(history, save_path):
    """Plot all metrics in a single figure."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    best_epoch = np.argmin(history['val_loss']) + 1
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    best_epoch = np.argmax(history['val_acc']) + 1
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-2 Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, history['train_top2'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_top2'], 'r-', label='Val', linewidth=2)
    best_epoch = np.argmax(history['val_top2']) + 1
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Top-2 Accuracy (%)')
    ax.set_title('Top-2 Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, history['learning_rate'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Metrics Overview', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Combined metrics saved to: {save_path}")


def plot_overfitting_analysis(history, save_path):
    """Plot train-val gap to analyze overfitting."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Calculate gaps
    loss_gap = np.array(history['val_loss']) - np.array(history['train_loss'])
    acc_gap = np.array(history['val_acc']) - np.array(history['train_acc'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss gap
    ax1.plot(epochs, loss_gap, 'purple', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.fill_between(epochs, 0, loss_gap, alpha=0.3, color='purple')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Val Loss - Train Loss')
    ax1.set_title('Loss Gap (Positive = Overfitting)')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy gap
    ax2.plot(epochs, acc_gap, 'orange', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.fill_between(epochs, 0, acc_gap, alpha=0.3, color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val Acc - Train Acc (%)')
    ax2.set_title('Accuracy Gap (Negative = Overfitting)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Overfitting analysis saved to: {save_path}")


def print_training_summary(history):
    """Print summary of training results."""
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    num_epochs = len(history['train_loss'])
    print(f"\nTotal Epochs: {num_epochs}")
    
    # Best metrics
    best_val_loss_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    
    best_val_acc_epoch = np.argmax(history['val_acc']) + 1
    best_val_acc = max(history['val_acc'])
    
    best_val_top2_epoch = np.argmax(history['val_top2']) + 1
    best_val_top2 = max(history['val_top2'])
    
    print(f"\nBest Validation Metrics:")
    print(f"  Loss:     {best_val_loss:.4f} (epoch {best_val_loss_epoch})")
    print(f"  Accuracy: {best_val_acc:.2f}% (epoch {best_val_acc_epoch})")
    print(f"  Top-2:    {best_val_top2:.2f}% (epoch {best_val_top2_epoch})")
    
    # Final metrics
    print(f"\nFinal Epoch Metrics:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Val Loss:   {history['val_loss'][-1]:.4f}")
    print(f"  Train Acc:  {history['train_acc'][-1]:.2f}%")
    print(f"  Val Acc:    {history['val_acc'][-1]:.2f}%")
    
    # Overfitting check
    final_loss_gap = history['val_loss'][-1] - history['train_loss'][-1]
    final_acc_gap = history['val_acc'][-1] - history['train_acc'][-1]
    
    print(f"\nGeneralization (Final Epoch):")
    print(f"  Loss Gap:  {final_loss_gap:+.4f}")
    print(f"  Acc Gap:   {final_acc_gap:+.2f}%")
    
    if final_loss_gap > 0.5 or final_acc_gap < -5:
        print("  ⚠️  Warning: Possible overfitting detected")
    else:
        print("  ✓ Good generalization")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument(
        "--history",
        type=str,
        default="results/training_curves/transformer/training_history.json",
        help="Path to training history JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/training_curves/transformer/plots",
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training history
    print(f"Loading training history from: {args.history}")
    history = load_training_history(args.history)
    
    # Print summary
    print_training_summary(history)
    
    # Generate plots
    print("Generating plots...")
    
    plot_loss_curves(history, output_dir / "loss_curves.png")
    plot_accuracy_curves(history, output_dir / "accuracy_curves.png")
    plot_learning_rate(history, output_dir / "learning_rate.png")
    plot_combined_metrics(history, output_dir / "combined_metrics.png")
    plot_overfitting_analysis(history, output_dir / "overfitting_analysis.png")
    
    print("\n" + "="*70)
    print("Visualization complete!")
    print(f"Plots saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
