"""
Offline evaluation of trained transformer model on test set.

Computes:
- Accuracy (exact action match)
- Top-2 accuracy (correct action in top 2 predictions)
- Per-action precision, recall, F1
- Confusion matrix
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    confusion_matrix,
    top_k_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from tqdm import tqdm

from models.transformer import TransformerPolicy
from training.dataloader import create_dataloaders
from training.utils import get_device


def compute_metrics(model, dataloader, device):
    """Compute comprehensive metrics on a dataset."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            boards = batch['board'].to(device)
            actions = batch['action'].to(device)
            
            logits = model(boards)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(actions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)
    
    # Top-2 accuracy
    metrics['top2_accuracy'] = top_k_accuracy_score(
        all_labels, 
        all_probs, 
        k=2,
        labels=np.arange(4)
    )
    
    # Per-class metrics
    class_report = classification_report(
        all_labels, 
        all_preds,
        target_names=['UP', 'DOWN', 'LEFT', 'RIGHT'],
        output_dict=True,
        zero_division=0
    )
    metrics['class_report'] = class_report
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = conf_matrix
    
    return metrics, all_preds, all_labels


def plot_confusion_matrix(conf_matrix, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    # Normalize by row (true label)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        conf_matrix_norm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=['UP', 'DOWN', 'LEFT', 'RIGHT'],
        yticklabels=['UP', 'DOWN', 'LEFT', 'RIGHT'],
        cbar_kws={'label': 'Proportion'}
    )
    
    plt.title('Confusion Matrix (Normalized by True Label)')
    plt.ylabel('True Action')
    plt.xlabel('Predicted Action')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {save_path}")


def plot_per_class_metrics(class_report, save_path):
    """Plot per-class precision, recall, F1."""
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    precision = [class_report[action]['precision'] for action in actions]
    recall = [class_report[action]['recall'] for action in actions]
    f1 = [class_report[action]['f1-score'] for action in actions]
    
    x = np.arange(len(actions))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Action')
    ax.set_ylabel('Score')
    ax.set_title('Per-Action Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(actions)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Per-class metrics saved to: {save_path}")


def print_evaluation_summary(metrics):
    """Print evaluation summary to console."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:      {metrics['accuracy']:.2%}")
    print(f"  Top-2 Accuracy: {metrics['top2_accuracy']:.2%}")
    
    print(f"\nPer-Action Metrics:")
    print(f"{'Action':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    for action in actions:
        stats = metrics['class_report'][action]
        print(f"{action:<10} {stats['precision']:<12.2%} {stats['recall']:<12.2%} "
              f"{stats['f1-score']:<12.2%} {int(stats['support']):<10}")
    
    macro_avg = metrics['class_report']['macro avg']
    print("-" * 70)
    print(f"{'Macro Avg':<10} {macro_avg['precision']:<12.2%} {macro_avg['recall']:<12.2%} "
          f"{macro_avg['f1-score']:<12.2%}")
    
    print("\nConfusion Matrix:")
    print("     ", "  ".join([f"{a:>6}" for a in actions]))
    for i, action in enumerate(actions):
        row = metrics['confusion_matrix'][i]
        print(f"{action:<6}", "  ".join([f"{v:>6}" for v in row]))
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate transformer model on test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/transformer/best_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/training_games.jsonl",
        help="Path to training data JSONL file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation (faster evaluation)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    model = TransformerPolicy(
        embed_dim=checkpoint.get('embed_dim', 128),
        num_heads=checkpoint.get('num_heads', 8),
        num_layers=checkpoint.get('num_layers', 4),
        dim_feedforward=checkpoint.get('dim_feedforward', 512),
        dropout=checkpoint.get('dropout', 0.1),
        head_dropout=checkpoint.get('head_dropout', 0.2)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded (epoch {checkpoint['epoch']})")
    
    # Create dataloaders
    print(f"\nLoading data from: {args.data_path}")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=not args.no_augment
    )
    print(f"✓ Test set: {len(test_loader.dataset):,} samples")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics, preds, labels = compute_metrics(model, test_loader, device)
    
    # Print results
    print_evaluation_summary(metrics)
    
    # Save results
    results_path = output_dir / "test_metrics.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_metrics = {
            'accuracy': float(metrics['accuracy']),
            'top2_accuracy': float(metrics['top2_accuracy']),
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'class_report': metrics['class_report'],
            'checkpoint': str(args.checkpoint),
            'test_samples': len(labels)
        }
        json.dump(serializable_metrics, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Plot confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(metrics['confusion_matrix'], cm_path)
    
    # Plot per-class metrics
    class_metrics_path = output_dir / "per_class_metrics.png"
    plot_per_class_metrics(metrics['class_report'], class_metrics_path)
    
    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
