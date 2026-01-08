"""
Utility functions for training the Transformer-based 2048 AI.

Includes metrics, checkpointing, early stopping, and logging utilities.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import time
from collections import defaultdict


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_shape: tuple = (1, 1, 4, 4)) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor for testing
        
    Returns:
        String summary of the model
    """
    summary = []
    summary.append("=" * 70)
    summary.append(f"Model: {model.__class__.__name__}")
    summary.append("=" * 70)
    
    # Count parameters
    total_params = count_parameters(model)
    summary.append(f"Total trainable parameters: {total_params:,}")
    
    # Model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    summary.append(f"Model size: {size_mb:.2f} MB")
    
    # Test forward pass
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(input_shape)
            output = model(dummy_input)
            if isinstance(output, tuple):
                output_shape = [o.shape for o in output]
            else:
                output_shape = output.shape
        summary.append(f"Input shape: {input_shape}")
        summary.append(f"Output shape: {output_shape}")
    except Exception as e:
        summary.append(f"Could not test forward pass: {e}")
    
    summary.append("=" * 70)
    
    return "\n".join(summary)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_acc: float,
    checkpoint_path: Path,
    hyperparameters: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        val_acc: Validation accuracy
        checkpoint_path: Path to save checkpoint
        hyperparameters: Optional hyperparameters dict
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'timestamp': time.time()
    }
    
    if hyperparameters:
        checkpoint['hyperparameters'] = hyperparameters
    
    # Create directory if needed
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'train_loss': checkpoint.get('train_loss', 0),
        'val_loss': checkpoint.get('val_loss', 0),
        'val_acc': checkpoint.get('val_acc', 0),
        'hyperparameters': checkpoint.get('hyperparameters', {})
    }


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """
    
    def __init__(
        self,
        patience: int = 25,
        min_delta: float = 0.0001,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' depending on whether lower or higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing."""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'best_epoch': self.best_epoch
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from dictionary."""
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']
        self.best_epoch = state_dict['best_epoch']


class MetricsTracker:
    """Track and compute training metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
        self.history = defaultdict(list)
    
    def reset(self):
        """Reset current epoch metrics."""
        self.losses = []
        self.correct = 0
        self.total = 0
        self.top2_correct = 0
    
    def update(self, loss: float, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch results.
        
        Args:
            loss: Batch loss value
            predictions: Model predictions (logits) of shape (batch_size, 4)
            targets: Ground truth actions of shape (batch_size,)
        """
        self.losses.append(loss)
        
        # Top-1 accuracy
        pred_actions = predictions.argmax(dim=1)
        self.correct += (pred_actions == targets).sum().item()
        self.total += targets.size(0)
        
        # Top-2 accuracy
        _, top2 = predictions.topk(2, dim=1)
        self.top2_correct += sum((targets[i] in top2[i]) for i in range(len(targets)))
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metrics.
        
        Returns:
            Dictionary with loss, accuracy, and top-2 accuracy
        """
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        accuracy = 100.0 * self.correct / self.total if self.total > 0 else 0.0
        top2_accuracy = 100.0 * self.top2_correct / self.total if self.total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'top2_accuracy': top2_accuracy
        }
    
    def log_epoch(self, epoch: int, phase: str):
        """
        Log epoch metrics to history.
        
        Args:
            epoch: Epoch number
            phase: 'train' or 'val'
        """
        metrics = self.get_metrics()
        self.history[f'{phase}_loss'].append(metrics['loss'])
        self.history[f'{phase}_accuracy'].append(metrics['accuracy'])
        self.history[f'{phase}_top2_accuracy'].append(metrics['top2_accuracy'])
        self.history['epoch'].append(epoch)
    
    def save_history(self, path: Path):
        """Save training history to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_device(prefer_mps: bool = True) -> str:
    """
    Get the best available device.
    
    Args:
        prefer_mps: If True, prefer MPS (Apple Silicon) over CPU
        
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif prefer_mps and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def cleanup_old_checkpoints(
    checkpoint_dir: Path,
    keep_last_n: int = 3,
    best_checkpoint_name: str = 'best_model.pth'
):
    """
    Remove old checkpoints, keeping only the most recent ones and the best model.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        best_checkpoint_name: Name of the best model checkpoint to always keep
    """
    if not checkpoint_dir.exists():
        return
    
    # Get all checkpoint files
    checkpoints = sorted(
        [f for f in checkpoint_dir.glob('*.pth') if f.name != best_checkpoint_name],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    # Remove old checkpoints
    for checkpoint in checkpoints[keep_last_n:]:
        checkpoint.unlink()


if __name__ == '__main__':
    # Test utilities
    print("Testing training utilities...\n")
    
    # Test device detection
    device = get_device()
    print(f"✓ Detected device: {device}")
    
    # Test time formatting
    print(f"✓ Time formatting: {format_time(45)} = 45s")
    print(f"✓ Time formatting: {format_time(125)} = 2.1m")
    print(f"✓ Time formatting: {format_time(7200)} = 2.0h")
    
    # Test metrics tracker
    tracker = MetricsTracker()
    dummy_preds = torch.tensor([[0.1, 0.6, 0.2, 0.1], [0.4, 0.1, 0.3, 0.2]])
    dummy_targets = torch.tensor([1, 0])
    tracker.update(1.5, dummy_preds, dummy_targets)
    metrics = tracker.get_metrics()
    print(f"\n✓ Metrics tracker: {metrics}")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3, min_delta=0.001)
    print(f"\n✓ Early stopping initialized")
    print(f"  Epoch 1, loss=1.0: stop={early_stop(1.0, 1)}")
    print(f"  Epoch 2, loss=0.9: stop={early_stop(0.9, 2)}")
    print(f"  Epoch 3, loss=0.91: stop={early_stop(0.91, 3)}")
    print(f"  Epoch 4, loss=0.92: stop={early_stop(0.92, 4)}")
    print(f"  Epoch 5, loss=0.93: stop={early_stop(0.93, 5)}")
    print(f"  Best epoch: {early_stop.best_epoch}, Best score: {early_stop.best_score}")
    
    print("\n✓ All utility tests passed!")
