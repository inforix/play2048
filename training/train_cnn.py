"""
Training script for CNN-based 2048 AI.

This script implements the complete training loop with:
- SGD/Adam optimizer with weight decay
- Learning rate warmup and cosine annealing
- Early stopping
- Checkpointing
- TensorBoard logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import time
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cnn import CNNPolicy, DualCNNPolicy
from training.dataloader import create_dataloaders
from training.utils import (
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    MetricsTracker,
    get_device,
    cleanup_old_checkpoints,
    format_time,
    get_model_summary
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: str,
    metrics_tracker: MetricsTracker,
    epoch: int,
    total_epochs: int,
    gradient_clip: float = 1.0
) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        metrics_tracker: Metrics tracker
        epoch: Current epoch number
        total_epochs: Total number of epochs
        gradient_clip: Maximum gradient norm (0 to disable)
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    metrics_tracker.reset()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Train]')
    
    for batch in pbar:
        # Move data to device
        boards = batch['board'].to(device)
        actions = batch['action'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(boards)
        loss = criterion(logits, actions)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        
        optimizer.step()
        
        # Update metrics
        metrics_tracker.update(loss.item(), logits.detach(), actions)
        
        # Update progress bar
        current_metrics = metrics_tracker.get_metrics()
        pbar.set_postfix({
            'loss': f"{current_metrics['loss']:.4f}",
            'acc': f"{current_metrics['accuracy']:.1f}%"
        })
    
    return metrics_tracker.get_metrics()


def validate(
    model: nn.Module,
    dataloader,
    criterion,
    device: str,
    metrics_tracker: MetricsTracker,
    epoch: int,
    total_epochs: int
) -> dict:
    """
    Validate the model.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        metrics_tracker: Metrics tracker
        epoch: Current epoch number
        total_epochs: Total number of epochs
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    metrics_tracker.reset()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Val]')
    
    with torch.no_grad():
        for batch in pbar:
            # Move data to device
            boards = batch['board'].to(device)
            actions = batch['action'].to(device)
            
            # Forward pass
            logits = model(boards)
            loss = criterion(logits, actions)
            
            # Update metrics
            metrics_tracker.update(loss.item(), logits, actions)
            
            # Update progress bar
            current_metrics = metrics_tracker.get_metrics()
            pbar.set_postfix({
                'loss': f"{current_metrics['loss']:.4f}",
                'acc': f"{current_metrics['accuracy']:.1f}%"
            })
    
    return metrics_tracker.get_metrics()


def train(args):
    """Main training function."""
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Create output directories
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=not args.no_augment,
        seed=args.seed
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("Creating model...")
    if args.dual_head:
        model = DualCNNPolicy(
            base_channels=args.base_channels,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            head_dropout=args.head_dropout
        )
    else:
        model = CNNPolicy(
            base_channels=args.base_channels,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            head_dropout=args.head_dropout
        )
    
    model = model.to(device)
    logger.info(get_model_summary(model))
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True
        )
    
    # Learning rate scheduler with warmup
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=args.warmup_epochs
    )
    
    if args.scheduler == 'cosine':
        main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,
            T_mult=2,
            eta_min=1e-6
        )
    elif args.scheduler == 'step':
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.5
        )
    else:  # plateau
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=0.0001,
        mode='min'
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Metrics trackers
    train_tracker = MetricsTracker()
    val_tracker = MetricsTracker()
    lr_history = []
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint_info = load_checkpoint(
            Path(args.resume),
            model,
            optimizer,
            warmup_scheduler if start_epoch < args.warmup_epochs else main_scheduler,
            device
        )
        start_epoch = checkpoint_info['epoch'] + 1
        best_val_loss = checkpoint_info['val_loss']
    
    # Training loop
    logger.info(f"\nStarting training for {args.epochs} epochs...")
    logger.info("=" * 70)
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, train_tracker, epoch + 1, args.epochs,
            gradient_clip=args.gradient_clip
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion,
            device, val_tracker, epoch + 1, args.epochs
        )
        
        # Learning rate step
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
            current_lr = warmup_scheduler.get_last_lr()[0]
        else:
            if args.scheduler == 'plateau':
                main_scheduler.step(val_metrics['loss'])
                current_lr = optimizer.param_groups[0]['lr']
            else:
                main_scheduler.step()
                current_lr = main_scheduler.get_last_lr()[0]
        
        # Log metrics
        train_tracker.log_epoch(epoch + 1, 'train')
        val_tracker.log_epoch(epoch + 1, 'val')
        lr_history.append(current_lr)
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch + 1)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch + 1)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch + 1)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch + 1)
        writer.add_scalar('Accuracy/train_top2', train_metrics['top2_accuracy'], epoch + 1)
        writer.add_scalar('Accuracy/val_top2', val_metrics['top2_accuracy'], epoch + 1)
        writer.add_scalar('LearningRate', current_lr, epoch + 1)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time
        eta = (args.epochs - epoch - 1) * epoch_time
        
        logger.info(f"\nEpoch [{epoch + 1}/{args.epochs}] - {format_time(epoch_time)}")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f} | "
                   f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                   f"Train Top-2: {train_metrics['top2_accuracy']:.2f}%")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f} | "
                   f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                   f"Val Top-2: {val_metrics['top2_accuracy']:.2f}%")
        logger.info(f"  LR: {current_lr:.6f} | "
                   f"Time: {format_time(total_time)} | "
                   f"ETA: {format_time(eta)}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_checkpoint_path = checkpoint_dir / 'best_model.pth'
            save_checkpoint(
                model, optimizer,
                warmup_scheduler if epoch < args.warmup_epochs else main_scheduler,
                epoch, train_metrics['loss'], val_metrics['loss'], val_metrics['accuracy'],
                best_checkpoint_path,
                hyperparameters=vars(args)
            )
            logger.info(f"  ✓ Best model saved! (val_loss: {val_metrics['loss']:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch{epoch + 1}.pth'
            save_checkpoint(
                model, optimizer,
                warmup_scheduler if epoch < args.warmup_epochs else main_scheduler,
                epoch, train_metrics['loss'], val_metrics['loss'], val_metrics['accuracy'],
                checkpoint_path,
                hyperparameters=vars(args)
            )
            logger.info(f"  ✓ Checkpoint saved: {checkpoint_path.name}")
            
            # Cleanup old checkpoints
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3)
        
        # Early stopping check
        if early_stopping(val_metrics['loss'], epoch + 1):
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Early stopping triggered! No improvement for {args.early_stop_patience} epochs.")
            logger.info(f"Best epoch: {early_stopping.best_epoch}")
            logger.info(f"Best val loss: {early_stopping.best_score:.4f}")
            logger.info(f"{'=' * 70}")
            break
    
    # Training complete
    total_training_time = time.time() - training_start_time
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Training complete!")
    logger.info(f"Total time: {format_time(total_training_time)}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"{'=' * 70}")
    
    # Save training history
    history_path = log_dir / 'training_history.json'
    combined_history = {**train_tracker.history, **val_tracker.history}
    combined_history['learning_rate'] = lr_history
    with open(history_path, 'w') as f:
        json.dump(dict(combined_history), f, indent=2)
    logger.info(f"Training history saved to: {history_path}")
    
    # Close writer
    writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train CNN for 2048')
    
    # Data arguments
    parser.add_argument('--data', type=str, default='data/training_games.jsonl',
                        help='Path to training data (JSONL file)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    
    # Model arguments
    parser.add_argument('--base-channels', type=int, default=128,
                        help='Number of base channels in CNN')
    parser.add_argument('--num-blocks', type=int, default=4,
                        help='Number of residual blocks')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate in residual blocks')
    parser.add_argument('--head-dropout', type=float, default=0.3,
                        help='Dropout rate in policy head')
    parser.add_argument('--dual-head', action='store_true',
                        help='Use dual-head model (policy + value)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau'],
                        help='Learning rate scheduler type')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Maximum gradient norm (0 to disable)')
    parser.add_argument('--early-stop-patience', type=int, default=25,
                        help='Early stopping patience')
    
    # Checkpointing arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/cnn',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint-freq', type=int, default=25,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Logging arguments
    parser.add_argument('--log-dir', type=str, default='results/training_curves/cnn',
                        help='Directory for TensorBoard logs')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (small dataset, few epochs)')
    
    args = parser.parse_args()
    
    # Debug mode overrides
    if args.debug:
        args.epochs = 5
        args.batch_size = 8
        args.num_workers = 0
        args.warmup_epochs = 2
        args.checkpoint_freq = 2
        logger.info("DEBUG MODE: Using small dataset and few epochs")
    
    train(args)


if __name__ == '__main__':
    main()
