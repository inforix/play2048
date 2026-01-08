"""DataLoader utilities for training."""

import torch
from torch.utils.data import DataLoader
from .dataset import Game2048Dataset
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_dataloaders(
    data_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    augment_train: bool = True,
    seed: int = 42
) -> tuple:
    """Create train, validation, and test dataloaders.
    
    Args:
        data_path: Path to JSONL file with game histories
        batch_size: Batch size for all dataloaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory (faster GPU transfer)
        augment_train: Whether to apply augmentation to training data
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = Game2048Dataset(
        data_path=data_path,
        split='train',
        augment=augment_train,
        seed=seed
    )
    
    val_dataset = Game2048Dataset(
        data_path=data_path,
        split='val',
        augment=False,  # Never augment validation
        seed=seed
    )
    
    test_dataset = Game2048Dataset(
        data_path=data_path,
        split='test',
        augment=False,  # Never augment test
        seed=seed
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True  # Drop incomplete batches for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
        persistent_workers=max(1, num_workers // 2) > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
        persistent_workers=max(1, num_workers // 2) > 0
    )
    
    logger.info(f"Created dataloaders with batch_size={batch_size}, "
                f"num_workers={num_workers}")
    
    return train_loader, val_loader, test_loader


def test_dataloaders():
    """Test dataloader creation and iteration."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path='data/training_games.jsonl',
        batch_size=64,
        num_workers=0,  # Use 0 for testing
        augment_train=True
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test iteration
    print("\nTesting train loader iteration...")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Board shape: {batch['board'].shape}")
        print(f"  Action shape: {batch['action'].shape}")
        print(f"  Score shape: {batch['score'].shape}")
        
        # Verify shapes
        assert batch['board'].shape == (64, 1, 4, 4), "Unexpected board shape"
        assert batch['action'].shape == (64,), "Unexpected action shape"
        assert batch['score'].shape == (64,), "Unexpected score shape"
        
        if i >= 2:  # Test first 3 batches
            break
    
    print("\n✓ DataLoader test passed!")


if __name__ == '__main__':
    test_dataloaders()
