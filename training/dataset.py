"""PyTorch Dataset for 2048 game training data."""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Game2048Dataset(Dataset):
    """Dataset for 2048 game training data.
    
    Loads game histories from JSONL file and provides (board, action) pairs.
    Supports data augmentation and train/val/test splits.
    """
    
    # Action mapping: direction name to integer
    ACTION_MAP = {
        'up': 0,
        'down': 1,
        'left': 2,
        'right': 3
    }
    
    ACTION_NAMES = ['up', 'down', 'left', 'right']
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        augment: bool = False,
        seed: int = 42
    ):
        """Initialize dataset.
        
        Args:
            data_path: Path to JSONL file with game histories
            split: 'train', 'val', or 'test'
            train_ratio: Proportion of games for training
            val_ratio: Proportion of games for validation
            test_ratio: Proportion of games for testing
            augment: Whether to apply data augmentation
            seed: Random seed for reproducible splits
        """
        self.data_path = Path(data_path)
        self.split = split
        self.augment = augment
        self.seed = seed
        
        # Validate split ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Load and parse data
        logger.info(f"Loading data from {self.data_path}")
        self.samples = self._load_and_parse()
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        
        if augment:
            from .augmentation import DataAugmentation
            self.augmentation = DataAugmentation()
            logger.info("Data augmentation enabled")
    
    def _load_and_parse(self) -> List[Dict]:
        """Load JSONL file and extract training samples.
        
        Returns:
            List of sample dictionaries with keys: board, action, score, game_id, move_number
        """
        games = []
        with open(self.data_path, 'r') as f:
            for line in f:
                game = json.loads(line.strip())
                games.append(game)
        
        logger.info(f"Loaded {len(games)} games")
        
        # Split games into train/val/test
        np.random.seed(self.seed)
        indices = np.random.permutation(len(games))
        
        n_train = int(len(games) * self.train_ratio)
        n_val = int(len(games) * self.val_ratio)
        
        if self.split == 'train':
            game_indices = indices[:n_train]
        elif self.split == 'val':
            game_indices = indices[n_train:n_train + n_val]
        elif self.split == 'test':
            game_indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Extract samples from selected games
        samples = []
        for game_idx in game_indices:
            game = games[game_idx]
            for move_idx, move in enumerate(game['moves']):
                # Extract board and action
                board = np.array(move['board'], dtype=np.float32)
                action_str = move['direction']
                
                # Skip if invalid action
                if action_str not in self.ACTION_MAP:
                    continue
                
                action = self.ACTION_MAP[action_str]
                score = move['score']
                
                samples.append({
                    'board': board,
                    'action': action,
                    'score': score,
                    'game_id': int(game_idx),
                    'move_number': move_idx
                })
        
        return samples
    
    def _encode_board(self, board: np.ndarray) -> np.ndarray:
        """Encode board using log2 normalization.
        
        Args:
            board: 4x4 numpy array of tile values
            
        Returns:
            Normalized board (values in [0, 1])
        """
        # Log2 normalization: log2(tile) / 11.0
        # 0 -> 0, 2 -> 1/11, 4 -> 2/11, ..., 2048 -> 11/11
        encoded = np.zeros_like(board, dtype=np.float32)
        mask = board > 0
        encoded[mask] = np.log2(board[mask]) / 11.0
        return encoded
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with keys:
                - board: (1, 4, 4) normalized board tensor
                - action: (,) action index tensor
                - score: (,) score tensor
                - game_id: (,) game ID tensor
                - move_number: (,) move number tensor
        """
        sample = self.samples[idx]
        
        board = sample['board'].copy()
        action = sample['action']
        
        # Apply data augmentation if enabled
        if self.augment:
            board, action = self.augmentation.augment(board, action)
        
        # Encode board
        board_encoded = self._encode_board(board)
        
        # Convert to tensors
        return {
            'board': torch.from_numpy(board_encoded).unsqueeze(0),  # (1, 4, 4)
            'action': torch.tensor(action, dtype=torch.long),
            'score': torch.tensor(sample['score'], dtype=torch.float32),
            'game_id': torch.tensor(sample['game_id'], dtype=torch.long),
            'move_number': torch.tensor(sample['move_number'], dtype=torch.long),
        }
    
    def get_action_distribution(self) -> Dict[str, int]:
        """Get distribution of actions in dataset.
        
        Returns:
            Dictionary mapping action names to counts
        """
        action_counts = {name: 0 for name in self.ACTION_NAMES}
        for sample in self.samples:
            action_name = self.ACTION_NAMES[sample['action']]
            action_counts[action_name] += 1
        return action_counts
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics.
        
        Returns:
            Dictionary with various statistics
        """
        scores = [s['score'] for s in self.samples]
        game_ids = set(s['game_id'] for s in self.samples)
        
        return {
            'num_samples': len(self.samples),
            'num_games': len(game_ids),
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'action_distribution': self.get_action_distribution(),
        }


def test_dataset():
    """Test dataset loading and processing."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Test loading
    dataset = Game2048Dataset(
        data_path='data/training_games.jsonl',
        split='train',
        augment=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Statistics: {dataset.get_statistics()}")
    
    # Test getting a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Board shape: {sample['board'].shape}")
    print(f"  Action: {sample['action']}")
    print(f"  Score: {sample['score']}")
    
    # Test with augmentation
    dataset_aug = Game2048Dataset(
        data_path='data/training_games.jsonl',
        split='train',
        augment=True
    )
    
    sample_aug = dataset_aug[0]
    print(f"\nAugmented sample 0:")
    print(f"  Board shape: {sample_aug['board'].shape}")
    print(f"  Action: {sample_aug['action']}")
    
    # Test validation split
    val_dataset = Game2048Dataset(
        data_path='data/training_games.jsonl',
        split='val'
    )
    print(f"\nValidation dataset size: {len(val_dataset)}")
    
    # Test test split
    test_dataset = Game2048Dataset(
        data_path='data/training_games.jsonl',
        split='test'
    )
    print(f"Test dataset size: {len(test_dataset)}")


if __name__ == '__main__':
    test_dataset()
