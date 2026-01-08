"""Data augmentation for 2048 game boards.

Implements 8-fold symmetry: 4 rotations × 2 reflections.
Actions are consistently remapped according to transformations.
"""

import numpy as np
from typing import Tuple
import random


class DataAugmentation:
    """Data augmentation with 8-fold symmetry for 2048 boards."""
    
    # Action indices: 0=up, 1=down, 2=left, 3=right
    
    # Action remapping for each transformation
    # Format: new_action = REMAP[transformation][old_action]
    REMAP = {
        'identity': [0, 1, 2, 3],  # No change
        'rot90': [3, 2, 0, 1],      # up->right, down->left, left->up, right->down
        'rot180': [1, 0, 3, 2],     # up->down, down->up, left->right, right->left
        'rot270': [2, 3, 1, 0],     # up->left, down->right, left->down, right->up
        'fliph': [0, 1, 3, 2],      # up->up, down->down, left->right, right->left
        'flipv': [1, 0, 2, 3],      # up->down, down->up, left->left, right->right
        'flipd1': [2, 3, 0, 1],     # up->left, down->right, left->up, right->down (main diagonal)
        'flipd2': [3, 2, 1, 0],     # up->right, down->left, left->down, right->up (anti-diagonal)
    }
    
    TRANSFORMATIONS = list(REMAP.keys())
    
    def __init__(self, prob: float = 0.5):
        """Initialize augmentation.
        
        Args:
            prob: Probability of applying augmentation (default: 0.5)
        """
        self.prob = prob
    
    def rotate_90(self, board: np.ndarray) -> np.ndarray:
        """Rotate board 90 degrees clockwise."""
        return np.rot90(board, k=-1)
    
    def rotate_180(self, board: np.ndarray) -> np.ndarray:
        """Rotate board 180 degrees."""
        return np.rot90(board, k=2)
    
    def rotate_270(self, board: np.ndarray) -> np.ndarray:
        """Rotate board 270 degrees clockwise (90 counter-clockwise)."""
        return np.rot90(board, k=1)
    
    def flip_horizontal(self, board: np.ndarray) -> np.ndarray:
        """Flip board horizontally (left-right)."""
        return np.fliplr(board)
    
    def flip_vertical(self, board: np.ndarray) -> np.ndarray:
        """Flip board vertically (up-down)."""
        return np.flipud(board)
    
    def flip_diagonal_main(self, board: np.ndarray) -> np.ndarray:
        """Flip along main diagonal (top-left to bottom-right)."""
        return board.T
    
    def flip_diagonal_anti(self, board: np.ndarray) -> np.ndarray:
        """Flip along anti-diagonal (top-right to bottom-left)."""
        return np.fliplr(board.T)
    
    def apply_transformation(
        self,
        board: np.ndarray,
        action: int,
        transform: str
    ) -> Tuple[np.ndarray, int]:
        """Apply a specific transformation to board and action.
        
        Args:
            board: 4x4 board array
            action: Action index (0-3)
            transform: Transformation name
            
        Returns:
            Tuple of (transformed_board, transformed_action)
        """
        # Apply board transformation
        if transform == 'identity':
            transformed_board = board.copy()
        elif transform == 'rot90':
            transformed_board = self.rotate_90(board)
        elif transform == 'rot180':
            transformed_board = self.rotate_180(board)
        elif transform == 'rot270':
            transformed_board = self.rotate_270(board)
        elif transform == 'fliph':
            transformed_board = self.flip_horizontal(board)
        elif transform == 'flipv':
            transformed_board = self.flip_vertical(board)
        elif transform == 'flipd1':
            transformed_board = self.flip_diagonal_main(board)
        elif transform == 'flipd2':
            transformed_board = self.flip_diagonal_anti(board)
        else:
            raise ValueError(f"Unknown transformation: {transform}")
        
        # Remap action
        transformed_action = self.REMAP[transform][action]
        
        return transformed_board, transformed_action
    
    def augment(
        self,
        board: np.ndarray,
        action: int
    ) -> Tuple[np.ndarray, int]:
        """Apply random augmentation with probability self.prob.
        
        Args:
            board: 4x4 board array
            action: Action index (0-3)
            
        Returns:
            Tuple of (augmented_board, augmented_action)
        """
        if random.random() < self.prob:
            # Randomly choose a transformation (excluding identity)
            transform = random.choice(self.TRANSFORMATIONS[1:])
            return self.apply_transformation(board, action, transform)
        else:
            return board.copy(), action
    
    def get_all_augmentations(
        self,
        board: np.ndarray,
        action: int
    ) -> list:
        """Get all 8 augmented versions.
        
        Args:
            board: 4x4 board array
            action: Action index (0-3)
            
        Returns:
            List of 8 tuples (augmented_board, augmented_action)
        """
        augmentations = []
        for transform in self.TRANSFORMATIONS:
            aug_board, aug_action = self.apply_transformation(board, action, transform)
            augmentations.append((aug_board, aug_action))
        return augmentations


def augment_sample(board: np.ndarray, action: int, prob: float = 0.5) -> Tuple[np.ndarray, int]:
    """Convenience function to augment a single sample.
    
    Args:
        board: 4x4 board array
        action: Action index (0-3)
        prob: Probability of applying augmentation
        
    Returns:
        Tuple of (augmented_board, augmented_action)
    """
    aug = DataAugmentation(prob=prob)
    return aug.augment(board, action)


def visualize_augmentations():
    """Visualize all 8 augmentations of a sample board."""
    # Create a test board with distinct values
    board = np.array([
        [2, 4, 8, 16],
        [32, 64, 128, 256],
        [512, 1024, 2048, 4096],
        [0, 0, 0, 2]
    ], dtype=np.float32)
    
    action = 0  # up
    action_names = ['up', 'down', 'left', 'right']
    
    aug = DataAugmentation()
    
    print("Original board:")
    print(board.astype(int))
    print(f"Original action: {action_names[action]}\n")
    
    print("All augmentations:")
    print("=" * 60)
    
    for transform in aug.TRANSFORMATIONS:
        aug_board, aug_action = aug.apply_transformation(board, action, transform)
        print(f"\n{transform}:")
        print(aug_board.astype(int))
        print(f"Action: {action_names[action]} -> {action_names[aug_action]}")
        print("-" * 40)


def test_augmentation():
    """Test augmentation correctness."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Test board
    board = np.array([
        [2, 4, 8, 16],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.float32)
    
    action = 0  # up
    action_names = ['up', 'down', 'left', 'right']
    
    aug = DataAugmentation(prob=1.0)
    
    # Test identity
    board_id, action_id = aug.apply_transformation(board, action, 'identity')
    assert np.array_equal(board, board_id), "Identity transformation failed"
    assert action == action_id, "Identity action remap failed"
    print("✓ Identity transformation correct")
    
    # Test rot90
    board_rot90, action_rot90 = aug.apply_transformation(board, action, 'rot90')
    expected_rot90 = np.array([
        [0, 0, 0, 2],
        [0, 0, 0, 4],
        [0, 0, 0, 8],
        [0, 0, 0, 16]
    ], dtype=np.float32)
    assert np.array_equal(board_rot90, expected_rot90), "Rot90 transformation failed"
    assert action_rot90 == 3, f"Rot90 action should be right (3), got {action_rot90}"
    print(f"✓ Rot90: up -> {action_names[action_rot90]}")
    
    # Test rot180
    board_rot180, action_rot180 = aug.apply_transformation(board, action, 'rot180')
    expected_rot180 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [16, 8, 4, 2]
    ], dtype=np.float32)
    assert np.array_equal(board_rot180, expected_rot180), "Rot180 transformation failed"
    assert action_rot180 == 1, f"Rot180 action should be down (1), got {action_rot180}"
    print(f"✓ Rot180: up -> {action_names[action_rot180]}")
    
    # Test fliph
    board_fliph, action_fliph = aug.apply_transformation(board, action, 'fliph')
    expected_fliph = np.array([
        [16, 8, 4, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.float32)
    assert np.array_equal(board_fliph, expected_fliph), "Fliph transformation failed"
    assert action_fliph == 0, f"Fliph action should stay up (0), got {action_fliph}"
    print(f"✓ Fliph: up -> {action_names[action_fliph]}")
    
    # Test random augmentation
    aug_random = DataAugmentation(prob=1.0)
    board_aug, action_aug = aug_random.augment(board, action)
    print(f"✓ Random augmentation: up -> {action_names[action_aug]}")
    
    print("\n✓ All augmentation tests passed!")
    
    # Visualize
    print("\n" + "=" * 60)
    visualize_augmentations()


if __name__ == '__main__':
    test_augmentation()
