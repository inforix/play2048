"""
Self-play data generation for AlphaZero training.

Generates training data by playing games using MCTS-enhanced policy.
Includes temperature sampling, data augmentation, and parallel execution.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from collections import deque
import random
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import pickle


class ReplayBuffer:
    """
    Experience replay buffer for AlphaZero training.
    
    Stores (state, policy, value) tuples from self-play games.
    Supports sampling mini-batches for training.
    """
    
    def __init__(self, max_size: int = 500000):
        """
        Initialize replay buffer.
        
        Args:
            max_size: Maximum number of samples to store
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, examples: List[Dict]):
        """
        Add training examples to buffer.
        
        Args:
            examples: List of dicts with keys 'state', 'policy', 'value'
        """
        for example in examples:
            self.buffer.append(example)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample random mini-batch.
        
        Args:
            batch_size: Number of samples
            
        Returns:
            (states, policies, values) tuple of tensors
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        samples = random.sample(self.buffer, batch_size)
        
        states = torch.stack([s['state'] for s in samples])
        policies = torch.stack([s['policy'] for s in samples])
        values = torch.tensor([s['value'] for s in samples], dtype=torch.float32).unsqueeze(1)
        
        return states, policies, values
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def save(self, path: str):
        """Save buffer to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load(self, path: str):
        """Load buffer from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.buffer.clear()
            self.add(data)


class Game2048:
    """
    2048 game environment for self-play.
    Simplified version focused on core mechanics.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize game."""
        if seed is not None:
            np.random.seed(seed)
        
        self.board = np.zeros((4, 4), dtype=np.float32)
        self.score = 0
        self.game_over = False
        
        # Add initial tiles
        self._add_random_tile()
        self._add_random_tile()
    
    def reset(self) -> np.ndarray:
        """Reset game to initial state."""
        self.board = np.zeros((4, 4), dtype=np.float32)
        self.score = 0
        self.game_over = False
        self._add_random_tile()
        self._add_random_tile()
        return self.board.copy()
    
    def get_state(self) -> torch.Tensor:
        """Get current board state as tensor."""
        return torch.from_numpy(self.board.copy()).float()
    
    def _add_random_tile(self) -> bool:
        """Add a random tile (2 or 4) to empty position."""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            return False
        
        row, col = random.choice(empty_cells)
        self.board[row, col] = 2 if np.random.random() < 0.9 else 4
        return True
    
    def move(self, action: int) -> Tuple[bool, int]:
        """
        Execute move.
        
        Args:
            action: 0=up, 1=down, 2=left, 3=right
            
        Returns:
            (moved, reward) tuple
        """
        old_board = self.board.copy()
        reward = 0
        
        if action == 0:  # Up
            self.board, moved = self._slide_up(self.board)
        elif action == 1:  # Down
            self.board, moved = self._slide_down(self.board)
        elif action == 2:  # Left
            self.board, moved = self._slide_left(self.board)
        elif action == 3:  # Right
            self.board, moved = self._slide_right(self.board)
        else:
            return False, 0
        
        if moved:
            # Calculate reward (score increase)
            reward = int(np.sum(self.board) - np.sum(old_board))
            self.score += reward
            
            # Add random tile
            if not self._add_random_tile():
                self.game_over = True
            
            # Check if game over
            if not self._has_valid_moves():
                self.game_over = True
        
        return moved, reward
    
    def _slide_left(self, board: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Slide and merge tiles to the left."""
        new_board = board.copy()
        moved = False
        
        for i in range(4):
            row = new_board[i, :]
            non_zero = row[row != 0]
            
            # Merge adjacent equal tiles
            merged = []
            skip = False
            for j in range(len(non_zero)):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(non_zero) and non_zero[j] == non_zero[j + 1]:
                    merged.append(non_zero[j] * 2)
                    skip = True
                else:
                    merged.append(non_zero[j])
            
            # Pad with zeros
            new_row = np.array(merged + [0] * (4 - len(merged)), dtype=np.float32)
            
            if not np.array_equal(row, new_row):
                moved = True
            
            new_board[i, :] = new_row
        
        return new_board, moved
    
    def _slide_right(self, board: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Slide and merge tiles to the right."""
        flipped = np.fliplr(board)
        new_board, moved = self._slide_left(flipped)
        return np.fliplr(new_board), moved
    
    def _slide_up(self, board: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Slide and merge tiles up."""
        transposed = board.T
        new_board, moved = self._slide_left(transposed)
        return new_board.T, moved
    
    def _slide_down(self, board: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Slide and merge tiles down."""
        transposed = board.T
        new_board, moved = self._slide_right(transposed)
        return new_board.T, moved
    
    def _has_valid_moves(self) -> bool:
        """Check if any valid moves remain."""
        for action in range(4):
            test_board = self.board.copy()
            if action == 0:
                _, moved = self._slide_up(test_board)
            elif action == 1:
                _, moved = self._slide_down(test_board)
            elif action == 2:
                _, moved = self._slide_left(test_board)
            elif action == 3:
                _, moved = self._slide_right(test_board)
            
            if moved:
                return True
        return False
    
    def get_max_tile(self) -> int:
        """Get maximum tile value."""
        return int(np.max(self.board))


def augment_example(state: torch.Tensor, policy: np.ndarray) -> List[Dict]:
    """
    Apply 8-fold symmetry augmentation.
    
    Args:
        state: Board state (4, 4)
        policy: Action probabilities [up, down, left, right]
        
    Returns:
        List of 8 augmented examples
    """
    examples = []
    board = state.numpy()
    
    # Rotation and reflection combinations
    for rot in range(4):  # 0°, 90°, 180°, 270°
        for flip in [False, True]:  # No flip, horizontal flip
            # Transform board
            aug_board = np.rot90(board, k=rot)
            if flip:
                aug_board = np.fliplr(aug_board)
            
            # Transform policy (action mapping)
            aug_policy = transform_policy(policy, rot, flip)
            
            examples.append({
                'state': torch.from_numpy(aug_board.copy()).float(),
                'policy': torch.from_numpy(aug_policy).float(),
                'value': None  # Will be filled later
            })
    
    return examples


def transform_policy(policy: np.ndarray, rotation: int, flip: bool) -> np.ndarray:
    """
    Transform action probabilities for rotated/flipped board.
    
    Action mapping:
    - 0: up, 1: down, 2: left, 3: right
    
    Args:
        policy: Original policy [up, down, left, right]
        rotation: Number of 90° CW rotations (0-3)
        flip: Whether horizontally flipped
        
    Returns:
        Transformed policy
    """
    # Action indices: [up, down, left, right]
    actions = np.array(policy)
    
    # Apply rotation (each 90° CW rotation shifts actions)
    # up -> right -> down -> left -> up
    for _ in range(rotation):
        actions = np.array([actions[2], actions[3], actions[1], actions[0]])
    
    # Apply flip (left <-> right)
    if flip:
        actions = np.array([actions[0], actions[1], actions[3], actions[2]])
    
    return actions


def sample_action(policy: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample action from policy with temperature.
    
    Args:
        policy: Action probabilities
        temperature: Exploration parameter (0 = greedy, 1 = stochastic)
        
    Returns:
        Selected action
    """
    if temperature < 0.01:
        # Greedy selection
        return int(np.argmax(policy))
    
    # Temperature-adjusted probabilities
    adjusted = policy ** (1.0 / temperature)
    adjusted = adjusted / adjusted.sum()
    
    # Sample
    return int(np.random.choice(len(policy), p=adjusted))


def self_play_game(
    model,
    device: torch.device,
    mcts_simulations: int = 100,
    temperature_moves: int = 30,
    max_moves: int = 5000,
    add_noise: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Play one self-play game using MCTS.
    
    Args:
        model: Neural network
        device: Torch device
        mcts_simulations: Number of MCTS simulations per move
        temperature_moves: Use temperature=1.0 for first N moves
        max_moves: Maximum moves before terminating
        add_noise: Add Dirichlet noise for exploration
        
    Returns:
        (training_examples, game_stats) tuple
    """
    from training.mcts import MCTS
    
    game = Game2048()
    mcts = MCTS(model, device, num_simulations=mcts_simulations)
    
    training_examples = []
    move_count = 0
    
    while not game.game_over and move_count < max_moves:
        # Get current state
        state = game.get_state()
        
        # Run MCTS to get improved policy
        mcts_policy = mcts.search(state.numpy(), add_noise=add_noise and move_count < 5)
        
        # Store training example (value will be filled after game ends)
        training_examples.append({
            'state': state,
            'policy': torch.from_numpy(mcts_policy).float(),
            'value': None
        })
        
        # Sample action with temperature
        tau = 1.0 if move_count < temperature_moves else 0.1
        action = sample_action(mcts_policy, temperature=tau)
        
        # Execute move
        moved, reward = game.move(action)
        
        if not moved:
            # Invalid move - try alternatives
            for alt_action in range(4):
                if alt_action != action:
                    moved, reward = game.move(alt_action)
                    if moved:
                        break
        
        if moved:
            move_count += 1
        else:
            # No valid moves
            break
    
    # Calculate game result
    max_tile = game.get_max_tile()
    if max_tile >= 2048:
        result = 1.0
    elif max_tile >= 1024:
        result = 0.5
    elif max_tile >= 512:
        result = 0.0
    elif max_tile >= 256:
        result = -0.3
    else:
        result = -0.5
    
    # Fill in values for all examples
    for example in training_examples:
        example['value'] = result
    
    game_stats = {
        'score': game.score,
        'max_tile': max_tile,
        'moves': move_count,
        'won': max_tile >= 2048
    }
    
    return training_examples, game_stats


def generate_self_play_data(
    model,
    device: torch.device,
    num_games: int = 100,
    mcts_simulations: int = 100,
    augment: bool = True,
    verbose: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate self-play training data.
    
    Args:
        model: Neural network
        device: Torch device
        num_games: Number of games to play
        mcts_simulations: MCTS simulations per move
        augment: Apply 8-fold augmentation
        verbose: Show progress bar
        
    Returns:
        (training_examples, game_stats) tuple
    """
    all_examples = []
    all_stats = []
    
    iterator = tqdm(range(num_games), desc="Self-play") if verbose else range(num_games)
    
    for i in iterator:
        # Play one game
        examples, stats = self_play_game(
            model, 
            device, 
            mcts_simulations=mcts_simulations,
            add_noise=True
        )
        
        # Apply augmentation
        if augment:
            augmented = []
            for example in examples:
                aug_examples = augment_example(example['state'], example['policy'].numpy())
                for aug_ex in aug_examples:
                    aug_ex['value'] = example['value']
                    augmented.append(aug_ex)
            all_examples.extend(augmented)
        else:
            all_examples.extend(examples)
        
        all_stats.append(stats)
        
        # Update progress bar
        if verbose and isinstance(iterator, tqdm):
            avg_score = np.mean([s['score'] for s in all_stats])
            win_rate = np.mean([s['won'] for s in all_stats]) * 100
            iterator.set_postfix({
                'avg_score': f'{avg_score:.0f}',
                'win_rate': f'{win_rate:.1f}%'
            })
    
    return all_examples, all_stats


def test_self_play():
    """Test self-play implementation."""
    print("Testing self-play...")
    
    # Add parent directory to path for imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Create a simple dummy network
    from models.dual import AlphaZeroNetwork
    
    device = torch.device('cpu')
    model = AlphaZeroNetwork(num_blocks=2, channels=64)
    model.to(device)
    model.eval()
    
    print("\n1. Testing single game...")
    examples, stats = self_play_game(model, device, mcts_simulations=20, temperature_moves=10)
    print(f"  ✓ Generated {len(examples)} examples")
    print(f"  ✓ Score: {stats['score']}, Max tile: {stats['max_tile']}, Moves: {stats['moves']}")
    
    print("\n2. Testing augmentation...")
    aug_examples = augment_example(examples[0]['state'], examples[0]['policy'].numpy())
    print(f"  ✓ Generated {len(aug_examples)} augmented examples (8x)")
    
    print("\n3. Testing ReplayBuffer...")
    buffer = ReplayBuffer(max_size=1000)
    buffer.add(examples)
    print(f"  ✓ Buffer size: {len(buffer)}")
    
    states, policies, values = buffer.sample(batch_size=10)
    print(f"  ✓ Sampled batch: states {states.shape}, policies {policies.shape}, values {values.shape}")
    
    print("\n4. Testing batch generation...")
    examples, stats = generate_self_play_data(
        model, device, num_games=5, mcts_simulations=20, augment=True, verbose=True
    )
    print(f"  ✓ Generated {len(examples)} total examples from 5 games")
    print(f"  ✓ Average score: {np.mean([s['score'] for s in stats]):.0f}")
    
    print("\n✓ All self-play tests passed!")


if __name__ == "__main__":
    test_self_play()
