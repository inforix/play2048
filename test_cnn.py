#!/usr/bin/env python3
"""
Test script for CNN model on 2048 game.

This script evaluates a trained CNN model by playing games and collecting metrics.
"""

import torch
import argparse
import json
import logging
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cnn import CNNPolicy, DualCNNPolicy
from training.utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Game2048:
    """Simple 2048 game simulator for evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((4, 4), dtype=np.int32)
        self._add_random_tile()
        self._add_random_tile()
        self.score = 0
        self.game_over = False
        return self.board.copy()
    
    def _add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell."""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            i, j = empty_cells[np.random.randint(len(empty_cells))]
            self.board[i, j] = 2 if np.random.random() < 0.9 else 4
    
    def _merge_left(self, row):
        """Merge a row to the left and return (new_row, score_delta)."""
        # Remove zeros
        non_zero = row[row != 0]
        
        # Merge
        merged = []
        score = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged.append(non_zero[i] * 2)
                score += non_zero[i] * 2
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1
        
        # Pad with zeros
        merged = merged + [0] * (4 - len(merged))
        return np.array(merged, dtype=np.int32), score
    
    def move(self, action):
        """
        Execute a move.
        
        Args:
            action: 0=up, 1=down, 2=left, 3=right
            
        Returns:
            (reward, done, valid_move)
        """
        if self.game_over:
            return 0, True, False
        
        old_board = self.board.copy()
        score_delta = 0
        
        # Transform board based on action
        if action == 0:  # up
            self.board = self.board.T
            for i in range(4):
                self.board[i], delta = self._merge_left(self.board[i])
                score_delta += delta
            self.board = self.board.T
        elif action == 1:  # down
            self.board = self.board.T
            for i in range(4):
                self.board[i] = self.board[i][::-1]
                self.board[i], delta = self._merge_left(self.board[i])
                self.board[i] = self.board[i][::-1]
                score_delta += delta
            self.board = self.board.T
        elif action == 2:  # left
            for i in range(4):
                self.board[i], delta = self._merge_left(self.board[i])
                score_delta += delta
        else:  # right
            for i in range(4):
                self.board[i] = self.board[i][::-1]
                self.board[i], delta = self._merge_left(self.board[i])
                self.board[i] = self.board[i][::-1]
                score_delta += delta
        
        # Check if move was valid
        valid_move = not np.array_equal(old_board, self.board)
        
        if valid_move:
            self.score += score_delta
            self._add_random_tile()
            
            # Check if game is over
            if not self._has_valid_moves():
                self.game_over = True
        
        return score_delta, self.game_over, valid_move
    
    def _has_valid_moves(self):
        """Check if there are any valid moves left."""
        # Check for empty cells
        if np.any(self.board == 0):
            return True
        
        # Check for adjacent equal cells
        for i in range(4):
            for j in range(4):
                if j < 3 and self.board[i, j] == self.board[i, j + 1]:
                    return True
                if i < 3 and self.board[i, j] == self.board[i + 1, j]:
                    return True
        
        return False
    
    def get_max_tile(self):
        """Get the maximum tile value."""
        return np.max(self.board)


def normalize_board(board):
    """Normalize board for model input."""
    # Log2 transform
    board_norm = np.where(board > 0, np.log2(board), 0)
    # Normalize to [0, 1] range (assuming max tile is 2^17 = 131072)
    board_norm = board_norm / 17.0
    return board_norm.astype(np.float32)


def evaluate_model(model, num_games=100, device='cpu', use_greedy=True, verbose=False):
    """
    Evaluate model by playing games.
    
    Args:
        model: Trained CNN model
        num_games: Number of games to play
        device: Device to run model on
        use_greedy: If True, use greedy policy; otherwise sample from probabilities
        verbose: If True, print detailed game info
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    scores = []
    max_tiles = []
    move_counts = []
    
    for game_idx in tqdm(range(num_games), desc="Evaluating"):
        game = Game2048()
        board = game.reset()
        moves = 0
        invalid_moves = 0
        
        while not game.game_over and moves < 10000:
            # Prepare input
            board_norm = normalize_board(board)
            board_tensor = torch.from_numpy(board_norm).unsqueeze(0).unsqueeze(0).to(device)
            
            # Get action from model
            with torch.no_grad():
                if isinstance(model, DualCNNPolicy):
                    logits, value = model(board_tensor)
                else:
                    logits = model(board_tensor)
                
                probs = torch.softmax(logits, dim=-1)
                
                if use_greedy:
                    action = probs.argmax(dim=-1).item()
                else:
                    action = torch.multinomial(probs, 1).item()
            
            # Execute move
            reward, done, valid = game.move(action)
            
            if not valid:
                invalid_moves += 1
                # Try other actions
                for alt_action in range(4):
                    if alt_action != action:
                        _, _, valid = game.move(alt_action)
                        if valid:
                            break
            
            board = game.board
            moves += 1
        
        scores.append(game.score)
        max_tiles.append(game.get_max_tile())
        move_counts.append(moves)
        
        if verbose and (game_idx + 1) % 10 == 0:
            logger.info(f"Game {game_idx + 1}/{num_games}: Score={game.score}, Max Tile={game.get_max_tile()}, Moves={moves}")
    
    # Calculate statistics
    results = {
        'num_games': num_games,
        'scores': {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': int(np.min(scores)),
            'max': int(np.max(scores)),
            'median': float(np.median(scores))
        },
        'max_tiles': {
            'mean': float(np.mean(max_tiles)),
            'counts': {int(tile): int(np.sum(np.array(max_tiles) == tile)) for tile in np.unique(max_tiles)},
            'distribution': {
                '128': int(np.sum(np.array(max_tiles) >= 128)),
                '256': int(np.sum(np.array(max_tiles) >= 256)),
                '512': int(np.sum(np.array(max_tiles) >= 512)),
                '1024': int(np.sum(np.array(max_tiles) >= 1024)),
                '2048': int(np.sum(np.array(max_tiles) >= 2048)),
                '4096': int(np.sum(np.array(max_tiles) >= 4096)),
            }
        },
        'moves': {
            'mean': float(np.mean(move_counts)),
            'std': float(np.std(move_counts)),
            'min': int(np.min(move_counts)),
            'max': int(np.max(move_counts))
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate CNN model on 2048')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num-games', type=int, default=100,
                        help='Number of games to play')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results (JSON)')
    parser.add_argument('--greedy', action='store_true',
                        help='Use greedy policy (default: True)')
    parser.add_argument('--stochastic', dest='greedy', action='store_false',
                        help='Sample from action probabilities')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed game info')
    
    # Model architecture (should match training)
    parser.add_argument('--base-channels', type=int, default=128,
                        help='Number of base channels')
    parser.add_argument('--num-blocks', type=int, default=4,
                        help='Number of residual blocks')
    parser.add_argument('--dual-head', action='store_true',
                        help='Use dual-head model')
    
    parser.set_defaults(greedy=True)
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    if args.dual_head:
        model = DualCNNPolicy(
            base_channels=args.base_channels,
            num_blocks=args.num_blocks
        )
    else:
        model = CNNPolicy(
            base_channels=args.base_channels,
            num_blocks=args.num_blocks
        )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    logger.info(f"Model loaded successfully ({model.get_num_params():,} parameters)")
    
    # Evaluate
    logger.info(f"\nEvaluating on {args.num_games} games...")
    logger.info(f"Policy: {'Greedy' if args.greedy else 'Stochastic'}")
    logger.info("=" * 70)
    
    results = evaluate_model(
        model,
        num_games=args.num_games,
        device=device,
        use_greedy=args.greedy,
        verbose=args.verbose
    )
    
    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"\nScores:")
    logger.info(f"  Mean: {results['scores']['mean']:.1f} ± {results['scores']['std']:.1f}")
    logger.info(f"  Median: {results['scores']['median']:.1f}")
    logger.info(f"  Range: [{results['scores']['min']}, {results['scores']['max']}]")
    
    logger.info(f"\nMax Tiles:")
    for tile, count in sorted(results['max_tiles']['counts'].items(), reverse=True):
        logger.info(f"  {tile}: {count} games ({100 * count / args.num_games:.1f}%)")
    
    logger.info(f"\nMilestones:")
    for tile, count in sorted(results['max_tiles']['distribution'].items(), 
                              key=lambda x: int(x[0]), reverse=True):
        logger.info(f"  ≥{tile}: {count} games ({100 * count / args.num_games:.1f}%)")
    
    logger.info(f"\nMoves per game:")
    logger.info(f"  Mean: {results['moves']['mean']:.1f} ± {results['moves']['std']:.1f}")
    logger.info(f"  Range: [{results['moves']['min']}, {results['moves']['max']}]")
    
    logger.info("=" * 70)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
