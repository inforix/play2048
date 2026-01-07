#!/usr/bin/env python3
"""
2048 Dataset Generator using Expectimax Algorithm

Plays multiple games of 2048 using the Expectimax AI strategy and saves
the game histories to JSONL format for neural network training.

Usage:
    python generate_dataset.py --games 500 --output data/training_games.jsonl
    python generate_dataset.py --games 100 --depth 3 --output data/quick_test.jsonl
"""

import argparse
import json
import random
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import copy
from tqdm import tqdm


# ========== FAST BOARD OPERATIONS (for AI evaluation) ==========

def _process_line_fast(line: np.ndarray) -> Tuple[np.ndarray, int]:
    """Process a single line (merge and compact tiles). Returns (new_line, score_gain)."""
    non_zero = line[line != 0]
    merged = []
    score_gain = 0
    i = 0
    
    while i < len(non_zero):
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
            merged_value = non_zero[i] * 2
            merged.append(merged_value)
            score_gain += merged_value
            i += 2
        else:
            merged.append(non_zero[i])
            i += 1
    
    result = np.array(merged + [0] * (4 - len(merged)), dtype=np.int32)
    return result, score_gain


def simulate_move_fast(board: np.ndarray, direction: str) -> Tuple[np.ndarray, bool]:
    """Fast board move simulation without game state. Returns (new_board, moved)."""
    new_board = board.copy()
    moved = False
    
    if direction in ['left', 'right']:
        for row_idx in range(4):
            row = new_board[row_idx, :].copy()
            new_row, _ = _process_line_fast(row if direction == 'left' else row[::-1])
            
            if direction == 'right':
                new_row = new_row[::-1]
            
            if not np.array_equal(new_board[row_idx, :], new_row):
                new_board[row_idx, :] = new_row
                moved = True
    else:
        for col_idx in range(4):
            col = new_board[:, col_idx].copy()
            new_col, _ = _process_line_fast(col if direction == 'up' else col[::-1])
            
            if direction == 'down':
                new_col = new_col[::-1]
            
            if not np.array_equal(new_board[:, col_idx], new_col):
                new_board[:, col_idx] = new_col
                moved = True
    
    return new_board, moved


class Game2048:
    """2048 game logic implementation."""
    
    SIZE = 4
    
    def __init__(self):
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=np.int32)
        self.score = 0
        self.move_history = []
        self.game_over = False
        
        # Initialize with 2 random tiles
        self.add_random_tile()
        self.add_random_tile()
    
    def add_random_tile(self) -> bool:
        """Add a random 2 or 4 tile to an empty cell. Returns True if successful."""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            return False
        
        row, col = random.choice(empty_cells)
        self.board[row, col] = 2 if random.random() < 0.9 else 4
        return True
    
    def move(self, direction: str) -> bool:
        """
        Execute a move in the given direction.
        Returns True if the move changed the board state.
        """
        board_before = self.board.copy()
        score_before = self.score
        
        if direction in ['left', 'right']:
            moved = self._move_horizontal(direction)
        else:  # up or down
            moved = self._move_vertical(direction)
        
        if moved:
            # Record the move (board state BEFORE the move + direction)
            self.move_history.append({
                'board': board_before.tolist(),
                'direction': direction,
                'score': int(score_before)
            })
            
            # Add new tile after successful move
            self.add_random_tile()
            
            # Check if game is over
            if self.is_game_over():
                self.game_over = True
        
        return moved
    
    def _move_horizontal(self, direction: str) -> bool:
        """Move tiles horizontally (left or right)."""
        moved = False
        
        for row_idx in range(self.SIZE):
            row = self.board[row_idx].copy()
            new_row = self._process_line(row if direction == 'left' else row[::-1])
            
            if direction == 'right':
                new_row = new_row[::-1]
            
            if not np.array_equal(self.board[row_idx], new_row):
                self.board[row_idx] = new_row
                moved = True
        
        return moved
    
    def _move_vertical(self, direction: str) -> bool:
        """Move tiles vertically (up or down)."""
        moved = False
        
        for col_idx in range(self.SIZE):
            col = self.board[:, col_idx].copy()
            new_col = self._process_line(col if direction == 'up' else col[::-1])
            
            if direction == 'down':
                new_col = new_col[::-1]
            
            if not np.array_equal(self.board[:, col_idx], new_col):
                self.board[:, col_idx] = new_col
                moved = True
        
        return moved
    
    def _process_line(self, line: np.ndarray) -> np.ndarray:
        """Process a single line (merge and compact tiles)."""
        # Remove zeros
        non_zero = line[line != 0]
        
        # Merge adjacent equal tiles
        merged = []
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                # Merge tiles
                merged_value = non_zero[i] * 2
                merged.append(merged_value)
                self.score += merged_value
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1
        
        # Pad with zeros
        result = np.array(merged + [0] * (self.SIZE - len(merged)), dtype=np.int32)
        return result
    
    def is_move_valid(self, direction: str) -> bool:
        """Check if a move in the given direction would change the board."""
        board_copy = self.board.copy()
        score_copy = self.score
        
        if direction in ['left', 'right']:
            moved = self._move_horizontal(direction)
        else:
            moved = self._move_vertical(direction)
        
        # Restore state
        self.board = board_copy
        self.score = score_copy
        
        return moved
    
    def is_game_over(self) -> bool:
        """Check if no valid moves are available."""
        # Check if any cell is empty
        if np.any(self.board == 0):
            return False
        
        # Check for possible merges horizontally
        for row in range(self.SIZE):
            for col in range(self.SIZE - 1):
                if self.board[row, col] == self.board[row, col + 1]:
                    return False
        
        # Check for possible merges vertically
        for col in range(self.SIZE):
            for row in range(self.SIZE - 1):
                if self.board[row, col] == self.board[row + 1, col]:
                    return False
        
        return True
    
    def get_max_tile(self) -> int:
        """Get the maximum tile value on the board."""
        return int(np.max(self.board))
    
    def has_won(self) -> bool:
        """Check if 2048 tile is reached."""
        return self.get_max_tile() >= 2048
    
    def clone(self) -> 'Game2048':
        """Create a deep copy of the game state."""
        new_game = Game2048.__new__(Game2048)
        new_game.board = self.board.copy()
        new_game.score = self.score
        new_game.move_history = copy.deepcopy(self.move_history)
        new_game.game_over = self.game_over
        return new_game


class ExpectimaxAI:
    """Expectimax AI player for 2048."""
    
    # Snake pattern weight matrix (higher weights in top-right corner)
    WEIGHT_MATRIX = np.array([
        [4**15, 4**14, 4**13, 4**12],
        [4**8,  4**9,  4**10, 4**11],
        [4**7,  4**6,  4**5,  4**4],
        [4**0,  4**1,  4**2,  4**3]
    ], dtype=np.float64)
    
    def __init__(self, depth: int = 4, use_cache: bool = True):
        self.depth = depth
        self.use_cache = use_cache
        self.nodes_evaluated = 0
        self.cache_hits = 0
        self._eval_cache = {}  # Transposition table
    
    def get_best_move(self, game: Game2048) -> Optional[str]:
        """Get the best move using Expectimax algorithm."""
        self.nodes_evaluated = 0
        self.cache_hits = 0
        if self.use_cache:
            self._eval_cache.clear()  # Clear cache for new move decision
        
        directions = ['up', 'down', 'left', 'right']
        best_move = None
        best_score = -float('inf')
        
        board = game.board
        for direction in directions:
            new_board, moved = simulate_move_fast(board, direction)
            if not moved:
                continue
            
            # Evaluate the resulting state
            score = self._expectimax_value_fast(new_board, self.depth - 1, is_player_turn=False)
            
            if score > best_score:
                best_score = score
                best_move = direction
        
        return best_move
    
    def _expectimax_value_fast(self, board: np.ndarray, depth: int, is_player_turn: bool) -> float:
        """Optimized expectimax using board-only operations and caching."""
        self.nodes_evaluated += 1
        
        # Check cache
        if self.use_cache and depth > 0:
            cache_key = (board.tobytes(), depth, is_player_turn)
            if cache_key in self._eval_cache:
                self.cache_hits += 1
                return self._eval_cache[cache_key]
        
        # Terminal condition
        if depth == 0:
            result = self._evaluate_board_fast(board)
            return result
        
        if is_player_turn:
            # Maximize over possible moves
            max_score = -float('inf')
            any_moved = False
            
            for direction in ['up', 'down', 'left', 'right']:
                new_board, moved = simulate_move_fast(board, direction)
                if not moved:
                    continue
                
                any_moved = True
                score = self._expectimax_value_fast(new_board, depth - 1, is_player_turn=False)
                max_score = max(max_score, score)
            
            result = max_score if any_moved else self._evaluate_board_fast(board)
        else:
            # Chance node - expected value over random tile placements
            empty_cells = list(zip(*np.where(board == 0)))
            
            if not empty_cells:
                result = self._expectimax_value_fast(board, depth - 1, is_player_turn=True)
            else:
                # Sample a subset of empty cells for efficiency
                sample_size = min(len(empty_cells), 4)
                sampled_cells = random.sample(empty_cells, sample_size)
                
                expected_value = 0.0
                for row, col in sampled_cells:
                    # Try placing a 2 (90% probability)
                    board_2 = board.copy()
                    board_2[row, col] = 2
                    expected_value += 0.9 * self._expectimax_value_fast(board_2, depth - 1, is_player_turn=True)
                    
                    # Try placing a 4 (10% probability)
                    board_4 = board.copy()
                    board_4[row, col] = 4
                    expected_value += 0.1 * self._expectimax_value_fast(board_4, depth - 1, is_player_turn=True)
                
                result = expected_value / sample_size
        
        # Store in cache
        if self.use_cache and depth > 0:
            self._eval_cache[cache_key] = result
        
        return result
    
    def _evaluate_board_fast(self, board: np.ndarray) -> float:
        """
        Optimized board evaluation function.
        Based on the HTML implementation's evaluateBoardExpectimax.
        """
        # 1. Weighted position score (snake pattern) - vectorized
        score = np.sum(board * self.WEIGHT_MATRIX)
        
        # 2. Empty tiles bonus (critical for survival)
        empty_tiles = np.count_nonzero(board == 0)
        score += empty_tiles * 50000
        
        # 3. Monotonicity bonus
        monotonicity = self._calculate_monotonicity_fast(board)
        score += monotonicity * 10000
        
        # 4. Smoothness bonus
        smoothness = self._calculate_smoothness_fast(board)
        score += smoothness * 1000
        
        # 5. Max tile in corner bonus
        max_tile = np.max(board)
        if board[0, 3] == max_tile:  # Top-right corner
            score += max_tile * 10000
        
        return score
    
    def _calculate_monotonicity_fast(self, board: np.ndarray) -> float:
        """Optimized monotonicity calculation using vectorized operations."""
        mono = 0.0
        
        # Check rows - vectorized
        for row in range(4):
            row_data = board[row, :]
            increasing = np.sum(row_data[:-1] <= row_data[1:])
            decreasing = np.sum(row_data[:-1] >= row_data[1:])
            mono += max(increasing, decreasing)
        
        # Check columns - vectorized
        for col in range(4):
            col_data = board[:, col]
            increasing = np.sum(col_data[:-1] <= col_data[1:])
            decreasing = np.sum(col_data[:-1] >= col_data[1:])
            mono += max(increasing, decreasing)
        
        return mono
    
    def _calculate_smoothness_fast(self, board: np.ndarray) -> float:
        """Optimized smoothness calculation."""
        smoothness = 0.0
        
        # Pre-compute log values for non-zero tiles
        log_board = np.zeros_like(board, dtype=np.float64)
        mask = board > 0
        log_board[mask] = np.log2(board[mask])
        
        # Check horizontal neighbors
        for row in range(4):
            for col in range(3):
                if board[row, col] > 0 and board[row, col + 1] > 0:
                    smoothness -= abs(log_board[row, col] - log_board[row, col + 1])
        
        # Check vertical neighbors
        for row in range(3):
            for col in range(4):
                if board[row, col] > 0 and board[row + 1, col] > 0:
                    smoothness -= abs(log_board[row, col] - log_board[row + 1, col])
        
        return smoothness


def play_game(ai: ExpectimaxAI, verbose: bool = False) -> Dict:
    """
    Play a single game using the AI.
    Returns game statistics and move history.
    """
    game = Game2048()
    move_count = 0
    total_nodes = 0
    total_cache_hits = 0
    
    while not game.game_over and move_count < 5000:  # Safety limit
        best_move = ai.get_best_move(game)
        
        if best_move is None:
            # No valid moves
            break
        
        total_nodes += ai.nodes_evaluated
        total_cache_hits += ai.cache_hits
        
        success = game.move(best_move)
        if success:
            move_count += 1
            
            if verbose and move_count % 50 == 0:
                cache_rate = (total_cache_hits / total_nodes * 100) if total_nodes > 0 else 0
                print(f"Move {move_count}: Score {game.score}, Max tile {game.get_max_tile()}, "
                      f"Cache hit rate: {cache_rate:.1f}%")
    
    if verbose:
        cache_rate = (total_cache_hits / total_nodes * 100) if total_nodes > 0 else 0
        print(f"Game finished: {total_nodes} nodes evaluated, {total_cache_hits} cache hits ({cache_rate:.1f}%)")
    
    return {
        'totalMoves': len(game.move_history),
        'finalScore': int(game.score),
        'maxTile': int(game.get_max_tile()),
        'won': bool(game.has_won()),
        'finalBoard': game.board.tolist(),
        'moves': game.move_history,
        'timestamp': datetime.now().isoformat()
    }


def save_games_jsonl(games: List[Dict], output_path: str):
    """Save games to JSONL format (one JSON object per line)."""
    with open(output_path, 'w') as f:
        for game in games:
            json.dump(game, f)
            f.write('\n')


def save_games_json(games: List[Dict], output_path: str):
    """Save games to JSON format (array of games)."""
    with open(output_path, 'w') as f:
        json.dump(games, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Generate 2048 training dataset using Expectimax AI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--games', '-g',
        type=int,
        default=500,
        help='Number of games to play'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/training_games.jsonl',
        help='Output file path (.jsonl or .json)'
    )
    
    parser.add_argument(
        '--depth', '-d',
        type=int,
        default=4,
        help='Expectimax search depth (higher = better but slower)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print progress for each game'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Create output directory if needed
    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    print(f"🎮 2048 Dataset Generator")
    print(f"=" * 60)
    print(f"Games to play: {args.games}")
    print(f"Search depth: {args.depth}")
    print(f"Output: {args.output}")
    print(f"=" * 60)
    print()
    
    # Initialize AI
    ai = ExpectimaxAI(depth=args.depth)
    
    # Play games
    games = []
    stats = {
        'total_games': args.games,
        'wins': 0,
        'total_score': 0,
        'total_moves': 0,
        'max_tiles': []
    }
    
    for i in tqdm(range(args.games), desc="Playing games"):
        if args.verbose:
            print(f"\n🎯 Game {i + 1}/{args.games}")
        
        game_data = play_game(ai, verbose=args.verbose)
        games.append(game_data)
        
        # Update statistics
        if game_data['won']:
            stats['wins'] += 1
        stats['total_score'] += game_data['finalScore']
        stats['total_moves'] += game_data['totalMoves']
        stats['max_tiles'].append(game_data['maxTile'])
        
        if args.verbose:
            print(f"✓ Finished: Score {game_data['finalScore']}, "
                  f"Max tile {game_data['maxTile']}, "
                  f"Moves {game_data['totalMoves']}")
    
    # Save dataset
    print(f"\n💾 Saving dataset to {args.output}...")
    if args.output.endswith('.jsonl'):
        save_games_jsonl(games, args.output)
    else:
        save_games_json(games, args.output)
    
    # Print statistics
    print(f"\n📊 Dataset Statistics")
    print(f"=" * 60)
    print(f"Total games: {stats['total_games']}")
    print(f"Wins (2048+): {stats['wins']} ({stats['wins']/stats['total_games']*100:.1f}%)")
    print(f"Average score: {stats['total_score']/stats['total_games']:.0f}")
    print(f"Average moves per game: {stats['total_moves']/stats['total_games']:.1f}")
    print(f"Max tile achieved: {max(stats['max_tiles'])}")
    
    # Max tile distribution
    from collections import Counter
    tile_counts = Counter(stats['max_tiles'])
    print(f"\nMax tile distribution:")
    for tile in sorted(tile_counts.keys(), reverse=True):
        count = tile_counts[tile]
        pct = count / stats['total_games'] * 100
        print(f"  {tile:5d}: {count:4d} games ({pct:5.1f}%)")
    
    # Calculate total training samples (with moves)
    total_samples = sum(len(g['moves']) for g in games)
    print(f"\nTotal training samples: {total_samples:,}")
    print(f"Average samples per game: {total_samples/stats['total_games']:.0f}")
    
    print(f"\n✅ Dataset generation complete!")
    print(f"Output saved to: {args.output}")


if __name__ == '__main__':
    main()
