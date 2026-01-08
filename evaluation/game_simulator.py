"""
Game simulator for evaluating transformer model by playing complete 2048 games.

Computes:
- Win rate (reaching 2048 tile)
- Average score
- Average max tile
- Move distribution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from collections import defaultdict
import json
import argparse
from tqdm import tqdm

from models.transformer import TransformerPolicy
from training.utils import get_device


class Game2048:
    """2048 game environment."""
    
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.game_over = False
        self._add_random_tile()
        self._add_random_tile()
    
    def _add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell."""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            i, j = empty_cells[np.random.randint(len(empty_cells))]
            self.board[i, j] = 2 if np.random.random() < 0.9 else 4
    
    def _merge_line(self, line):
        """Merge a single line (left direction)."""
        # Remove zeros
        non_zero = line[line != 0]
        
        # Merge adjacent equal tiles
        merged = []
        skip = False
        score_gain = 0
        
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_value = non_zero[i] * 2
                merged.append(merged_value)
                score_gain += merged_value
                skip = True
            else:
                merged.append(non_zero[i])
        
        # Pad with zeros
        merged = np.array(merged + [0] * (4 - len(merged)), dtype=np.int32)
        
        return merged, score_gain
    
    def move(self, action):
        """
        Execute a move.
        
        Args:
            action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
            
        Returns:
            reward: Score gained from this move
            valid: Whether the move changed the board
        """
        if self.game_over:
            return 0, False
        
        old_board = self.board.copy()
        score_gain = 0
        
        if action == 0:  # UP
            for j in range(4):
                col = self.board[:, j]
                merged, gain = self._merge_line(col)
                self.board[:, j] = merged
                score_gain += gain
                
        elif action == 1:  # DOWN
            for j in range(4):
                col = self.board[::-1, j]
                merged, gain = self._merge_line(col)
                self.board[::-1, j] = merged
                score_gain += gain
                
        elif action == 2:  # LEFT
            for i in range(4):
                row = self.board[i, :]
                merged, gain = self._merge_line(row)
                self.board[i, :] = merged
                score_gain += gain
                
        elif action == 3:  # RIGHT
            for i in range(4):
                row = self.board[i, ::-1]
                merged, gain = self._merge_line(row)
                self.board[i, ::-1] = merged
                score_gain += gain
        
        # Check if board changed
        valid = not np.array_equal(old_board, self.board)
        
        if valid:
            self.score += score_gain
            self._add_random_tile()
            
            # Check game over
            if not self._has_valid_moves():
                self.game_over = True
        
        return score_gain, valid
    
    def _has_valid_moves(self):
        """Check if any valid moves exist."""
        # Check for empty cells
        if np.any(self.board == 0):
            return True
        
        # Check for adjacent equal tiles
        for i in range(4):
            for j in range(4):
                current = self.board[i, j]
                # Check right
                if j < 3 and self.board[i, j + 1] == current:
                    return True
                # Check down
                if i < 3 and self.board[i + 1, j] == current:
                    return True
        
        return False
    
    def get_max_tile(self):
        """Get the maximum tile value on the board."""
        return np.max(self.board)
    
    def get_state(self):
        """Get board state as tensor."""
        return torch.from_numpy(self.board.copy()).float()


def play_game(model, device, max_moves=10000, epsilon=0.0):
    """
    Play one complete game.
    
    Args:
        model: Trained transformer policy
        device: Torch device
        max_moves: Maximum number of moves before terminating
        epsilon: Probability of random action (for exploration)
        
    Returns:
        dict with game statistics
    """
    game = Game2048()
    model.eval()
    
    move_count = 0
    action_counts = defaultdict(int)
    invalid_count = 0
    
    with torch.no_grad():
        while not game.game_over and move_count < max_moves:
            # Get current state
            board = game.get_state().unsqueeze(0).to(device)  # (1, 4, 4)
            
            # Get action from model
            if np.random.random() < epsilon:
                # Random exploration
                action = np.random.randint(4)
            else:
                # Model prediction
                logits = model(board)
                action = torch.argmax(logits, dim=-1).item()
            
            # Execute move
            reward, valid = game.move(action)
            
            if valid:
                action_counts[action] += 1
                move_count += 1
            else:
                invalid_count += 1
                # If invalid, try other actions
                for alt_action in range(4):
                    if alt_action != action:
                        _, valid = game.move(alt_action)
                        if valid:
                            action_counts[alt_action] += 1
                            move_count += 1
                            break
    
    max_tile = game.get_max_tile()
    
    return {
        'score': game.score,
        'max_tile': max_tile,
        'moves': move_count,
        'won': max_tile >= 2048,
        'action_distribution': dict(action_counts),
        'invalid_moves': invalid_count
    }


def simulate_games(model, device, num_games=100, epsilon=0.0):
    """Simulate multiple games and collect statistics."""
    results = []
    
    for i in tqdm(range(num_games), desc="Playing games"):
        result = play_game(model, device, epsilon=epsilon)
        results.append(result)
    
    return results


def analyze_results(results):
    """Analyze game results and compute statistics."""
    stats = {
        'num_games': len(results),
        'win_rate': np.mean([r['won'] for r in results]),
        'avg_score': np.mean([r['score'] for r in results]),
        'std_score': np.std([r['score'] for r in results]),
        'max_score': np.max([r['score'] for r in results]),
        'min_score': np.min([r['score'] for r in results]),
        'avg_max_tile': np.mean([r['max_tile'] for r in results]),
        'avg_moves': np.mean([r['moves'] for r in results]),
        'avg_invalid_moves': np.mean([r['invalid_moves'] for r in results])
    }
    
    # Tile distribution
    max_tiles = [r['max_tile'] for r in results]
    tile_counts = defaultdict(int)
    for tile in max_tiles:
        tile_counts[int(tile)] += 1
    stats['max_tile_distribution'] = dict(sorted(tile_counts.items()))
    
    # Action distribution (aggregate across all games)
    total_actions = defaultdict(int)
    for r in results:
        for action, count in r['action_distribution'].items():
            total_actions[action] += count
    
    total_moves = sum(total_actions.values())
    action_dist = {
        int(action): count / total_moves if total_moves > 0 else 0
        for action, count in total_actions.items()
    }
    stats['action_distribution'] = action_dist
    
    return stats


def print_simulation_summary(stats):
    """Print simulation results."""
    print("\n" + "="*70)
    print("GAME SIMULATION RESULTS")
    print("="*70)
    
    print(f"\nOverall Performance:")
    print(f"  Games Played:    {stats['num_games']}")
    print(f"  Win Rate:        {stats['win_rate']:.2%} (reached 2048)")
    print(f"  Average Score:   {stats['avg_score']:.0f} ± {stats['std_score']:.0f}")
    print(f"  Max Score:       {stats['max_score']:.0f}")
    print(f"  Min Score:       {stats['min_score']:.0f}")
    
    print(f"\nGame Statistics:")
    print(f"  Avg Max Tile:    {stats['avg_max_tile']:.0f}")
    print(f"  Avg Moves:       {stats['avg_moves']:.1f}")
    print(f"  Avg Invalid:     {stats['avg_invalid_moves']:.1f}")
    
    print(f"\nMax Tile Distribution:")
    for tile, count in sorted(stats['max_tile_distribution'].items(), reverse=True):
        pct = count / stats['num_games'] * 100
        print(f"  {tile:>5}: {count:>4} games ({pct:>5.1f}%)")
    
    print(f"\nAction Distribution:")
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    for action_id in range(4):
        pct = stats['action_distribution'].get(action_id, 0) * 100
        print(f"  {action_names[action_id]:<6}: {pct:>5.1f}%")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Simulate 2048 games with trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/transformer/best_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of games to simulate"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Epsilon for random exploration (0.0 = greedy)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
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
    
    # Simulate games
    print(f"\nSimulating {args.num_games} games (epsilon={args.epsilon})...")
    results = simulate_games(model, device, num_games=args.num_games, epsilon=args.epsilon)
    
    # Analyze results
    stats = analyze_results(results)
    
    # Print summary
    print_simulation_summary(stats)
    
    # Save results
    output_file = output_dir / f"game_results_{args.num_games}games.json"
    with open(output_file, 'w') as f:
        save_data = {
            'statistics': stats,
            'individual_games': results,
            'config': {
                'checkpoint': str(args.checkpoint),
                'num_games': args.num_games,
                'epsilon': args.epsilon,
                'seed': args.seed
            }
        }
        json.dump(save_data, f, indent=2)
    
    print(f"✓ Results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
