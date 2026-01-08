#!/usr/bin/env python3
"""Quick test script to play 2048 games with trained model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from models.transformer import TransformerPolicy
from training.utils import get_device


class SimpleGame2048:
    """Simplified 2048 game for testing."""
    
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.game_over = False
        self._add_tile()
        self._add_tile()
    
    def _add_tile(self):
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            i, j = empty[np.random.randint(len(empty))]
            self.board[i, j] = 2 if np.random.random() < 0.9 else 4
    
    def _merge(self, line):
        nz = line[line != 0]
        merged, gain, skip = [], 0, False
        for i in range(len(nz)):
            if skip:
                skip = False
                continue
            if i + 1 < len(nz) and nz[i] == nz[i + 1]:
                v = nz[i] * 2
                merged.append(v)
                gain += v
                skip = True
            else:
                merged.append(nz[i])
        return np.array(merged + [0] * (4 - len(merged)), dtype=np.int32), gain
    
    def move(self, action):
        old = self.board.copy()
        gain = 0
        
        if action == 0:  # UP
            for j in range(4):
                self.board[:, j], g = self._merge(self.board[:, j])
                gain += g
        elif action == 1:  # DOWN
            for j in range(4):
                col_rev = self.board[::-1, j]
                merged, g = self._merge(col_rev)
                self.board[::-1, j] = merged
                gain += g
        elif action == 2:  # LEFT
            for i in range(4):
                self.board[i, :], g = self._merge(self.board[i, :])
                gain += g
        else:  # RIGHT
            for i in range(4):
                row_rev = self.board[i, ::-1]
                merged, g = self._merge(row_rev)
                self.board[i, ::-1] = merged
                gain += g
        
        valid = not np.array_equal(old, self.board)
        if valid:
            self.score += gain
            self._add_tile()
            if not self._can_move():
                self.game_over = True
        return valid
    
    def _can_move(self):
        if np.any(self.board == 0):
            return True
        for i in range(4):
            for j in range(4):
                if j < 3 and self.board[i, j] == self.board[i, j + 1]:
                    return True
                if i < 3 and self.board[i, j] == self.board[i + 1, j]:
                    return True
        return False


def main():
    # Load model
    device = get_device()
    print(f'Using device: {device}')
    print('Loading model...')
    
    checkpoint = torch.load('checkpoints/transformer/best_model.pth', 
                           map_location=device, weights_only=False)
    
    model = TransformerPolicy(
        embed_dim=checkpoint.get('embed_dim', 128),
        num_heads=checkpoint.get('num_heads', 8),
        num_layers=checkpoint.get('num_layers', 4),
        dim_feedforward=checkpoint.get('dim_feedforward', 512),
        dropout=checkpoint.get('dropout', 0.1),
        head_dropout=checkpoint.get('head_dropout', 0.2)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('✓ Model loaded successfully!\n')
    
    # Play games
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_games = 10
    results = []
    
    print(f'Playing {num_games} games...\n')
    
    for game_num in range(num_games):
        game = SimpleGame2048()
        moves = 0
        
        with torch.no_grad():
            while not game.game_over and moves < 10000:
                board_tensor = torch.from_numpy(game.board.copy()).float().unsqueeze(0).to(device)
                logits = model(board_tensor)
                action = torch.argmax(logits, dim=-1).item()
                
                if not game.move(action):
                    # Invalid move, try others
                    for alt_action in range(4):
                        if alt_action != action:
                            if game.move(alt_action):
                                break
                
                moves += 1
        
        max_tile = np.max(game.board)
        won = max_tile >= 2048
        
        results.append({
            'score': game.score,
            'max_tile': max_tile,
            'moves': moves,
            'won': won
        })
        
        status = '✓ WON' if won else '  ---'
        print(f'Game {game_num+1:2d} {status}: '
              f'Score={game.score:6d} | Max Tile={max_tile:4d} | Moves={moves:4d}')
    
    # Summary
    print('\n' + '='*70)
    print('GAME RESULTS SUMMARY')
    print('='*70)
    
    win_rate = sum(r['won'] for r in results) / len(results) * 100
    avg_score = np.mean([r['score'] for r in results])
    max_score = max(r['score'] for r in results)
    avg_moves = np.mean([r['moves'] for r in results])
    
    print(f'\nGames Played:    {len(results)}')
    print(f'Win Rate:        {win_rate:.1f}% (reached 2048 tile)')
    print(f'Average Score:   {avg_score:.0f}')
    print(f'Best Score:      {max_score}')
    print(f'Average Moves:   {avg_moves:.1f}')
    
    # Tile distribution
    tiles = [r['max_tile'] for r in results]
    print(f'\nMax Tile Distribution:')
    for tile_value in sorted(set(tiles), reverse=True):
        count = tiles.count(tile_value)
        pct = count / len(results) * 100
        print(f'  {tile_value:5d}: {count:2d} games ({pct:5.1f}%)')
    
    print('='*70)


if __name__ == '__main__':
    main()
