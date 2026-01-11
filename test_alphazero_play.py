#!/usr/bin/env python3
"""
Test AlphaZero model on 2048 game for 100 episodes.
Collects win rate, max tile, score, moves, etc.
"""
from pathlib import Path
import numpy as np
from tqdm import trange
import torch

from models.dual.alphazero_network import AlphaZeroNetwork
from training.mcts import MCTS
from training.selfplay import Game2048
from training.utils import get_device

CHECKPOINT_PATH = Path("checkpoints/alphazero/final_model.pth")
NUM_GAMES = 100
MCTS_SIMULATIONS = 200
MAX_MOVES = 10000

device = torch.device(get_device())

# Load model
model = AlphaZeroNetwork(num_blocks=4, channels=256)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["best_model"])
model.to(device)
model.eval()

results = []

for i in trange(NUM_GAMES, desc="Testing AlphaZero on 2048"):
    game = Game2048()
    mcts = MCTS(model, device, num_simulations=MCTS_SIMULATIONS)
    moves = 0
    while not game.game_over and moves < MAX_MOVES:
        # Use MCTS to select action
        state = game.get_state().numpy()
        policy = mcts.search(state, add_noise=False)
        action = int(np.argmax(policy))

        moved, reward = game.move(action)
        if not moved:
            # Try alternative actions if the top choice was invalid
            for alt_action in range(4):
                if alt_action == action:
                    continue
                moved, reward = game.move(alt_action)
                if moved:
                    break

        if moved:
            moves += 1
        else:
            # No valid moves remain
            break

    max_tile = game.get_max_tile()
    score = game.score
    win = max_tile >= 2048
    results.append({
        "win": win,
        "max_tile": max_tile,
        "score": score,
        "moves": moves
    })

# Statistics
win_rate = np.mean([r["win"] for r in results])
avg_max_tile = np.mean([r["max_tile"] for r in results])
avg_score = np.mean([r["score"] for r in results])
avg_moves = np.mean([r["moves"] for r in results])
max_tile_hist = {}
for r in results:
    t = r["max_tile"]
    max_tile_hist[t] = max_tile_hist.get(t, 0) + 1

print("\nAlphaZero 2048 Test Results ({} games):".format(NUM_GAMES))
print("Win rate (2048 reached): {:.2f}%".format(win_rate * 100))
print("Average max tile: {:.2f}".format(avg_max_tile))
print("Average score: {:.2f}".format(avg_score))
print("Average moves: {:.2f}".format(avg_moves))
print("Max tile distribution:")
for tile, count in sorted(max_tile_hist.items()):
    print(f"  Tile {tile}: {count} times")
