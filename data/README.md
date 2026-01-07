# Data Directory

Storage for all dataset files in various processing stages.

## Structure

```
data/
├── raw/                    # Original generated game histories
│   ├── training_games.jsonl
│   ├── validation_games.jsonl
│   └── test_games.jsonl
│
├── processed/              # Preprocessed PyTorch tensors
│   ├── train_boards.pt
│   ├── train_actions.pt
│   └── ...
│
└── augmented/              # 8x augmented dataset (rotations + reflections)
    └── augmented_train.pt
```

## Data Format

### Raw Data (JSONL)
Each line is a complete game:
```json
{
  "totalMoves": 167,
  "finalScore": 16234,
  "maxTile": 2048,
  "won": true,
  "moves": [
    {"board": [[...]], "direction": "up", "score": 0},
    ...
  ]
}
```

### Processed Data (PyTorch Tensors)
- `boards`: torch.FloatTensor of shape (N, 1, 4, 4)
- `actions`: torch.LongTensor of shape (N,)
- `scores`: torch.FloatTensor of shape (N,)

## Generating Raw Data

```bash
# Generate training set (500 games)
uv run python scripts/generate_dataset.py --games 500 --output data/raw/training_games.jsonl

# Generate validation set (100 games, higher quality)
uv run python scripts/generate_dataset.py --games 100 --depth 5 --output data/raw/validation_games.jsonl

# Generate test set (100 games)
uv run python scripts/generate_dataset.py --games 100 --seed 42 --output data/raw/test_games.jsonl
```

## Dataset Statistics (Expected)

For 500 games at depth=4:
- **Win rate**: 70-80%
- **Total samples**: ~83,000 moves
- **Augmented**: ~664,000 samples (8x)
- **File size**: ~50-100 MB (JSONL)

## .gitignore Rules

- Raw data files (.jsonl) are ignored (too large)
- Processed tensors (.pt) are ignored
- .gitkeep files preserve directory structure
