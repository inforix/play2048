# Training Directory

Training scripts for all three neural network methods.

## Scripts

- `dataset.py` - PyTorch Dataset class for 2048 games
- `augmentation.py` - Data augmentation (8-fold symmetry)
- `train_cnn.py` - Train Method 1 (CNN)
- `train_dual.py` - Train Method 2 (Dual Network)
- `train_transformer.py` - Train Method 3 (Transformer)
- `utils.py` - Training utilities

## Implementation Status

- [ ] PyTorch Dataset class
- [ ] Data augmentation pipeline
- [ ] Training loop for Method 1
- [ ] Training loop for Method 2
- [ ] Training loop for Method 3

## Training Reference

See `../specs/train_spec.md` for:
- Hyperparameters
- Training schedules
- Loss functions
- Evaluation metrics

## Usage (After Implementation)

```bash
# Train CNN
uv run python training/train_cnn.py \
  --data data/raw/training_games.jsonl \
  --epochs 100 \
  --batch-size 256

# Train Dual Network
uv run python training/train_dual.py \
  --data data/raw/training_games.jsonl \
  --epochs 150 \
  --batch-size 128

# Train Transformer
uv run python training/train_transformer.py \
  --data data/raw/training_games.jsonl \
  --epochs 200 \
  --batch-size 64
```
