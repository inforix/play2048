# CNN Implementation for 2048

This directory contains the Convolutional Neural Network (CNN) implementation for learning to play 2048.

## Architecture

### CNNPolicy (Single-Head)
- **Input**: 4×4 board (1 channel)
- **Initial Conv**: 1 → base_channels (default: 128)
- **Residual Blocks**: 4 blocks with skip connections
- **Feature Extraction**: Additional conv layer (128 → 256 channels)
- **Global Pooling**: Adaptive average pooling
- **Policy Head**: 256 → 4 actions

### DualCNNPolicy (Dual-Head)
- Shared backbone (same as above)
- **Policy Head**: 256 → 4 action logits
- **Value Head**: 256 → 1 state value (in [-1, 1])

## Key Features

1. **Residual Connections**: Skip connections for better gradient flow
2. **Batch Normalization**: Stabilizes training
3. **Dropout**: Prevents overfitting
4. **He Initialization**: Proper weight initialization for ReLU networks

## Model Configurations

| Configuration | Channels | Blocks | Parameters |
|---------------|----------|--------|------------|
| Small         | 64       | 3      | ~180K      |
| Medium        | 128      | 4      | ~400K      |
| Large         | 256      | 6      | ~1.8M      |

## Usage

### Testing the Model

```python
from models.cnn import CNNPolicy

# Create model
model = CNNPolicy(base_channels=128, num_blocks=4)

# Test forward pass
import torch
board = torch.randn(1, 1, 4, 4)  # Batch of 1
logits = model(board)  # Shape: (1, 4)
probs = model.get_action_probabilities(board)
```

### Training

See `training/train_cnn.py` for the complete training script:

```bash
python training/train_cnn.py \
    --data data/training_games.jsonl \
    --batch-size 128 \
    --epochs 200 \
    --base-channels 128 \
    --num-blocks 4
```

## Comparison with Other Architectures

### CNN vs Transformer
- **CNN**: Better at capturing local spatial patterns, fewer parameters
- **Transformer**: Better at long-range dependencies, more parameters
- **CNN**: Faster training and inference
- **Transformer**: More flexible attention mechanism

### CNN vs AlphaZero
- **AlphaZero**: Uses CNN + ResNet with MCTS for planning
- **This CNN**: Supervised learning on expert games
- **AlphaZero**: Better performance but requires more compute
- **This CNN**: Faster to train, good for baseline

## Files

- `__init__.py`: Package initialization
- `cnn_policy.py`: CNN model implementations
  - `ConvBlock`: Basic conv block with BN and ReLU
  - `ResidualBlock`: Residual block with skip connection
  - `CNNPolicy`: Single-head policy network
  - `DualCNNPolicy`: Dual-head policy + value network
