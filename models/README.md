# Models Directory

This directory contains neural network model implementations for 2048 AI.

## Structure

```
models/
├── cnn/                    # ✅ CNN-Based Policy Network
├── dual/                   # ✅ AlphaZero Dual Network
└── transformer/            # ✅ Transformer-Based Policy
```

## Implementation Status

### ✅ CNN Policy Network (`cnn/`)
- **Status**: Complete and tested
- **Files**: `cnn_policy.py`, `__init__.py`, `README.md`
- **Models**: 
  - `CNNPolicy` - Single-head policy network
  - `DualCNNPolicy` - Dual-head (policy + value)
- **Architecture**: Residual blocks + global pooling
- **Parameters**: 400K - 1.8M (configurable)
- **Input**: (batch, 1, 4, 4)
- **Output**: (batch, 4) action logits

### ✅ AlphaZero Dual Network (`dual/`)
- **Status**: Complete and tested
- **Files**: `alphazero_network.py`, `resblock.py`, `__init__.py`
- **Model**: `AlphaZeroNetwork`
- **Architecture**: ResNet backbone + dual heads
- **Parameters**: 1M - 5M (configurable)
- **Input**: (batch, 4, 4)
- **Output**: 
  - Policy: (batch, 4) action logits
  - Value: (batch, 1) state value

### ✅ Transformer Policy (`transformer/`)
- **Status**: Complete and tested
- **Files**: `transformer_policy.py`, `positional_encoding.py`, `__init__.py`
- **Models**:
  - `TransformerPolicy` - Single-head policy
  - `DualTransformerPolicy` - Dual-head (policy + value)
- **Architecture**: Multi-head attention + 2D positional encoding
- **Parameters**: 500K - 2M (configurable)
- **Input**: (batch, 1, 4, 4)
- **Output**: (batch, 4) action logits

## Quick Start

### Testing Models

```bash
# Test CNN
python -m models.cnn.cnn_policy

# Test Transformer
python -m models.transformer.transformer_policy

# Test AlphaZero
python -m models.dual.alphazero_network
```

### Using Models

```python
# CNN Policy
from models.cnn import CNNPolicy

model = CNNPolicy(base_channels=128, num_blocks=4)
logits = model(board_tensor)  # Shape: (batch, 4)
probs = model.get_action_probabilities(board_tensor)
action = model.predict_action(board_tensor)

# Transformer Policy
from models.transformer import TransformerPolicy

model = TransformerPolicy(embed_dim=128, num_layers=4)
logits = model(board_tensor)  # Shape: (batch, 4)
probs = model.get_action_probabilities(board_tensor)

# AlphaZero Network
from models.dual import AlphaZeroNetwork

model = AlphaZeroNetwork(num_blocks=4, channels=256)
policy_logits, value = model(board_tensor)
# Policy: (batch, 4), Value: (batch, 1)
```

### Loading Checkpoints

```python
import torch

# Load CNN checkpoint
model = CNNPolicy()
checkpoint = torch.load('checkpoints/cnn/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Load Transformer checkpoint
model = TransformerPolicy()
checkpoint = torch.load('checkpoints/transformer/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Load AlphaZero checkpoint
model = AlphaZeroNetwork()
checkpoint = torch.load('checkpoints/test_alphazero/final_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Model Comparison

| Model | Parameters | Speed | Performance | Best For |
|-------|------------|-------|-------------|----------|
| **CNN** | 400K-1.8M | Fast (0.1ms) | Good | Production |
| **Transformer** | 500K-2M | Medium (0.5ms) | Better | Research |
| **AlphaZero** | 1M-5M | Slow (50ms)* | Best | Competition |

*With MCTS (200 simulations)

See [../docs/MODEL-COMPARISON.md](../docs/MODEL-COMPARISON.md) for detailed comparison.

## Training

Each architecture has its own training script:

```bash
# Train CNN
python training/train_cnn.py --data data/training_games.jsonl

# Train Transformer
python training/train_transformer.py --data data/training_games.jsonl

# Train AlphaZero (self-play)
python training/train_alphazero.py --num-iterations 100
```

## Documentation

- **CNN**: [cnn/README.md](cnn/README.md), [../docs/CNN-TRAINING-GUIDE.md](../docs/CNN-TRAINING-GUIDE.md)
- **Transformer**: [transformer/README.md](transformer/README.md), [../docs/TRANSFORMER-TRAINING-PLAN.md](../docs/TRANSFORMER-TRAINING-PLAN.md)
- **AlphaZero**: [dual/README.md](dual/README.md), [../docs/ALPHAZERO-GUIDE.md](../docs/ALPHAZERO-GUIDE.md)
- **Comparison**: [../docs/MODEL-COMPARISON.md](../docs/MODEL-COMPARISON.md)

## Architecture Reference

See `../specs/train_spec.md` for detailed architecture specifications that guided these implementations.
