# Models Directory

This directory contains neural network model implementations for 2048 AI.

## Structure

```
models/
├── cnn/                    # Method 1: CNN-Based Policy Network
├── dual/                   # Method 2: Dual Network (AlphaZero-style)
└── transformer/            # Method 3: Transformer-Based Policy
```

## Implementation Status

- [ ] **CNN Policy Network** (`cnn/policy_cnn.py`)
  - 4 conv layers + FC layers
  - Input: (batch, 1, 4, 4)
  - Output: (batch, 4) action logits
  
- [ ] **Dual Network** (`dual/dual_network.py`)
  - Shared ResNet backbone
  - Policy head: (batch, 4)
  - Value head: (batch, 1)
  
- [ ] **Transformer Policy** (`transformer/transformer_policy.py`)
  - 4 transformer encoder layers
  - 2D positional encoding
  - Global pooling + policy head

## Architecture Reference

See `../specs/train_spec.md` for detailed architecture specifications.

## Usage (After Implementation)

```python
from models.cnn.policy_cnn import PolicyCNN
from models.dual.dual_network import DualNetwork
from models.transformer.transformer_policy import TransformerPolicy

# Load model
model = PolicyCNN()
model.load_state_dict(torch.load('checkpoints/method1_best.pth'))

# Inference
board_tensor = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0)
action_logits = model(board_tensor)
best_action = action_logits.argmax(dim=1)
```
