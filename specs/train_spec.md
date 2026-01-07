# 2048 Neural Network Training Specification

## Overview
This document specifies the training approach for three neural network architectures to play 2048. All models will be trained on exported game histories and compared for performance.

**Training Goal**: Learn optimal move selection from human gameplay data and potentially surpass the current Expectimax algorithm (~80% win rate).

---

## Data Format

### Input: Exported Game History (JSONL/JSON)
```json
{
  "game": "2048 Aurora Edition",
  "totalMoves": 150,
  "finalScore": 12345,
  "bestScore": 15000,
  "timestamp": "2026-01-07T10:30:00.000Z",
  "finalBoard": [[0, 2, 4, 8], [2, 4, 8, 16], ...],
  "moves": [
    {
      "moveNumber": 1,
      "direction": "up",
      "boardState": [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0], [0, 2, 0, 0]],
      "score": 0
    },
    ...
  ]
}
```

### Preprocessing Requirements
1. **Board Encoding**: Convert 4×4 grid to neural network input
   - Log2 normalization: `log2(tile) / 11.0` (2048 = 2^11)
   - One-hot encoding: 16 possible tile values (0, 2, 4, ..., 2048+)
   - Raw values with normalization: `tile / 2048.0`

2. **Action Encoding**: Map directions to integers
   - Up: 0, Down: 1, Left: 2, Right: 3

3. **Data Filtering**:
   - Remove invalid moves (moves that don't change board state)
   - Optional: Weight high-scoring games more heavily
   - Optional: Filter out early-game vs late-game moves separately

4. **Data Augmentation**:
   - 8-fold symmetry: 4 rotations × 2 reflections
   - Increases training data 8× with consistent action mapping

5. **Train/Val/Test Split**: 70% / 15% / 15%

---

## Model Architectures

### Method 1: CNN-Based Policy Network (Baseline)

**Architecture Overview**: Convolutional neural network that outputs action probabilities.

```
Input: (batch, 1, 4, 4) - Single channel board state

Conv Block 1:
  - Conv2d(1 → 128, kernel=2, padding=1)
  - BatchNorm2d(128)
  - ReLU
  - Output: (batch, 128, 5, 5)

Conv Block 2:
  - Conv2d(128 → 128, kernel=2, padding=1)
  - BatchNorm2d(128)
  - ReLU
  - Output: (batch, 128, 6, 6)

Conv Block 3:
  - Conv2d(128 → 256, kernel=2, padding=1)
  - BatchNorm2d(256)
  - ReLU
  - Output: (batch, 256, 7, 7)

Conv Block 4:
  - Conv2d(256 → 128, kernel=2, padding=0)
  - BatchNorm2d(128)
  - ReLU
  - Output: (batch, 128, 6, 6)

Flatten: → (batch, 4608)

FC Block:
  - Linear(4608 → 512)
  - ReLU
  - Dropout(0.3)
  - Linear(512 → 256)
  - ReLU
  - Dropout(0.2)
  - Linear(256 → 4)
  
Output: (batch, 4) - Action logits
```

**Loss Function**: Cross-Entropy Loss
**Optimizer**: Adam (lr=0.001, weight_decay=1e-5)
**Training Strategy**: Supervised learning from expert demonstrations

**Pros**:
- Simple, proven architecture
- Fast training and inference
- Good spatial pattern recognition

**Cons**:
- No explicit value estimation
- May overfit to specific board patterns

---

### Method 2: Dual Network (AlphaZero-Style)

**Architecture Overview**: Shared CNN backbone with separate policy and value heads.

```
Input: (batch, 1, 4, 4)

Shared Backbone:
  ResBlock 1:
    - Conv2d(1 → 128, kernel=3, padding=1)
    - BatchNorm2d(128)
    - ReLU
    - Conv2d(128 → 128, kernel=3, padding=1)
    - BatchNorm2d(128)
    - Skip connection + ReLU
    - Output: (batch, 128, 4, 4)
  
  ResBlock 2:
    - Conv2d(128 → 256, kernel=3, padding=1)
    - BatchNorm2d(256)
    - ReLU
    - Conv2d(256 → 256, kernel=3, padding=1)
    - BatchNorm2d(256)
    - Skip connection (1×1 conv for channel matching) + ReLU
    - Output: (batch, 256, 4, 4)
  
  ResBlock 3:
    - Conv2d(256 → 256, kernel=3, padding=1)
    - BatchNorm2d(256)
    - ReLU
    - Conv2d(256 → 256, kernel=3, padding=1)
    - BatchNorm2d(256)
    - Skip connection + ReLU
    - Output: (batch, 256, 4, 4)

Flatten: → (batch, 4096)

Policy Head:
  - Linear(4096 → 256)
  - ReLU
  - Dropout(0.3)
  - Linear(256 → 4)
  - Output: (batch, 4) - Action logits

Value Head:
  - Linear(4096 → 256)
  - ReLU
  - Dropout(0.3)
  - Linear(256 → 128)
  - ReLU
  - Linear(128 → 1)
  - Tanh (for normalized value)
  - Output: (batch, 1) - State value [-1, 1]
```

**Loss Function**: Combined loss
- Policy loss: Cross-Entropy on expert actions
- Value loss: MSE on normalized final score or win/loss
- Total: `λ_policy * policy_loss + λ_value * value_loss`
- Default weights: λ_policy = 1.0, λ_value = 0.5

**Value Target Options**:
1. Normalized final score: `(score - mean_score) / std_score`
2. Win/loss binary: `1.0 if max_tile >= 2048 else 0.0`
3. Future discounted score: TD-learning style

**Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
**Training Strategy**: Multi-task learning with policy and value prediction

**Pros**:
- Value head provides position evaluation
- More robust to uncertain positions
- Better generalization via multi-task learning

**Cons**:
- More complex training (need value labels)
- Slower training due to larger network
- Requires careful loss weight balancing

---

### Method 3: Transformer-Based Architecture

**Architecture Overview**: Transformer encoder treating 16 tiles as a sequence with positional encoding.

```
Input: (batch, 1, 4, 4)

Preprocessing:
  - Flatten spatial dims: → (batch, 16, 1)
  - Tile Embedding: Linear(1 → 128)
  - Positional Encoding: Add learned 2D position embeddings (4×4 grid)
  - Output: (batch, 16, 128) - Sequence of tile embeddings

Transformer Encoder:
  TransformerEncoderLayer 1:
    - MultiheadAttention(embed_dim=128, num_heads=8)
    - LayerNorm
    - FeedForward: Linear(128 → 512) → ReLU → Linear(512 → 128)
    - LayerNorm
    - Dropout(0.1)
    - Output: (batch, 16, 128)
  
  TransformerEncoderLayer 2:
    - Same structure
    - Output: (batch, 16, 128)
  
  TransformerEncoderLayer 3:
    - Same structure
    - Output: (batch, 16, 128)
  
  TransformerEncoderLayer 4:
    - Same structure
    - Output: (batch, 16, 128)

Global Pooling:
  - Mean pooling across sequence: → (batch, 128)
  - Or use [CLS] token approach

Policy Head:
  - Linear(128 → 256)
  - ReLU
  - Dropout(0.2)
  - Linear(256 → 128)
  - ReLU
  - Linear(128 → 4)
  - Output: (batch, 4) - Action logits

Optional Value Head (Dual-Transformer):
  - Linear(128 → 64)
  - ReLU
  - Linear(64 → 1)
  - Tanh
  - Output: (batch, 1) - State value
```

**Positional Encoding**: 2D learned embeddings
- Create 16 learnable position vectors (4×4 grid positions)
- Each position (i, j) has unique embedding
- Preserves spatial structure information

**Loss Function**: 
- Single-head: Cross-Entropy
- Dual-head: Combined policy + value loss (like Method 2)

**Optimizer**: AdamW (lr=0.0003, weight_decay=0.01, warmup steps)
**Training Strategy**: Supervised learning with learning rate warmup and cosine decay

**Pros**:
- Captures long-range tile dependencies
- No spatial bias (learns all patterns)
- State-of-the-art in many domains
- Excellent for capturing strategic patterns

**Cons**:
- Highest computational cost
- Requires more data to train effectively
- May be overkill for 4×4 grid
- Slower inference

---

## Training Configuration

### Hardware Requirements
- **Minimum**: CPU with 8GB RAM (slow, for testing only)
- **Recommended**: NVIDIA GPU with 8GB+ VRAM (GTX 1080, RTX 2060+)
- **Optimal**: NVIDIA GPU with 16GB+ VRAM (RTX 3090, RTX 4090, A100)

### Hyperparameters

| Parameter | Method 1 (CNN) | Method 2 (Dual) | Method 3 (Transformer) |
|-----------|---------------|-----------------|------------------------|
| Batch Size | 256 | 128 | 64 |
| Learning Rate | 0.001 | 0.001 | 0.0003 |
| Weight Decay | 1e-5 | 1e-4 | 0.01 |
| Optimizer | Adam | Adam | AdamW |
| Epochs | 100 | 150 | 200 |
| Dropout | 0.2-0.3 | 0.3 | 0.1-0.2 |
| LR Scheduler | ReduceLROnPlateau | ReduceLROnPlateau | CosineAnnealingWarmRestarts |
| Early Stopping | 15 epochs | 20 epochs | 25 epochs |

### Data Augmentation Pipeline
```python
def augment_board(board, action):
    """Apply random rotation/reflection with consistent action mapping."""
    # Rotation: 0°, 90°, 180°, 270°
    # Reflection: horizontal, vertical, or none
    # Update action accordingly
    # Example: 90° CW rotation: up→right, right→down, down→left, left→up
    return augmented_board, augmented_action
```

Apply augmentation with 50% probability during training.

---

## Training Pipeline

### Phase 1: Data Preparation (All Methods)
1. Export game histories from 2048 game (download button → JSON files)
2. Collect minimum 20+ games (recommended 100+ games, ideal 1000+)
3. Parse JSON and extract (board_state, action, score) tuples
4. Filter invalid moves
5. Apply data augmentation
6. Split into train/val/test sets
7. Create PyTorch DataLoaders

**Expected Data Volume**:
- 20 games × 150 moves avg × 8 augmentations = 24,000 training samples
- 100 games = 120,000 samples
- 1000 games = 1,200,000 samples

### Phase 2: Model Training (Per Method)

**Method 1 Training**:
```
Epochs: 100
- Train with CrossEntropyLoss
- Monitor validation accuracy
- Save best model based on val_loss
- Log: train_loss, train_acc, val_loss, val_acc per epoch
```

**Method 2 Training**:
```
Epochs: 150
- Train with combined policy + value loss
- Compute value targets from game outcomes
- Monitor both policy accuracy and value MSE
- Save best model based on combined val_loss
- Log: policy_loss, value_loss, total_loss, policy_acc, value_mae
```

**Method 3 Training**:
```
Epochs: 200 (with early stopping)
- Warmup learning rate for first 10 epochs
- Train with CrossEntropyLoss (or combined if dual-head)
- Use gradient clipping (max_norm=1.0)
- Monitor attention patterns (optional visualization)
- Save checkpoints every 25 epochs
- Log: same as Method 1 or 2 depending on variant
```

### Phase 3: Evaluation (All Methods)

**Offline Metrics**:
1. **Test Set Accuracy**: % of moves matching expert actions
2. **Top-2 Accuracy**: Expert action in top 2 predictions
3. **Per-Game-Stage Accuracy**: Early/mid/late game performance
4. **Confusion Matrix**: Which actions confused for which
5. **Value Prediction Error** (Methods 2 & 3 dual): MAE on score prediction

**Online Metrics** (Game Simulation):
1. **Win Rate**: % games reaching 2048 tile (100 games)
2. **Average Score**: Mean score across 100 games
3. **Max Tile Distribution**: Histogram of max tiles achieved
4. **Average Moves**: Mean game length
5. **Survival Rate**: % games exceeding 500 moves

**Comparison Baseline**:
- Current Expectimax: ~80% win rate
- Current Monte Carlo: ~70% win rate
- Current Weighted Heuristic: ~60% win rate

### Phase 4: Model Comparison

**Quantitative Comparison Table**:
```
| Metric                | Method 1 | Method 2 | Method 3 | Expectimax |
|-----------------------|----------|----------|----------|------------|
| Test Accuracy (%)     |          |          |          | N/A        |
| Win Rate (%)          |          |          |          | 80%        |
| Avg Score             |          |          |          |            |
| Max Tile (avg)        |          |          |          |            |
| Inference Time (ms)   |          |          |          | 5-10ms     |
| Training Time (hrs)   |          |          |          | N/A        |
| Model Size (MB)       |          |          |          | N/A        |
```

**Qualitative Analysis**:
- Which model handles corner protection best?
- Which model plans merges more effectively?
- Which model recovers from mistakes better?
- Which model's decisions are most interpretable?

---

## Implementation Roadmap

### Week 1: Data & Infrastructure
- [ ] Export 50+ games from 2048 game
- [ ] Implement `Game2048Dataset` class
- [ ] Implement data augmentation
- [ ] Create data loaders
- [ ] Verify data pipeline with visualization

### Week 2: Method 1 (CNN Baseline)
- [ ] Implement `PolicyCNN` model
- [ ] Implement training loop
- [ ] Train for 100 epochs
- [ ] Evaluate on test set
- [ ] Run 100 game simulations
- [ ] Record baseline metrics

### Week 3: Method 2 (Dual Network)
- [ ] Implement `DualNetwork` model with ResBlocks
- [ ] Implement combined loss function
- [ ] Prepare value targets from game data
- [ ] Train for 150 epochs
- [ ] Evaluate on test set
- [ ] Run 100 game simulations
- [ ] Compare with Method 1

### Week 4: Method 3 (Transformer)
- [ ] Implement `TransformerPolicy` model
- [ ] Implement 2D positional encoding
- [ ] Implement learning rate warmup
- [ ] Train for 200 epochs (with early stopping)
- [ ] Evaluate on test set
- [ ] Run 100 game simulations
- [ ] Compare with Methods 1 & 2

### Week 5: Analysis & Optimization
- [ ] Create comparison table
- [ ] Visualize attention patterns (Method 3)
- [ ] Analyze failure cases for each method
- [ ] Hyperparameter tuning for best method
- [ ] Ensemble experiments (optional)
- [ ] Write final report

---

## Advanced Techniques (Optional)

### 1. Curriculum Learning
- Start with end-game positions (simple)
- Progress to mid-game positions (medium)
- Finally train on full games (complex)
- Helps models learn strategic planning

### 2. Imitation Learning with DAgger
- Train initial model on expert data
- Generate new games with model
- Get expert corrections on mistakes
- Retrain with aggregated dataset
- Iterative improvement

### 3. Reinforcement Learning Fine-Tuning
- Use supervised model as initialization
- Fine-tune with policy gradient (REINFORCE/PPO)
- Learn to exceed expert performance
- Requires game environment wrapper

### 4. Ensemble Methods
- Combine predictions from all 3 methods
- Voting or weighted average
- May achieve best overall performance
- Trade-off: 3× inference time

### 5. Model Compression
- Knowledge distillation: Train small model from large
- Quantization: INT8 inference
- Pruning: Remove redundant weights
- Target: Deploy to web browser via ONNX

---

## Expected Outcomes

### Success Criteria (Minimum)
- [ ] Test accuracy ≥ 70% on expert moves
- [ ] Win rate ≥ 50% (reaches 2048 tile)
- [ ] Average score ≥ 10,000
- [ ] Inference time < 50ms per move

### Target Performance
- [ ] Test accuracy ≥ 80%
- [ ] Win rate ≥ 70% (competitive with Expectimax)
- [ ] Average score ≥ 15,000
- [ ] Inference time < 20ms

### Stretch Goals
- [ ] Win rate ≥ 85% (surpass Expectimax)
- [ ] Achieve 4096 tile in 10%+ of games
- [ ] Average score ≥ 20,000
- [ ] Deploy model to web browser

### Hypotheses
1. **Method 2 (Dual Network)** will likely perform best overall
   - Value head helps with position evaluation
   - ResNet backbone captures spatial patterns well
   - Multi-task learning improves generalization

2. **Method 1 (CNN)** will be fastest but less strategic
   - Good for reactive play
   - May miss long-term planning
   - Best inference speed

3. **Method 3 (Transformer)** will excel with large datasets
   - Requires 100K+ samples to shine
   - Best at capturing complex patterns
   - May overfit with small data

---

## File Structure

```
play2048/
├── index.html                 # Game interface
├── spec.md                    # Game specification
├── train_spec.md             # This file
├── data/
│   ├── exported_games/       # JSON files from game
│   ├── processed/            # Preprocessed .pt files
│   └── augmented/            # Augmented dataset
├── models/
│   ├── cnn_policy.py         # Method 1
│   ├── dual_network.py       # Method 2
│   ├── transformer_policy.py # Method 3
│   └── base_model.py         # Shared utilities
├── training/
│   ├── dataset.py            # Game2048Dataset
│   ├── augmentation.py       # Data augmentation
│   ├── train_cnn.py          # Train Method 1
│   ├── train_dual.py         # Train Method 2
│   ├── train_transformer.py  # Train Method 3
│   └── utils.py              # Training utilities
├── evaluation/
│   ├── offline_eval.py       # Test set metrics
│   ├── game_simulator.py     # Play full games
│   └── compare_models.py     # Comparison analysis
├── checkpoints/              # Saved models
│   ├── method1_best.pth
│   ├── method2_best.pth
│   └── method3_best.pth
├── results/                  # Metrics, plots, logs
│   ├── training_curves/
│   ├── comparison_table.csv
│   └── game_simulations/
└── requirements.txt          # Python dependencies
```

---

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.13.0
scikit-learn>=1.3.0
pandas>=2.0.0
tqdm>=4.65.0
```

---

## Monitoring & Logging

### TensorBoard Logging
- Training/validation loss curves
- Accuracy metrics
- Learning rate schedule
- Gradient norms
- Model parameter histograms
- Confusion matrices
- Sample predictions

### Checkpointing Strategy
- Save best model based on validation loss
- Save checkpoints every 25 epochs
- Keep last 3 checkpoints
- Save final model at end of training

### Early Stopping
- Monitor validation loss
- Patience: 15-25 epochs depending on method
- Restore best weights on termination

---

## Next Steps

1. **Export game data**: Play 50-100 games, download JSON histories
2. **Set up environment**: Install PyTorch, create project structure
3. **Implement dataset**: Start with `dataset.py` and verify data loading
4. **Train Method 1**: Get baseline results (simplest model first)
5. **Iterate**: Move to Methods 2 and 3 based on Method 1 results
6. **Compare**: Run all evaluations and create comparison report
7. **Deploy**: Export best model for web integration (ONNX.js or API)

---

## Questions for Investigation

1. Does data augmentation significantly improve performance?
2. How much training data is sufficient for each method?
3. Can we identify which board patterns are hardest to learn?
4. Does the value head in Method 2 improve policy learning?
5. Do transformers justify the extra complexity for 4×4 grids?
6. Can we combine NN predictions with Expectimax for best results?
7. What is the performance ceiling with current expert data?
8. Would RL fine-tuning push beyond expert performance?

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-07  
**Status**: Ready for Implementation
