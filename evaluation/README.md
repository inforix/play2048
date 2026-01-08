# Evaluation Tools

Comprehensive evaluation suite for trained 2048 transformer models.

## Scripts

### 1. offline_eval.py - Test Set Metrics

Evaluates model on the test set and computes detailed metrics.

```bash
# Basic usage
uv run python evaluation/offline_eval.py

# With custom checkpoint
uv run python evaluation/offline_eval.py --checkpoint checkpoints/transformer/best_model.pth

# Options
uv run python evaluation/offline_eval.py \
    --checkpoint checkpoints/transformer/best_model.pth \
    --data-path data/training_games.jsonl \
    --batch-size 256 \
    --num-workers 4 \
    --output-dir results/evaluation \
    --no-augment  # Disable augmentation for faster evaluation
```

**Outputs:**
- `test_metrics.json` - Numerical results (accuracy, top-2, per-class metrics)
- `confusion_matrix.png` - Normalized confusion matrix heatmap
- `per_class_metrics.png` - Precision/recall/F1 bar chart

**Metrics Computed:**
- Overall accuracy (exact action match)
- Top-2 accuracy (correct action in top 2 predictions)
- Per-action precision, recall, F1-score
- Confusion matrix (shows which actions are confused)

---

### 2. game_simulator.py - Play Full Games

Simulates complete 2048 games using the trained model.

```bash
# Play 100 games
uv run python evaluation/game_simulator.py --num-games 100

# Options
uv run python evaluation/game_simulator.py \
    --checkpoint checkpoints/transformer/best_model.pth \
    --num-games 100 \
    --epsilon 0.0 \  # 0.0 = greedy, >0 = exploration
    --output-dir results/evaluation \
    --seed 42
```

**Outputs:**
- `game_results_100games.json` - Detailed results for all games

**Metrics Computed:**
- Win rate (% of games reaching 2048)
- Average score
- Average max tile achieved
- Average moves per game
- Action distribution (which actions are preferred)
- Max tile distribution (how often each tile is reached)

**Example Output:**
```
======================================================================
GAME SIMULATION RESULTS
======================================================================

Overall Performance:
  Games Played:    100
  Win Rate:        45.00% (reached 2048)
  Average Score:   15,432 ± 3,241
  Max Score:       28,764
  Min Score:       4,128

Game Statistics:
  Avg Max Tile:    1448
  Avg Moves:       234.5
  Avg Invalid:     12.3

Max Tile Distribution:
   2048:   45 games ( 45.0%)
   1024:   32 games ( 32.0%)
    512:   18 games ( 18.0%)
    256:    5 games (  5.0%)

Action Distribution:
  UP:     28.5%
  DOWN:   18.2%
  LEFT:   31.7%
  RIGHT:  21.6%
======================================================================
```

---

### 3. visualize_results.py - Training Curves

Visualizes training progress from training history.

```bash
# Visualize training results
uv run python evaluation/visualize_results.py

# With custom paths
uv run python evaluation/visualize_results.py \
    --history results/training_curves/transformer/training_history.json \
    --output-dir results/training_curves/transformer/plots
```

**Outputs:**
- `loss_curves.png` - Training and validation loss over epochs
- `accuracy_curves.png` - Exact and top-2 accuracy over epochs
- `learning_rate.png` - Learning rate schedule
- `combined_metrics.png` - All metrics in one figure
- `overfitting_analysis.png` - Train-val gap analysis

**Example Output:**
```
======================================================================
TRAINING SUMMARY
======================================================================

Total Epochs: 200

Best Validation Metrics:
  Loss:     0.9821 (epoch 127)
  Accuracy: 78.42% (epoch 145)
  Top-2:    91.23% (epoch 138)

Final Epoch Metrics:
  Train Loss: 0.8543
  Val Loss:   1.0234
  Train Acc:  82.34%
  Val Acc:    76.89%

Generalization (Final Epoch):
  Loss Gap:  +0.1691
  Acc Gap:   -5.45%
  ⚠️  Warning: Possible overfitting detected
======================================================================
```

---

## Complete Evaluation Workflow

After training, run all evaluation scripts:

```bash
# 1. Visualize training curves
uv run python evaluation/visualize_results.py

# 2. Evaluate on test set
uv run python evaluation/offline_eval.py --batch-size 256 --num-workers 4

# 3. Simulate games
uv run python evaluation/game_simulator.py --num-games 100

# 4. View results
ls -lh results/evaluation/
```

---

## Expected Performance Targets

Based on the training specification:

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| **Test Accuracy** | 70% | 80% | 85% |
| **Top-2 Accuracy** | 85% | 90% | 95% |
| **Win Rate** | 50% | 70% | 80% |
| **Avg Score** | 10,000 | 15,000 | 20,000 |

**Baseline Comparison:**
- Random policy: ~5% win rate, ~3,000 avg score
- Expectimax (depth=2): ~80% win rate, ~18,000 avg score

---

## Interpreting Results

### Test Set Metrics (offline_eval.py)

- **High accuracy** (>75%): Model learns good move selection
- **Confusion matrix**: Shows which actions are confused
  - Ideally diagonal (few confusions)
  - Common confusions: LEFT↔RIGHT, UP↔DOWN (opposite directions)
- **Per-action F1**: Should be balanced (~0.7-0.8 for all actions)
  - Low F1 for specific action → model struggles with that direction

### Game Simulation (game_simulator.py)

- **Win rate** vs **avg score**: Both should be high
  - High win rate, low score → reaches 2048 but stops growing
  - Low win rate, high score → grows well but can't reach 2048
- **Action distribution**: Should be relatively balanced (20-30% each)
  - Heavy bias (>50%) → model has learned bad strategy
- **Invalid moves**: Should be low (<5% of total moves)
  - High invalid rate → model doesn't understand game rules well

### Training Curves (visualize_results.py)

- **Loss gap** (val - train):
  - Small positive gap (0.1-0.3) → good generalization
  - Large positive gap (>0.5) → overfitting
  - Negative gap → possible data leakage or bug
- **Accuracy gap** (val - train):
  - Small gap (-5% to +5%) → good generalization
  - Large negative gap (<-10%) → overfitting
- **Learning rate**: Should decrease over time (warmup → cosine decay)

---

## Troubleshooting

### Low Test Accuracy (<60%)

- Check if model is actually learning (train accuracy should improve)
- Increase model capacity (more layers/heads)
- Train for more epochs
- Check for bugs in data preprocessing

### Low Win Rate (<30%) Despite Good Test Accuracy

- Model may be overfitting to specific patterns
- Try epsilon-greedy exploration during evaluation
- Check action distribution (heavy bias indicates bad strategy)
- Model may need more training data or game experience

### High Train-Val Gap (Overfitting)

- Increase dropout rate
- Add more data augmentation
- Use smaller model
- Add weight decay
- Stop training earlier (use early stopping)

---

## Next Steps

After evaluation:

1. **If results are good** (meet targets):
   - Save final model
   - Document performance
   - Consider deploying for inference

2. **If results are poor**:
   - Analyze failure modes (confusion matrix, game logs)
   - Adjust hyperparameters
   - Collect more training data
   - Try different architectures

3. **For further improvement**:
   - Ensemble multiple models
   - Fine-tune with reinforcement learning
   - Increase model capacity
   - Add auxiliary tasks (value prediction, game state prediction)
