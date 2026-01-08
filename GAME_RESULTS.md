# 2048 Game Playing Results - Trained Transformer Model

**Date:** January 8, 2026  
**Model:** Transformer Policy (862K parameters)  
**Training:** 5 epochs (debug mode, ~1 hour)  
**Checkpoint:** best_model.pth (epoch 1, val_loss=1.114, val_acc=44.34%)

---

## Test Results: 10 Games

### Individual Game Results

| Game | Score | Max Tile | Moves | Status |
|------|-------|----------|-------|--------|
| 1    | 1,324 | 128      | 137   | Lost   |
| 2    | 592   | 64       | 82    | Lost   |
| 3    | 712   | 64       | 91    | Lost   |
| 4    | 640   | 64       | 89    | Lost   |
| 5    | 1,044 | 64       | 126   | Lost   |
| 6    | 1,372 | 128      | 141   | Lost   |
| 7    | 1,436 | 128      | 147   | Lost   |
| 8    | 484   | 32       | 78    | Lost   |
| 9    | 2,240 | 256      | 189   | Lost   |
| 10   | 3,024 | 256      | 256   | Lost   |

### Summary Statistics

- **Games Played:** 10
- **Win Rate:** 0.0% (0/10 reached 2048)
- **Average Score:** 1,287
- **Best Score:** 3,024
- **Average Moves:** 133.6

### Max Tile Distribution

| Tile | Games | Percentage |
|------|-------|------------|
| 256  | 2     | 20.0%      |
| 128  | 3     | 30.0%      |
| 64   | 4     | 40.0%      |
| 32   | 1     | 10.0%      |

---

## Analysis

### Performance Assessment

The model shows **very poor performance**, which is expected given:
1. **Minimal training**: Only 5 epochs (should be 100-200 epochs)
2. **Early convergence**: Best model from epoch 1 suggests training issues
3. **Low validation accuracy**: 44.34% is barely better than random (25%)

### Comparison to Baselines

| Method | Win Rate | Avg Score | Max Tile |
|--------|----------|-----------|----------|
| **Current Model** | 0% | 1,287 | 64-256 |
| Random Policy | ~0% | 300-500 | 32-64 |
| Human Beginner | 5-10% | 2,000-5,000 | 256-512 |
| Expectimax (d=2) | ~80% | 18,000+ | 2048+ |

The model performs slightly better than random but far below human or algorithmic baselines.

### Issues Identified

1. **Insufficient Training**
   - Only 5 epochs vs recommended 200
   - Training loss still high (1.347)
   - Validation metrics plateaued immediately

2. **Poor Action Selection**
   - Model struggles to create merges
   - Often reaches game over quickly (avg 134 moves vs expected 300+)
   - Max tile of 256 suggests basic strategy missing

3. **Limited Learning**
   - Val accuracy 44% means wrong action chosen 56% of time
   - Top-2 accuracy 74% shows some pattern learning but insufficient

---

## Next Steps

To achieve good performance (70%+ win rate, 15,000+ avg score):

### 1. Complete Training
```bash
# Full training run (40-50 hours)
uv run python training/train_transformer.py \
    --epochs 200 \
    --batch-size 64 \
    --num-workers 4
```

### 2. Monitor Training
```bash
# Watch training progress
tensorboard --logdir results/training_curves/transformer
```

### 3. Expected Outcomes After Full Training

| Metric | Target |
|--------|--------|
| Validation Accuracy | 75-85% |
| Top-2 Accuracy | 90-95% |
| Win Rate | 60-80% |
| Average Score | 12,000-18,000 |
| Max Tile | 2048-4096 |

### 4. Evaluation After Training

```bash
# Comprehensive evaluation
uv run python evaluation/offline_eval.py
uv run python evaluation/game_simulator.py --num-games 100
uv run python evaluation/visualize_results.py
```

---

## Conclusion

The current model demonstrates **proof of concept** functionality:
- ✓ Model architecture works correctly
- ✓ Can play complete games without crashes
- ✓ Shows slight improvement over random
- ✗ Performance far below target due to insufficient training

**Recommendation:** Complete full 200-epoch training run to achieve competitive performance.

---

## Technical Notes

- **Device:** Apple Silicon MPS (Metal Performance Shaders)
- **Inference Speed:** ~100ms per move (acceptable for gameplay)
- **Model Size:** 9.9MB checkpoint file
- **Memory Usage:** ~500MB during inference
