# CNN Training Guide for 2048

This guide covers training the CNN-based model for playing 2048.

## Quick Start

### 1. Basic Training

Train a standard CNN model with default settings:

```bash
python training/train_cnn.py \
    --data data/training_games.jsonl \
    --checkpoint-dir checkpoints/cnn \
    --log-dir results/training_curves/cnn
```

### 2. Custom Configuration

Train with custom hyperparameters:

```bash
python training/train_cnn.py \
    --data data/training_games.jsonl \
    --base-channels 256 \
    --num-blocks 6 \
    --batch-size 128 \
    --lr 0.001 \
    --epochs 200 \
    --optimizer adam \
    --scheduler cosine
```

### 3. Dual-Head Model

Train a model with both policy and value heads:

```bash
python training/train_cnn.py \
    --data data/training_games.jsonl \
    --dual-head \
    --base-channels 128 \
    --num-blocks 4
```

## Training Arguments

### Data Arguments
- `--data`: Path to training data JSONL file
- `--batch-size`: Batch size (default: 128)
- `--num-workers`: Number of data loading workers (default: 4)
- `--no-augment`: Disable data augmentation

### Model Arguments
- `--base-channels`: Number of base channels (default: 128)
- `--num-blocks`: Number of residual blocks (default: 4)
- `--dropout`: Dropout rate in residual blocks (default: 0.1)
- `--head-dropout`: Dropout rate in policy head (default: 0.3)
- `--dual-head`: Use dual-head model (policy + value)

### Training Arguments
- `--epochs`: Number of training epochs (default: 200)
- `--lr`: Learning rate (default: 0.001)
- `--weight-decay`: L2 regularization (default: 0.0001)
- `--optimizer`: Optimizer type (adam, adamw, sgd)
- `--scheduler`: LR scheduler (cosine, step, plateau)
- `--warmup-epochs`: Warmup epochs (default: 10)
- `--gradient-clip`: Max gradient norm (default: 1.0)
- `--early-stop-patience`: Early stopping patience (default: 25)

### Checkpointing
- `--checkpoint-dir`: Directory to save checkpoints
- `--checkpoint-freq`: Save every N epochs (default: 25)
- `--resume`: Resume from checkpoint

### Other
- `--device`: Device (auto, cuda, mps, cpu)
- `--seed`: Random seed (default: 42)
- `--debug`: Debug mode (small dataset)

## Recommended Configurations

### Small Model (Fast Training)
```bash
python training/train_cnn.py \
    --base-channels 64 \
    --num-blocks 3 \
    --batch-size 256 \
    --lr 0.002
```
- Parameters: ~180K
- Training time: ~1-2 hours
- Good for quick experiments

### Medium Model (Balanced)
```bash
python training/train_cnn.py \
    --base-channels 128 \
    --num-blocks 4 \
    --batch-size 128 \
    --lr 0.001
```
- Parameters: ~400K
- Training time: ~3-4 hours
- Recommended for most users

### Large Model (Best Performance)
```bash
python training/train_cnn.py \
    --base-channels 256 \
    --num-blocks 6 \
    --batch-size 64 \
    --lr 0.0005 \
    --optimizer adamw
```
- Parameters: ~1.8M
- Training time: ~8-10 hours
- Best accuracy, slower training

## Optimizer Comparison

### Adam
- Good default choice
- Adaptive learning rates
- Fast convergence

```bash
--optimizer adam --lr 0.001
```

### AdamW
- Adam with better weight decay
- Better generalization
- Recommended for larger models

```bash
--optimizer adamw --lr 0.0005 --weight-decay 0.01
```

### SGD with Momentum
- More stable training
- Better final performance
- Requires careful LR tuning

```bash
--optimizer sgd --lr 0.01 --weight-decay 0.0001
```

## Learning Rate Schedules

### Cosine Annealing (Default)
```bash
--scheduler cosine --warmup-epochs 10
```
- Smooth decay with restarts
- Good for most cases

### Step Decay
```bash
--scheduler step --warmup-epochs 10
```
- Reduces LR every 30 epochs
- More aggressive decay

### ReduceLROnPlateau
```bash
--scheduler plateau --warmup-epochs 10
```
- Adapts based on validation loss
- Good for difficult problems

## Monitoring Training

### TensorBoard

View training curves in real-time:

```bash
tensorboard --logdir results/training_curves/cnn
```

Metrics tracked:
- Training/validation loss
- Training/validation accuracy
- Top-2 accuracy
- Learning rate

### Training Logs

Check console output for:
- Epoch progress
- Loss and accuracy
- Learning rate
- ETA

## Evaluation

### Test the Trained Model

```bash
python test_cnn.py \
    --checkpoint checkpoints/cnn/best_model.pth \
    --num-games 100 \
    --greedy
```

### Stochastic Policy

```bash
python test_cnn.py \
    --checkpoint checkpoints/cnn/best_model.pth \
    --num-games 100 \
    --stochastic
```

### Save Results

```bash
python test_cnn.py \
    --checkpoint checkpoints/cnn/best_model.pth \
    --num-games 1000 \
    --output results/cnn_eval.json
```

## Tips and Best Practices

### 1. Data Augmentation
- Always use data augmentation (enabled by default)
- Includes rotation and reflection
- 8x data multiplier

### 2. Regularization
- Use dropout (0.1-0.3)
- Use weight decay (0.0001-0.01)
- Use batch normalization

### 3. Learning Rate
- Start with warmup (5-10 epochs)
- Use cosine annealing for smooth decay
- Monitor for divergence

### 4. Gradient Clipping
- Use gradient clipping (max_norm=1.0)
- Prevents exploding gradients
- Especially important for SGD

### 5. Early Stopping
- Set patience to 25-50 epochs
- Saves time on overfitting
- Always keeps best model

### 6. Checkpointing
- Save every 25 epochs
- Keep last 3 checkpoints
- Always save best model

## Troubleshooting

### Training Loss Not Decreasing
- Reduce learning rate
- Check data preprocessing
- Increase model capacity

### Overfitting (Train >> Val)
- Increase dropout
- Increase weight decay
- Add more training data
- Use smaller model

### Underfitting (Both High)
- Increase model capacity
- Reduce regularization
- Train longer
- Increase learning rate

### NaN Loss
- Reduce learning rate
- Check data normalization
- Use gradient clipping
- Switch to AdamW

## Expected Results

### After 50 Epochs
- Train accuracy: ~60-70%
- Val accuracy: ~55-65%
- Average score: 2000-5000

### After 100 Epochs
- Train accuracy: ~70-80%
- Val accuracy: ~65-75%
- Average score: 5000-10000

### After 200 Epochs
- Train accuracy: ~75-85%
- Val accuracy: ~70-80%
- Average score: 10000-20000
- 2048 tile: 20-40% of games

## Next Steps

After training:

1. **Evaluate Performance**: Run `test_cnn.py` to get game statistics
2. **Compare Models**: Compare with Transformer and AlphaZero
3. **Fine-tune**: Adjust hyperparameters based on results
4. **Export**: Save best model for deployment
