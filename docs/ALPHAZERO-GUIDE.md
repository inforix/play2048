# AlphaZero 训练指南 - 2048 游戏

## 🎯 快速开始

### 测试 AlphaZero 管道（3 次迭代，~35 分钟）

```bash
cd /Users/wyp/develop/play2048
uv run python test_alphazero.py
```

**测试结果示例**：
```
AlphaZero Training - 2048
Device: mps
Model parameters: 4,875,653
Iterations: 3

Iteration 1/3:
  Self-play: 5 games, 4,856 examples (×8 aug), avg_score=0, win_rate=0%
  Training: Loss=8.71, Policy=8.37, Value=0.67
  Time: 1.3m

Iteration 2/3:
  Self-play: 5 games, 3,608 examples
  Training: Loss=6.28, Policy=6.12, Value=0.31
  Evaluation: 50% win rate vs best
  Time: 17.9m

Iteration 3/3:
  Self-play: 5 games, 3,376 examples
  Training: Loss=5.44, Policy=5.30, Value=0.29
  Time: 16.1m

✅ Total time: 35.3m
```

### 完整训练（100 次迭代，~7-10 天）

```bash
# 基础配置（推荐用于测试）
uv run python training/train_alphazero.py \
    --iterations 100 \
    --games 100 \
    --mcts-sims 100 \
    --epochs 10 \
    --batch-size 256 \
    --eval-interval 5 \
    --eval-games 50

# 快速配置（更快迭代，但性能稍低）
uv run python training/train_alphazero.py \
    --iterations 100 \
    --games 50 \
    --mcts-sims 50 \
    --epochs 5 \
    --batch-size 128

# 高质量配置（更慢但性能更好）
uv run python training/train_alphazero.py \
    --iterations 150 \
    --games 200 \
    --mcts-sims 200 \
    --epochs 15 \
    --batch-size 256 \
    --eval-interval 5
```

### 从检查点恢复训练

```bash
uv run python training/train_alphazero.py \
    --resume checkpoints/alphazero/checkpoint_iter50.pth \
    --iterations 100
```

---

## 📊 训练参数说明

### 核心参数

| 参数 | 默认值 | 说明 | 建议范围 |
|------|-------|------|---------|
| `--iterations` | 100 | 训练迭代次数 | 50-150 |
| `--games` | 100 | 每次迭代的自我对弈游戏数 | 50-200 |
| `--mcts-sims` | 100 | MCTS 每步模拟次数 | 50-400 |
| `--epochs` | 10 | 每次迭代的训练轮数 | 5-20 |
| `--batch-size` | 256 | 训练批次大小 | 128-512 |

### 评估参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--eval-interval` | 5 | 每 N 次迭代评估一次 |
| `--eval-games` | 50 | 评估游戏数量 |

### 系统参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--device` | auto | cpu / cuda / mps / auto |
| `--output-dir` | checkpoints/alphazero | 输出目录 |
| `--save-interval` | 10 | 保存检查点间隔 |

---

## 📈 训练监控

### TensorBoard 可视化

```bash
# 启动 TensorBoard
tensorboard --logdir checkpoints/alphazero/tensorboard

# 在浏览器访问
http://localhost:6006
```

**监控指标**：
- `SelfPlay/AvgScore` - 自我对弈平均分数
- `SelfPlay/WinRate` - 胜率（达到 2048）
- `Train/Loss` - 总损失
- `Train/PolicyLoss` - 策略损失
- `Train/ValueLoss` - 价值损失
- `Eval/WinRate` - 新模型 vs 最佳模型胜率

### 检查点文件

训练过程中会保存：
- `checkpoint_iter10.pth` - 每 10 次迭代保存
- `final_model.pth` - 最终模型
- `tensorboard/` - TensorBoard 日志

---

## 🎮 使用训练好的模型玩游戏

### 方法 1：使用评估脚本

```python
import torch
from models.dual import AlphaZeroNetwork
from training.mcts import MCTS
import numpy as np

# 加载模型
device = torch.device('mps')  # 或 'cpu', 'cuda'
model = AlphaZeroNetwork(num_blocks=4, channels=256)
checkpoint = torch.load('checkpoints/alphazero/final_model.pth', weights_only=False)
model.load_state_dict(checkpoint['best_model'])
model.to(device)
model.eval()

# 创建 MCTS
mcts = MCTS(model, device, num_simulations=200)

# 玩游戏
from training.selfplay import Game2048

game = Game2048()
while not game.game_over:
    state = game.get_state().numpy()
    policy = mcts.search(state, add_noise=False)
    action = np.argmax(policy)
    game.move(action)

print(f"Final score: {game.score}")
print(f"Max tile: {game.get_max_tile()}")
```

### 方法 2：批量测试

```python
from training.selfplay import self_play_game

# 测试 100 局
results = []
for i in range(100):
    _, stats = self_play_game(model, device, mcts_simulations=200, add_noise=False)
    results.append(stats)

# 统计
import pandas as pd
df = pd.DataFrame(results)
print(f"Win rate: {df['won'].mean()*100:.1f}%")
print(f"Avg score: {df['score'].mean():.0f}")
print(f"Max tile distribution:\n{df['max_tile'].value_counts().sort_index()}")
```

---

## 📊 预期训练进度

### 迭代 1-20（探索阶段）

**目标**：学习基本策略
- 胜率：0-20%
- 平均分数：500-2000
- 最大砖块：主要 128-256
- 训练损失：逐渐下降（8.0 → 5.0）

**特点**：
- 模型在探索各种策略
- 自我对弈游戏质量较低
- MCTS 开始起作用

### 迭代 21-50（稳定阶段）

**目标**：形成稳定策略
- 胜率：20-50%
- 平均分数：3000-8000
- 最大砖块：主要 256-512，偶尔 1024
- 训练损失：继续下降（5.0 → 3.5）

**特点**：
- 开始学习角落保护
- 合并策略更有效
- 模型评估准确性提高

### 迭代 51-100（优化阶段）

**目标**：接近/超越 Expectimax
- 胜率：50-85%
- 平均分数：10000-18000
- 最大砖块：主要 1024-2048，5-10% 达到 4096
- 训练损失：接近收敛（3.5 → 2.5）

**特点**：
- 长期规划能力增强
- 策略接近最优
- 开始超越专家水平

---

## ⚙️ 超参数调优建议

### 1. MCTS 模拟次数

**影响**：搜索质量 vs 速度

```bash
# 快速迭代（开发/调试）
--mcts-sims 50

# 平衡（推荐训练）
--mcts-sims 100

# 高质量（后期/评估）
--mcts-sims 200-400
```

### 2. 自我对弈游戏数

**影响**：数据多样性 vs 迭代速度

```bash
# 快速迭代
--games 50

# 标准配置
--games 100

# 数据丰富
--games 200
```

### 3. 训练轮数

**影响**：网络拟合 vs 过拟合风险

```bash
# 轻量训练（避免过拟合）
--epochs 5

# 标准配置
--epochs 10

# 充分训练（大数据集）
--epochs 15-20
```

### 4. 批次大小

**影响**：训练稳定性 vs 内存占用

```bash
# 小内存设备
--batch-size 128

# 推荐配置
--batch-size 256

# 高内存设备
--batch-size 512
```

---

## 🔧 故障排除

### 问题 1：训练损失不下降

**可能原因**：
- 学习率过低或过高
- MCTS 模拟次数太少
- 数据质量差

**解决方案**：
```bash
# 增加 MCTS 模拟
--mcts-sims 200

# 增加自我对弈游戏
--games 200

# 调整学习率（需修改代码）
# 在 train_alphazero.py 中设置 learning_rate=0.002
```

### 问题 2：胜率长期停滞

**可能原因**：
- 陷入局部最优
- 探索不足

**解决方案**：
- 增加 Dirichlet 噪声（代码中已包含）
- 提高温度参数持续时间（修改 `temperature_moves`）
- 重新初始化模型（从零开始）

### 问题 3：显存/内存不足

**解决方案**：
```bash
# 减小批次大小
--batch-size 64

# 减小网络规模（需修改代码）
# 在 train_alphazero.py 中：
# model = AlphaZeroNetwork(num_blocks=3, channels=128)

# 减少 MCTS 模拟
--mcts-sims 50
```

### 问题 4：训练速度太慢

**优化方案**：
```bash
# 减少自我对弈游戏
--games 50

# 减少 MCTS 模拟（早期迭代）
--mcts-sims 50

# 减少评估频率
--eval-interval 10

# 使用 GPU（如果可用）
--device cuda  # 或 mps (Mac)
```

---

## 📚 高级技巧

### 1. 课程学习（Curriculum Learning）

逐步增加难度：

```bash
# 阶段 1：快速探索（1-30 次迭代）
--mcts-sims 50 --games 50

# 阶段 2：稳定训练（31-70 次迭代）
--mcts-sims 100 --games 100

# 阶段 3：精细优化（71-100 次迭代）
--mcts-sims 200 --games 150
```

### 2. 从监督学习初始化

```python
# 加载预训练的 Transformer 模型权重
# 注意：需要手动映射参数（架构不同）
# 这可以加速前 10-20 次迭代
```

### 3. 集成多个模型

训练多个独立模型并集成预测：

```python
models = [model1, model2, model3]
ensemble_policy = sum([mcts.search(state) for mcts in mcts_list]) / 3
```

---

## 📈 性能对比

### vs 监督学习

| 指标 | 监督学习（200 epochs） | AlphaZero（100 iters） |
|------|---------------------|---------------------|
| 训练时间 | ~50 小时 | ~200-300 小时 |
| 胜率 | 70%（预期） | 85%（目标） |
| 平均分数 | ~12,000 | ~18,000 |
| 性能上限 | 受限于专家 | 可超越专家 |
| 实现复杂度 | 中 | 高 |

### vs Expectimax

| 指标 | Expectimax | AlphaZero（目标） |
|------|-----------|---------------|
| 胜率 | 80% | 85%+ |
| 平均分数 | ~15,000 | ~18,000 |
| 推理时间 | 5-10ms | 50-100ms |
| 需要训练 | 否 | 是 |

---

## 🎯 下一步计划

1. **完成初步训练**（20-30 次迭代）
   - 验证训练流程
   - 观察损失曲线
   - 调整超参数

2. **中期评估**（50 次迭代）
   - vs Expectimax 对战测试
   - 分析策略差异
   - 优化 MCTS 参数

3. **完整训练**（100 次迭代）
   - 达到目标性能
   - 保存最佳模型
   - 撰写结果报告

4. **模型压缩**（可选）
   - 知识蒸馏到小模型
   - 量化加速推理
   - 部署到 Web 浏览器

---

## 📝 引用

如果使用此实现，请引用：

```bibtex
@misc{play2048_alphazero,
  title={AlphaZero for 2048 Game},
  author={Your Name},
  year={2026},
  url={https://github.com/inforix/play2048}
}
```

**参考论文**：
- Silver et al. (2017) - AlphaGo Zero
- Silver et al. (2018) - AlphaZero
- Browne et al. (2012) - MCTS Survey

---

**最后更新**：2026-01-08  
**版本**：1.0  
**状态**：✅ 完整实现并测试通过
