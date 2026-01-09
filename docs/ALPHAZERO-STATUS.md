# AlphaZero Implementation Status

## 已完成的组件 ✅

### 1. 核心文档
- **[docs/ALPHAZERO-TRAINING-PLAN.md](docs/ALPHAZERO-TRAINING-PLAN.md)** - 完整的 AlphaZero 训练方案（中文）
  - AlphaZero vs 监督学习对比
  - 神经网络架构设计
  - MCTS 算法详解
  - 自我对弈流程
  - 训练循环设计
  - 实现路线图

- **[docs/ALPHAZERO-IMPLEMENTATION-CHECKLIST.md](docs/ALPHAZERO-IMPLEMENTATION-CHECKLIST.md)** - 实现进度清单
  - 7 个阶段的详细任务
  - 时间估算：17-24 天
  - 当前进度追踪

### 2. MCTS (蒙特卡洛树搜索) ✅
- **文件**: [training/mcts.py](training/mcts.py)
- **功能**:
  - `MCTSNode` 类 - 树节点，支持 UCB 计算
  - `MCTS` 类 - 主搜索算法
  - 处理 2048 的单人游戏 + 随机性（机会节点）
  - 与神经网络集成（策略先验 + 价值估计）
  - Dirichlet 噪声用于根节点探索
  - 完整的游戏逻辑（滑动、合并砖块）

- **测试结果**:
  ```
  Initial board:
  [[2. 0. 0. 0.]
   [0. 4. 0. 0.]
   [0. 0. 0. 0.]
   [0. 0. 0. 0.]]
  
  Action probabilities after 50 simulations:
  Up: 0.640
  Down: 0.080
  Left: 0.160
  Right: 0.120
  
  Selected action: Up
  ✓ MCTS test passed!
  ```

- **关键特性**:
  - UCB 探索 vs 利用平衡
  - 处理随机砖块生成（机会节点）
  - 高效的棋盘状态表示
  - 支持可配置的模拟次数（默认 100）

### 3. AlphaZero 双头网络 ✅
- **文件**: 
  - [models/dual/resblock.py](models/dual/resblock.py) - 残差块
  - [models/dual/alphazero_network.py](models/dual/alphazero_network.py) - 主网络
  - [models/dual/__init__.py](models/dual/__init__.py) - 包初始化

- **网络架构**:
  ```
  Input (batch, 4, 4)
  -> Conv(1 -> channels) + BN + ReLU
  -> ResBlock × num_blocks
  -> Policy Head -> (batch, 4) logits
  -> Value Head -> (batch, 1) value [-1, 1]
  ```

- **ResBlock 设计**:
  - 两层卷积 (3×3, padding=1)
  - BatchNorm + ReLU
  - Skip connection (通道变化时使用 1×1 卷积)
  - 参数量：~74K (128 channels)

- **三种配置**:
  | 配置 | Blocks | Channels | 参数量 |
  |------|--------|----------|--------|
  | Small | 4 | 128 | 1.3M |
  | Medium | 4 | 256 | 4.9M |
  | Large | 6 | 256 | 7.2M |

- **测试结果**:
  ```
  Small configuration (blocks=4, channels=128):
    ✓ Forward pass: (2, 4, 4) -> policy (2, 4), value (2, 1)
    ✓ Predict: policy sums to 1.000000
    ✓ Gradients flow correctly
    ✓ Trainable parameters: 1,325,061
  
  Medium configuration (blocks=4, channels=256):
    ✓ Trainable parameters: 4,875,653
  
  Memory estimates (single forward pass):
    Model size: ~18.6 MB
    Activation memory (batch=64): ~1.0 MB
  ```

- **特点**:
  - 共享主干提取棋盘特征
  - 策略头：预测 4 个动作的概率
  - 价值头：评估当前局面（-1 到 +1）
  - Dropout (0.3) 防止过拟合
  - He 初始化保证训练稳定性

---

## 待实现的组件 ⏳

### 4. 自我对弈数据生成 (进行中)
- **文件**: `training/selfplay.py`
- **需要实现**:
  - `ReplayBuffer` - 经验池（容量 500K samples）
  - `self_play_game()` - 使用 MCTS 进行一局游戏
  - 温度采样（前 30 步 τ=1.0，后续 τ=0.1）
  - 数据增强（8 重对称性）
  - 并行自我对弈（多进程）

### 5. 训练循环
- **文件**: `training/train_alphazero.py`
- **需要实现**:
  - `train_network()` - 从经验池训练
  - `evaluate_models()` - 新旧模型对战评估
  - `train_alphazero()` - 主训练循环
  - 损失函数：`L = (z - v)² - π^T log(p) + c||θ||²`
  - 检查点保存与恢复
  - TensorBoard 日志

### 6. 评估脚本
- **文件**: `evaluation/evaluate_alphazero.py`
- **需要实现**:
  - 使用训练模型玩游戏
  - vs Expectimax 对战评估
  - vs 监督学习模型对战
  - 可视化训练曲线
  - MCTS 搜索树可视化

---

## 快速开始指南

### 测试已实现的组件

1. **测试 MCTS**:
   ```bash
   cd /Users/wyp/develop/play2048
   uv run python training/mcts.py
   ```

2. **测试 ResBlock**:
   ```bash
   uv run python models/dual/resblock.py
   ```

3. **测试 AlphaZero 网络**:
   ```bash
   uv run python models/dual/alphazero_network.py
   ```

### 下一步计划

1. **实现自我对弈** (预计 2 天):
   - 创建 `training/selfplay.py`
   - 实现 `ReplayBuffer` 类
   - 实现 `self_play_game()` 函数
   - 测试数据生成流程

2. **实现训练循环** (预计 2 天):
   - 创建 `training/train_alphazero.py`
   - 实现训练函数
   - 实现模型评估
   - 小规模测试（10 次迭代）

3. **完整训练** (预计 7-14 天):
   - 运行 100 次迭代
   - 监控训练进度
   - 定期评估性能
   - 对比监督学习模型

---

## 技术亮点

### MCTS 适配 2048
- **单人游戏**：不需要对手节点，但需要处理随机性
- **机会节点**：模拟随机砖块生成（90% 为 2，10% 为 4）
- **UCB 公式**：`UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`
- **探索增强**：Dirichlet 噪声在根节点增加探索

### 双头网络设计
- **共享表示**：ResNet 主干提取通用特征
- **策略头**：轻量级（32 通道 + 1 层 FC）
- **价值头**：两层 FC 用于更准确的评估
- **Tanh 激活**：确保价值在 [-1, 1] 范围

### 内存效率
- **Medium 配置**：~4.9M 参数，~19 MB 模型
- **推理内存**：batch=64 时约 1 MB 激活
- **可扩展**：支持 Small/Medium/Large 配置

---

## 性能预期

### 训练收敛时间
- **10 次迭代**：~6-12 小时（验证实现）
- **50 次迭代**：~3-5 天（接近 Expectimax）
- **100 次迭代**：~1-2 周（超越 Expectimax）

### 目标性能
| 迭代次数 | 胜率目标 | 平均分数 | 状态 |
|---------|---------|---------|------|
| 20-30 | > 50% | > 5,000 | 超过随机 |
| 50-70 | > 70% | > 12,000 | 接近 Expectimax |
| 100+ | > 85% | > 18,000 | 超越 Expectimax |

### 对比基准
- **Expectimax**：80% 胜率，~15,000 分
- **监督学习 Transformer**（5 epochs）：0% 胜率，~1,300 分
- **监督学习 Transformer**（200 epochs，预期）：70% 胜率，~12,000 分
- **AlphaZero**（100 iterations，目标）：85% 胜率，~18,000 分

---

## 文件结构

```
play2048/
├── docs/
│   ├── ALPHAZERO-TRAINING-PLAN.md          # 训练方案（中文）✅
│   └── ALPHAZERO-IMPLEMENTATION-CHECKLIST.md  # 进度清单 ✅
├── models/
│   └── dual/
│       ├── __init__.py                     # 包初始化 ✅
│       ├── resblock.py                     # 残差块 ✅
│       └── alphazero_network.py            # 主网络 ✅
├── training/
│   ├── mcts.py                             # MCTS 算法 ✅
│   ├── selfplay.py                         # 自我对弈 ⏳
│   └── train_alphazero.py                  # 训练循环 ⏳
└── evaluation/
    └── evaluate_alphazero.py               # 评估脚本 ⏳
```

**图例**: ✅ 已完成 | ⏳ 待实现

---

## 常见问题

### Q1: AlphaZero 比监督学习慢多少？
**A**: 慢约 3-5 倍，但可以超越专家水平：
- 监督学习：200 epochs × 15 分钟 = 50 小时
- AlphaZero：100 iterations × 2-3 小时 = 200-300 小时
- 收益：性能提升 10-15%（70% → 85% 胜率）

### Q2: 可以从监督学习模型初始化吗？
**A**: 理论上可以，但架构不同：
- Transformer：Attention 机制，862K 参数
- AlphaZero：ResNet 主干，4.9M 参数
- 建议：从头训练，或使用监督数据预热前 10 次迭代

### Q3: MCTS 模拟次数如何选择？
**A**: 权衡性能 vs 速度：
- 训练早期：100 次（快速迭代）
- 训练后期：200-400 次（更精确策略）
- 评估对战：800 次（最佳性能）
- 实际游戏：100-200 次（平衡性能和速度）

### Q4: 能在 CPU 上训练吗？
**A**: 可以但很慢：
- GPU (MPS/CUDA)：1 iteration ~2-3 小时
- CPU：1 iteration ~10-15 小时
- 建议：至少使用 M1/M2 Mac（MPS）或 NVIDIA GPU

### Q5: 需要多少自我对弈游戏？
**A**: 逐步增加：
- 初期（1-20 iterations）：100 games/iteration
- 中期（21-50）：200 games/iteration
- 后期（51-100）：300-500 games/iteration
- 总计：~15,000-30,000 games

---

## 参考资源

### 论文
1. **AlphaGo Zero**: [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
2. **AlphaZero**: [A general reinforcement learning algorithm](https://arxiv.org/abs/1712.01815)
3. **MCTS Survey**: [IEEE TCIAIG Survey](https://ieeexplore.ieee.org/document/6145622)

### 实现参考
- **2048 Expectimax**: [scripts/generate_dataset.py](scripts/generate_dataset.py)
- **Transformer 训练**: [training/train_transformer.py](training/train_transformer.py)
- **游戏模拟**: [evaluation/game_simulator.py](evaluation/game_simulator.py)

---

**最后更新**: 2026-01-08  
**当前进度**: 阶段 1-3 完成（MCTS + 网络架构），阶段 4 进行中（自我对弈）  
**预计完成**: 2026-01-24（完整训练）
