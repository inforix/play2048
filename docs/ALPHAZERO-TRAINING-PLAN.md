# AlphaZero 训练方案 - 2048 游戏

## 概述

AlphaZero 是一种强化学习方法，通过**自我对弈**和**蒙特卡洛树搜索（MCTS）**实现智能体的持续改进。不同于监督学习方法（从专家数据学习），AlphaZero 通过自我博弈探索和学习最优策略，理论上可以超越人类/算法专家水平。

### AlphaZero vs 监督学习

| 特性 | 监督学习 (Method 3) | AlphaZero |
|------|-------------------|-----------|
| 数据来源 | 专家演示（Expectimax） | 自我对弈 |
| 性能上限 | 受限于专家水平 | 可超越专家 |
| 训练时间 | 中等（200 epochs） | 长（数百次迭代） |
| 探索能力 | 无 | 通过 MCTS 探索 |
| 训练数据质量 | 高（专家级） | 逐步提升 |

---

## AlphaZero 核心组件

### 1. 神经网络架构（Dual Network）

```
输入: (batch, 1, 4, 4) - 棋盘状态

共享主干（Shared Backbone）:
  ResBlock 1:
    - Conv2d(1 → 128, kernel=3, padding=1)
    - BatchNorm2d(128)
    - ReLU
    - Conv2d(128 → 128, kernel=3, padding=1)
    - BatchNorm2d(128)
    - Skip connection + ReLU
  
  ResBlock 2:
    - Conv2d(128 → 256, kernel=3, padding=1)
    - BatchNorm2d(256)
    - ReLU
    - Conv2d(256 → 256, kernel=3, padding=1)
    - BatchNorm2d(256)
    - Skip connection (1×1 conv) + ReLU
  
  ResBlock 3-4: 同上

策略头（Policy Head）:
  - Flatten → Linear(4096 → 256)
  - ReLU + Dropout(0.3)
  - Linear(256 → 4)
  - 输出: (batch, 4) - 动作概率 logits

价值头（Value Head）:
  - Flatten → Linear(4096 → 256)
  - ReLU + Dropout(0.3)
  - Linear(256 → 128) → ReLU
  - Linear(128 → 1) → Tanh
  - 输出: (batch, 1) - 局面价值 [-1, 1]
```

**网络输出**：
- **策略 π(a|s)**：4 个动作的概率分布（经过 softmax）
- **价值 v(s)**：当前局面的胜率估计（-1 = 必输，+1 = 必胜）

---

### 2. 蒙特卡洛树搜索（MCTS）

MCTS 用于**增强神经网络的策略**，通过模拟对弈来评估不同动作的长期价值。

#### MCTS 四个阶段

1. **选择（Selection）**
   - 从根节点开始，选择 UCB 值最高的子节点
   - UCB 公式：`UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`
     - `Q(s,a)`: 平均价值（累积价值 / 访问次数）
     - `P(s,a)`: 神经网络预测的策略先验
     - `N(s)`: 父节点访问次数
     - `N(s,a)`: 动作 a 的访问次数
     - `c_puct`: 探索常数（通常 1.0-2.0）

2. **扩展（Expansion）**
   - 到达叶子节点后，使用神经网络评估：
     - 获取策略先验 `P(s,a)` 用于所有合法动作
     - 获取价值估计 `v(s)`
   - 创建所有合法动作的子节点

3. **评估（Evaluation）**
   - 使用神经网络的价值输出 `v(s)` 作为叶子节点的评估

4. **回溯（Backpropagation）**
   - 将价值 `v` 沿着搜索路径向上回传
   - 更新每个节点的访问计数 `N` 和累积价值 `W`
   - `Q(s,a) = W(s,a) / N(s,a)`

#### 2048 游戏的 MCTS 适配

**挑战**：2048 是**单人游戏 + 随机性**（新砖块随机出现），不同于围棋的双人完全信息博弈。

**解决方案**：
- **玩家节点（Max 节点）**：选择 4 个移动方向之一
- **机会节点（Chance 节点）**：模拟随机砖块生成
  - 90% 概率生成 2
  - 10% 概率生成 4
  - 在所有空位置中随机选择

**MCTS 树结构**：
```
          根节点（玩家）
         /    |    \    \
       上    下    左    右  ← 玩家动作
       |     |     |     |
    机会  机会  机会  机会   ← 随机砖块
    / \   / \   / \   / \
   2   4 2   4 2   4 2   4  ← 砖块值
   |   |  |   |  |   |  |   |
  玩家 ...              ...  ← 继续搜索
```

**价值回传**：
- 玩家节点：取**最大值**（选择最优动作）
- 机会节点：取**期望值**（0.9 * v_2 + 0.1 * v_4，所有位置平均）

---

### 3. 自我对弈（Self-Play）

**流程**：
1. 初始化空棋盘，随机放置两个砖块
2. 对每个回合：
   - 运行 MCTS（例如 100-800 次模拟）
   - 根据 MCTS 访问计数生成改进的策略 π_MCTS
   - 使用温度参数采样动作（早期探索，后期利用）
   - 执行动作，随机生成新砖块
   - 记录 (s_t, π_t, z_t) 到训练数据
     - `s_t`: 棋盘状态
     - `π_t`: MCTS 改进的策略（访问计数分布）
     - `z_t`: 游戏结果（稍后填充）
3. 游戏结束后：
   - 计算结果 `z`：
     - `+1` 如果达到 2048（获胜）
     - `0` 如果达到 1024/512（中等）
     - `-1` 如果只达到 256 或更低（失败）
     - 或使用归一化分数：`(score - mean) / std`
4. 将 `z` 填充到所有步骤的 `z_t`

**温度参数**：
- **前 30 步**：`τ = 1.0`（探索，增加随机性）
- **后续步骤**：`τ → 0`（利用，选择最佳动作）
- 动作采样：`π_sample(a) ∝ N(a)^(1/τ)`

**数据增强**：
- 8 重对称性（4 旋转 × 2 翻转）
- 训练数据增加 8 倍

---

### 4. 训练循环

**损失函数**：
```python
total_loss = (z - v)^2 - π^T * log(p) + c * ||θ||^2
```
- `(z - v)^2`: 价值损失（MSE）
- `-π^T * log(p)`: 策略损失（交叉熵）
- `c * ||θ||^2`: L2 正则化

**训练流程**：
1. 从经验池中采样 mini-batch（例如 256 samples）
2. 计算损失并更新网络参数
3. 定期评估当前网络 vs 最佳网络
4. 如果新网络胜率 > 55%，更新最佳网络

**超参数**：
- Batch size: 256
- Learning rate: 0.001（初始），动态调整
- Optimizer: SGD with momentum 0.9
- Weight decay: 1e-4
- Training epochs per iteration: 10-20

---

## AlphaZero 训练流程

### 第 0 阶段：初始化（Bootstrap）

**选项 1：从零开始**
- 使用随机初始化的网络
- 前 10-20 局游戏质量较差
- 需要更多迭代才能收敛

**选项 2：从监督学习初始化（推荐）**
- 使用已训练的 transformer/dual network
- 快速启动，减少迭代次数
- 更快达到超越专家水平

**决策**：使用已有的 transformer 模型作为初始化（best_model.pth），转换为 ResNet 架构。

---

### 第 1 阶段：MCTS 实现

**文件**：`training/mcts.py`

**核心类**：
```python
class MCTSNode:
    """MCTS 树节点"""
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state           # 棋盘状态
        self.parent = parent         # 父节点
        self.action = action         # 到达此节点的动作
        self.prior = prior           # 策略先验 P(s,a)
        self.children = {}           # 子节点 {action: node}
        self.visit_count = 0         # N(s,a)
        self.total_value = 0.0       # W(s,a)
        self.is_chance_node = False  # 是否为机会节点
    
    def q_value(self):
        """平均价值 Q(s,a)"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def ucb_score(self, c_puct=1.4):
        """UCB 分数"""
        if self.visit_count == 0:
            return float('inf')
        exploration = c_puct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value() + exploration

class MCTS:
    """蒙特卡洛树搜索"""
    def __init__(self, model, num_simulations=100, c_puct=1.4):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    
    def search(self, state):
        """执行 MCTS 搜索，返回改进的策略"""
        root = MCTSNode(state)
        
        for _ in range(self.num_simulations):
            node = root
            # 1. Selection
            while node.children:
                node = self._select_child(node)
            
            # 2. Expansion & Evaluation
            if not is_terminal(node.state):
                value = self._expand_and_evaluate(node)
            else:
                value = self._terminal_value(node.state)
            
            # 3. Backpropagation
            self._backpropagate(node, value)
        
        # 返回访问计数分布作为改进策略
        return self._get_action_probs(root)
    
    def _select_child(self, node):
        """选择 UCB 最大的子节点"""
        return max(node.children.values(), key=lambda n: n.ucb_score(self.c_puct))
    
    def _expand_and_evaluate(self, node):
        """扩展节点并评估"""
        policy, value = self.model.predict(node.state)
        
        # 创建所有合法动作的子节点
        for action in range(4):
            if is_valid_action(node.state, action):
                # 创建机会节点
                chance_node = MCTSNode(node.state, parent=node, action=action, prior=policy[action])
                chance_node.is_chance_node = True
                node.children[action] = chance_node
        
        return value
    
    def _backpropagate(self, node, value):
        """向上回传价值"""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # 翻转价值（对手视角）
            node = node.parent
```

**实现要点**：
- 处理机会节点（随机砖块）
- 高效的棋盘状态表示
- 虚拟损失（Virtual Loss）用于并行搜索
- Dirichlet 噪声用于根节点探索（训练时）

---

### 第 2 阶段：Dual Network 实现

**文件**：`models/dual/alphazero_network.py`

**网络结构**：
```python
class AlphaZeroNetwork(nn.Module):
    """AlphaZero 双头网络"""
    def __init__(self, num_blocks=4, channels=256):
        super().__init__()
        
        # 初始卷积
        self.conv_input = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # ResNet blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(channels) for _ in range(num_blocks)
        ])
        
        # 策略头
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 4 * 4, 4)
        
        # 价值头
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 4 * 4, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        # Shared backbone
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # Logits
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v
```

---

### 第 3 阶段：自我对弈数据生成

**文件**：`training/selfplay.py`

**核心函数**：
```python
def self_play_game(model, mcts_simulations=100, temperature=1.0):
    """
    使用 MCTS 进行一局自我对弈
    
    Returns:
        training_examples: List[(state, mcts_policy, value)]
    """
    game = Game2048()
    training_examples = []
    move_count = 0
    
    while not game.game_over and move_count < 5000:
        # 运行 MCTS
        mcts = MCTS(model, num_simulations=mcts_simulations)
        mcts_policy = mcts.search(game.get_state())
        
        # 记录训练样本（结果稍后填充）
        training_examples.append({
            'state': game.get_state().clone(),
            'policy': mcts_policy.copy(),
            'value': None  # 游戏结束后填充
        })
        
        # 温度采样动作
        tau = 1.0 if move_count < 30 else 0.1
        action = sample_action(mcts_policy, temperature=tau)
        
        # 执行动作
        game.move(action)
        move_count += 1
    
    # 计算游戏结果
    max_tile = np.max(game.board)
    if max_tile >= 2048:
        result = 1.0
    elif max_tile >= 1024:
        result = 0.5
    elif max_tile >= 512:
        result = 0.0
    else:
        result = -0.5
    
    # 填充结果到所有步骤
    for example in training_examples:
        example['value'] = result
    
    return training_examples, {
        'score': game.score,
        'max_tile': max_tile,
        'moves': move_count
    }
```

**并行自我对弈**：
- 使用多进程生成数据（例如 8 个进程）
- 每次迭代生成 100-500 局游戏
- 数据增强（8 倍对称性）

---

### 第 4 阶段：训练循环

**文件**：`training/train_alphazero.py`

**主训练循环**：
```python
def train_alphazero(iterations=100, games_per_iteration=100):
    """AlphaZero 主训练循环"""
    
    # 初始化
    model = AlphaZeroNetwork()
    best_model = copy.deepcopy(model)
    replay_buffer = ReplayBuffer(max_size=500000)
    
    for iteration in range(iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{iterations}")
        print(f"{'='*60}")
        
        # 1. 自我对弈生成数据
        print(f"🎮 Self-play: Generating {games_per_iteration} games...")
        new_examples = []
        stats = []
        
        for game_idx in tqdm(range(games_per_iteration)):
            examples, game_stats = self_play_game(
                model, 
                mcts_simulations=100 if iteration < 10 else 200
            )
            new_examples.extend(examples)
            stats.append(game_stats)
        
        # 增强并添加到经验池
        augmented_examples = augment_examples(new_examples)
        replay_buffer.add(augmented_examples)
        
        print(f"  Generated {len(new_examples)} examples (×8 aug = {len(augmented_examples)})")
        print(f"  Avg score: {np.mean([s['score'] for s in stats]):.0f}")
        print(f"  Win rate: {np.mean([s['max_tile'] >= 2048 for s in stats])*100:.1f}%")
        
        # 2. 训练网络
        print(f"🔧 Training network...")
        train_metrics = train_network(
            model, 
            replay_buffer, 
            epochs=10, 
            batch_size=256
        )
        
        print(f"  Policy loss: {train_metrics['policy_loss']:.4f}")
        print(f"  Value loss: {train_metrics['value_loss']:.4f}")
        
        # 3. 评估新模型
        if iteration % 5 == 0:
            print(f"⚔️  Evaluating: New model vs Best model...")
            win_rate = evaluate_models(model, best_model, num_games=50)
            print(f"  New model win rate: {win_rate*100:.1f}%")
            
            if win_rate > 0.55:
                print(f"  ✓ New model is better! Updating best model.")
                best_model = copy.deepcopy(model)
            else:
                print(f"  ✗ Best model retained.")
        
        # 4. 保存检查点
        if iteration % 10 == 0:
            save_checkpoint({
                'iteration': iteration,
                'model': model.state_dict(),
                'best_model': best_model.state_dict(),
                'replay_buffer': replay_buffer.get_state()
            }, f'checkpoints/alphazero_iter{iteration}.pth')
    
    return best_model
```

---

### 第 5 阶段：评估与对比

**对比基准**：
- Expectimax (深度 4)：~80% 胜率
- 监督学习 Transformer（200 epochs）：预期 70% 胜率
- AlphaZero（100 次迭代）：目标 85%+ 胜率

**评估指标**：
1. **胜率**：达到 2048 的比例
2. **平均分数**：所有游戏的平均得分
3. **最大砖块分布**：512/1024/2048/4096
4. **对战胜率**：vs Expectimax, vs 监督模型

**评估脚本**：`evaluation/evaluate_alphazero.py`

---

## 实现计划

### 阶段 1：MCTS 实现（第 1-2 天）
- [ ] 实现 `MCTSNode` 类
- [ ] 实现 `MCTS` 类（选择、扩展、回溯）
- [ ] 处理机会节点（随机砖块）
- [ ] 测试 MCTS 与随机策略网络

### 阶段 2：Dual Network（第 3 天）
- [ ] 实现 `AlphaZeroNetwork`（ResNet + 双头）
- [ ] 实现 `ResBlock`
- [ ] 测试前向传播和梯度流
- [ ] 计算网络参数量

### 阶段 3：自我对弈（第 4-5 天）
- [ ] 实现 `self_play_game()`
- [ ] 实现温度采样
- [ ] 实现 `ReplayBuffer`
- [ ] 测试并行自我对弈

### 阶段 4：训练循环（第 6-7 天）
- [ ] 实现 `train_network()`
- [ ] 实现 `evaluate_models()`
- [ ] 实现主训练循环 `train_alphazero()`
- [ ] 测试完整训练流程（10 次迭代）

### 阶段 5：完整训练（第 8-14 天）
- [ ] 运行 100 次迭代
- [ ] 监控训练曲线
- [ ] 定期评估性能
- [ ] 与监督学习模型对比

### 阶段 6：优化与分析（第 15-16 天）
- [ ] 超参数调优（MCTS 模拟次数、学习率等）
- [ ] 可视化 MCTS 搜索树
- [ ] 分析策略改进过程
- [ ] 撰写结果报告

---

## 超参数配置

### MCTS 参数
- **模拟次数（num_simulations）**：
  - 初期：100（快速迭代）
  - 后期：200-400（更精确的策略）
  - 评估：800（最佳性能）
- **探索常数（c_puct）**：1.4
- **Dirichlet 噪声**：α=0.3, ε=0.25（训练时根节点）
- **温度参数（temperature）**：
  - 前 30 步：τ=1.0
  - 后续步骤：τ=0.1

### 网络架构
- **ResNet blocks**：4-6 层
- **Channels**：256
- **Dropout**：0.3（策略/价值头）

### 训练参数
- **每次迭代游戏数**：100-200
- **经验池大小**：500,000 samples
- **Batch size**：256
- **Learning rate**：0.001 → 0.0001（衰减）
- **Optimizer**：SGD with momentum 0.9
- **Weight decay**：1e-4
- **训练 epochs/iteration**：10-20

### 评估参数
- **对战游戏数**：50-100
- **更新阈值**：55% 胜率

---

## 预期结果

### 短期目标（20-30 次迭代，~3-5 天）
- [ ] 胜率 > 50%（超过随机 baseline）
- [ ] 平均分数 > 5,000
- [ ] 稳定的策略改进曲线

### 中期目标（50-70 次迭代，~1 周）
- [ ] 胜率 > 70%（接近 Expectimax）
- [ ] 平均分数 > 12,000
- [ ] 超越监督学习模型

### 长期目标（100+ 次迭代，~2 周）
- [ ] 胜率 > 85%（超越 Expectimax）
- [ ] 平均分数 > 18,000
- [ ] 10%+ 游戏达到 4096
- [ ] 策略具有明确的长期规划能力

---

## 与监督学习的对比

| 指标 | 监督学习（Transformer） | AlphaZero |
|------|----------------------|-----------|
| 训练数据 | 500 games (Expectimax) | 10,000+ games (self-play) |
| 训练时间 | ~50 小时（200 epochs） | ~100-200 小时（100 iterations） |
| 性能上限 | ~75% 胜率（受限于专家） | 85%+ 胜率（持续改进） |
| 泛化能力 | 中等 | 强（探索更多状态） |
| 数据效率 | 高 | 低（需要大量自我对弈） |
| 实现复杂度 | 中 | 高（MCTS + 训练循环） |

---

## 技术挑战

### 1. 随机性处理
- **问题**：2048 有随机砖块生成，不同于围棋
- **解决**：机会节点 + 期望值回传

### 2. 价值评估
- **问题**：游戏结果不是二元（胜/负）
- **解决**：归一化分数或分级奖励（2048=+1, 1024=+0.5, etc.）

### 3. 训练时间
- **问题**：自我对弈 + MCTS 计算量大
- **解决**：
  - 并行自我对弈（多进程）
  - 降低初期 MCTS 模拟次数
  - GPU 加速网络推理

### 4. 探索 vs 利用
- **问题**：早期模型质量差，需要探索
- **解决**：
  - Dirichlet 噪声增加根节点探索
  - 温度参数控制采样随机性
  - 经验池保留历史数据

---

## 下一步行动

1. **创建目录结构**：
   ```
   models/dual/
   training/mcts.py
   training/selfplay.py
   training/train_alphazero.py
   evaluation/evaluate_alphazero.py
   ```

2. **实现 MCTS**：从简单版本开始，逐步优化

3. **实现 Dual Network**：基于 ResNet 架构

4. **测试自我对弈**：确保数据生成正确

5. **小规模训练**：10 次迭代验证流程

6. **完整训练**：100+ 次迭代达到最优性能

---

**文档版本**：1.0  
**创建日期**：2026-01-08  
**状态**：待实现

**参考资源**：
- AlphaGo Zero 论文：Mastering the game of Go without human knowledge
- AlphaZero 论文：A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play
- 2048 Expectimax 实现：scripts/generate_dataset.py
