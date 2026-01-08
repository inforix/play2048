# Transformer训练计划 (Method 3)

## 📊 当前状态

- ✅ **训练数据**: 500个游戏记录在 `data/training_games.jsonl` (~50MB)
- ✅ **数据规模**: 平均每局1870步 → 约935,000个训练样本（未增强前）
- ✅ **数据增强后**: 8倍对称变换 → 约**7,500,000个样本**（Transformer理想数据量）

---

## 🎯 训练目标

### 性能指标
- **最低标准**: 测试准确率 ≥ 70%，胜率 ≥ 50%
- **目标性能**: 测试准确率 ≥ 80%，胜率 ≥ 70%（接近Expectimax的80%）
- **挑战目标**: 胜率 ≥ 85%，10%以上的游戏达到4096方块

### 对比基准
- **Expectimax算法**: ~80% 胜率（当前最佳）
- **Monte Carlo**: ~70% 胜率
- **加权启发式**: ~60% 胜率

---

## 📋 五阶段实施计划

### **阶段1: 数据基础设施**
**目标**: 构建Transformer训练所需的数据管道

#### 1.1 PyTorch数据集类 (`training/dataset.py`)

**功能**:
- 解析JSONL文件中的500个游戏记录
- 从每局游戏的moves中提取 (棋盘状态, 动作, 分数) 元组
- 实现 `__getitem__` 返回归一化的棋盘张量

**棋盘编码方案**:
```python
# Log2归一化（推荐）
# 0 → 0, 2 → 1/11, 4 → 2/11, ..., 2048 → 11/11
encoded_tile = log2(tile) / 11.0 if tile > 0 else 0
```

**数据结构**:
```python
{
    'board': torch.FloatTensor (1, 4, 4),    # 归一化的棋盘状态
    'action': torch.LongTensor,               # 0=上, 1=下, 2=左, 3=右
    'score': torch.FloatTensor,               # 当前分数（可选，用于价值预测）
    'game_id': int,                           # 游戏ID（用于分割）
    'move_number': int                        # 步数（用于阶段分析）
}
```

#### 1.2 数据增强 (`training/augmentation.py`)

**8倍对称变换**:
- 4次旋转: 0°, 90°, 180°, 270°
- 2次反射: 水平翻转, 垂直翻转
- 动作一致性映射

**动作映射表**:
```
原始动作 → 变换后动作
旋转90°顺时针:  上→右, 右→下, 下→左, 左→上
旋转180°:       上→下, 下→上, 左→右, 右→左
旋转270°顺时针: 上→左, 左→下, 下→右, 右→上
水平翻转:       左→右, 右→左, 上→上, 下→下
垂直翻转:       上→下, 下→上, 左→左, 右→右
```

**实现策略**:
```python
def augment_sample(board, action):
    """应用随机旋转/翻转并更新动作标签"""
    # 50%概率应用增强
    if random.random() < 0.5:
        aug_type = random.choice(['rot90', 'rot180', 'rot270', 'fliph', 'flipv'])
        board = apply_transform(board, aug_type)
        action = remap_action(action, aug_type)
    return board, action
```

#### 1.3 数据集划分

**划分策略**:
- 训练集: 350局游戏 (70%) → ~655,000步
- 验证集: 75局游戏 (15%) → ~140,000步
- 测试集: 75局游戏 (15%) → ~140,000步

**关键点**:
- 在**游戏级别**划分（非步级别）防止数据泄露
- 确保同一局游戏的所有步骤在同一集合中
- 保存划分索引以保证可复现性

#### 1.4 DataLoader配置

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,      # GPU加速
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
```

---

### **阶段2: Transformer模型架构**
**目标**: 实现带有2D位置编码的Transformer策略网络

#### 2.1 模型结构 (`models/transformer/transformer_policy.py`)

**完整架构**:
```
输入: (batch, 1, 4, 4) - 单通道棋盘状态

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
预处理层:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. 展平空间维度: (batch, 16, 1)
  2. 方块嵌入 (Tile Embedding):
     Linear(1 → 128)
     输出: (batch, 16, 128)
  
  3. 2D位置编码 (Positional Encoding):
     16个可学习的位置向量（对应4×4网格）
     每个位置(i,j)有唯一的128维嵌入
     与方块嵌入相加
     输出: (batch, 16, 128)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transformer编码器 (4层):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  每层包含:
    - MultiheadAttention(
        embed_dim=128,
        num_heads=8,
        dropout=0.1
      )
    - LayerNorm
    - FeedForward: 
        Linear(128 → 512) → ReLU
        Linear(512 → 128)
    - LayerNorm
    - Residual connections
  
  输出: (batch, 16, 128)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
全局池化:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Mean pooling across sequence dim
  输出: (batch, 128)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
策略头 (Policy Head):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Linear(128 → 256) → ReLU
  Dropout(0.2)
  Linear(256 → 128) → ReLU
  Linear(128 → 4)
  
  输出: (batch, 4) - 4个方向的动作logits
```

**关键参数**:
- 嵌入维度 (embed_dim): 128
- 注意力头数 (num_heads): 8
- 前馈网络维度 (dim_feedforward): 512
- Transformer层数: 4
- Dropout: 0.1 (编码器), 0.2 (策略头)

#### 2.2 2D位置编码实现 (`models/transformer/positional_encoding.py`)

```python
class PositionalEncoding2D(nn.Module):
    """2D可学习位置编码，用于4×4棋盘"""
    def __init__(self, embed_dim=128):
        super().__init__()
        # 16个位置，每个128维
        self.position_embeddings = nn.Parameter(
            torch.randn(16, embed_dim) * 0.02
        )
    
    def forward(self, x):
        # x: (batch, 16, embed_dim)
        return x + self.position_embeddings.unsqueeze(0)
```

**优势**:
- 保留空间结构信息
- 让模型区分不同位置的方块
- 可学习，自适应棋盘特征

#### 2.3 模型工具 (`models/transformer/base_model.py`)

**实用函数**:
- `preprocess_board()`: 棋盘归一化
- `decode_action()`: 动作ID转方向名
- `count_parameters()`: 统计模型参数量
- `get_model_summary()`: 打印模型结构

**预期模型大小**:
- 总参数: ~500K-800K
- 模型文件: ~2-3MB

---

### **阶段3: 训练基础设施**
**目标**: 完整的训练循环与高级调度

#### 3.1 损失函数

**交叉熵损失**:
```python
criterion = nn.CrossEntropyLoss(ignore_index=-1)
# ignore_index用于过滤无效移动
```

**损失计算**:
- 输入: 模型预测的动作logits (batch, 4)
- 目标: 专家动作标签 (batch,)
- 输出: 标量损失值

#### 3.2 优化器配置

**AdamW优化器**:
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0003,              # 初始学习率
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01       # L2正则化
)
```

**梯度裁剪**:
```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

#### 3.3 学习率调度

**两阶段策略**:

1. **预热阶段** (前10个epoch):
   ```python
   warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
       optimizer,
       start_factor=0.01,    # 从0.00003开始
       end_factor=1.0,       # 到0.0003
       total_iters=10
   )
   ```

2. **余弦退火** (10个epoch之后):
   ```python
   cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
       optimizer,
       T_0=20,               # 第一次重启周期
       T_mult=2,             # 每次周期翻倍
       eta_min=1e-6          # 最小学习率
   )
   ```

**学习率曲线示意**:
```
lr
0.0003 |     ╱╲      ╱╲
       |    ╱  ╲    ╱  ╲
       |   ╱    ╲  ╱    ╲
       |  ╱      ╲╱      ╲
0.00003|_╱________________╲___
       0  10  30  50  70  90  epochs
       └──┘  └────────────────
       预热    余弦退火
```

#### 3.4 训练循环特性

**核心流程**:
```python
for epoch in range(200):
    # 训练阶段
    model.train()
    for batch in train_loader:
        # 前向传播
        outputs = model(batch['board'])
        loss = criterion(outputs, batch['action'])
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_loss, val_acc = evaluate(model, val_loader)
    
    # 学习率调度
    scheduler.step()
    
    # 检查点保存
    if val_loss < best_val_loss:
        save_checkpoint(model, 'best_model.pth')
    
    # 早停检查
    if no_improvement_for(25):
        break
```

**监控指标**:
- 训练损失 (train_loss)
- 验证损失 (val_loss)
- 训练准确率 (train_acc)
- 验证准确率 (val_acc)
- Top-2准确率 (专家动作在前2预测中)
- 学习率 (current_lr)
- 梯度范数 (gradient_norm)

#### 3.5 检查点策略

**保存时机**:
1. **最佳模型**: 验证损失改善时立即保存
2. **定期检查点**: 每25个epoch保存一次
3. **最终模型**: 训练结束时保存

**检查点内容**:
```python
{
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_acc': val_acc,
    'hyperparameters': {
        'batch_size': 64,
        'lr': 0.0003,
        'embed_dim': 128,
        ...
    }
}
```

**检查点管理**:
- 保留最近3个检查点
- 始终保留最佳模型
- 文件命名: `transformer_epoch{epoch}_val{val_loss:.4f}.pth`

#### 3.6 早停机制

**配置**:
```python
patience = 25           # 等待25个epoch
min_delta = 0.0001     # 最小改进阈值
```

**逻辑**:
- 监控验证损失
- 25个epoch无改进则停止
- 恢复最佳权重

#### 3.7 TensorBoard日志

**记录内容**:
```python
# 标量
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/train', train_acc, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)
writer.add_scalar('LR', current_lr, epoch)
writer.add_scalar('Gradient_Norm', grad_norm, epoch)

# 直方图
writer.add_histogram('Predictions', predictions, epoch)
writer.add_histogram('Weights/fc1', model.fc1.weight, epoch)

# 混淆矩阵（每10个epoch）
if epoch % 10 == 0:
    writer.add_figure('Confusion_Matrix', cm_figure, epoch)
```

---

### **阶段4: 训练执行**
**目标**: 运行训练并监控进度

#### 4.1 环境准备

**依赖检查**:
```bash
# 检查PyTorch安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# 安装额外依赖
pip install tensorboard tqdm scikit-learn
```

**目录结构**:
```bash
mkdir -p checkpoints/transformer
mkdir -p results/training_curves/transformer
mkdir -p results/game_simulations
```

#### 4.2 训练命令

**基础训练**:
```bash
cd /Users/wyp/develop/play2048

python training/train_transformer.py \
  --data data/training_games.jsonl \
  --epochs 200 \
  --batch-size 64 \
  --lr 0.0003 \
  --embed-dim 128 \
  --num-heads 8 \
  --num-layers 4 \
  --dropout 0.1 \
  --weight-decay 0.01 \
  --warmup-epochs 10 \
  --early-stop-patience 25 \
  --checkpoint-dir checkpoints/transformer \
  --log-dir results/training_curves/transformer \
  --device auto \
  --seed 42
```

**高级选项**:
```bash
# 从检查点恢复
python training/train_transformer.py \
  --resume checkpoints/transformer/transformer_epoch75.pth \
  ...

# 调试模式（小批次）
python training/train_transformer.py \
  --debug \
  --epochs 5 \
  --batch-size 8 \
  ...
```

#### 4.3 训练监控

**TensorBoard启动**:
```bash
tensorboard --logdir results/training_curves/transformer --port 6006
```
访问: http://localhost:6006

**实时日志**:
```
Epoch [1/200] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Train Loss: 1.2456 | Train Acc: 42.3% | Val Loss: 1.1890 | Val Acc: 45.8%
LR: 0.000030 | Time: 12m 34s | ETA: 41h 23m

Epoch [10/200] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Train Loss: 0.8234 | Train Acc: 65.7% | Val Loss: 0.7891 | Val Acc: 67.2%
LR: 0.000300 | Time: 12m 28s | ETA: 39h 27m
✓ Best model saved! (val_loss improved: 1.1890 → 0.7891)

Epoch [50/200] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Train Loss: 0.4567 | Train Acc: 82.1% | Val Loss: 0.5123 | Val Acc: 79.5%
LR: 0.000156 | Time: 12m 31s | ETA: 31h 18m
✓ Checkpoint saved: transformer_epoch50.pth
```

**关键观察点**:
- 验证损失持续下降 → 模型学习中
- 训练/验证损失差距 > 0.2 → 可能过拟合
- 准确率稳定增长 → 训练健康
- 学习率平滑变化 → 调度器正常

#### 4.4 预期训练时间

**硬件性能估计**:

| 硬件配置 | 单epoch时间 | 200epoch总时长 | 备注 |
|---------|-----------|---------------|------|
| Apple M1/M2 (MPS) | 4-6分钟 | 13-20小时 | 推荐 |
| NVIDIA RTX 3090 | 3-4分钟 | 10-13小时 | 最佳 |
| NVIDIA RTX 2060 | 6-8分钟 | 20-27小时 | 可用 |
| CPU (8核) | 45-60分钟 | 6-8天 | 不推荐 |

**优化建议**:
- 使用 `--num-workers 4` 加速数据加载
- 启用 `--pin-memory` (GPU训练)
- 使用混合精度训练 `--fp16` (如支持)

#### 4.5 故障排除

**常见问题**:

1. **GPU内存不足**:
   ```bash
   # 减小batch size
   --batch-size 32
   
   # 或减少worker数量
   --num-workers 2
   ```

2. **训练不稳定**:
   ```bash
   # 降低学习率
   --lr 0.0001
   
   # 增加warmup
   --warmup-epochs 20
   ```

3. **验证准确率不增长**:
   - 检查数据增强是否正确
   - 验证集不应用增强
   - 检查动作映射逻辑

---

### **阶段5: 评估与分析**
**目标**: 验证模型性能并与基准对比

#### 5.1 离线评估 (`evaluation/offline_eval.py`)

**测试集指标**:

1. **Top-1准确率**:
   ```python
   # 模型最优预测与专家动作完全匹配的比例
   accuracy = (predictions == expert_actions).mean()
   ```
   **目标**: ≥ 80%

2. **Top-2准确率**:
   ```python
   # 专家动作在模型前2预测中的比例
   top2_acc = (expert_actions in top2_predictions).mean()
   ```
   **目标**: ≥ 90%

3. **混淆矩阵**:
   ```
              预测
           上  下  左  右
   真  上 [80  5  10  5]
   实  下 [6  78  8   8]
   值  左 [12  7  75  6]
       右 [7   9  6  78]
   ```
   分析: 哪些方向容易混淆

4. **分阶段准确率**:
   - 早期游戏 (步数 < 100): ? %
   - 中期游戏 (100-500步): ? %
   - 后期游戏 (步数 > 500): ? %
   
   预期: 后期准确率更高（策略更明确）

5. **按棋盘状态分析**:
   - 高分局 (>30000): ? %
   - 中分局 (10000-30000): ? %
   - 低分局 (<10000): ? %

**运行命令**:
```bash
python evaluation/offline_eval.py \
  --model checkpoints/transformer/best_model.pth \
  --data data/training_games.jsonl \
  --split test \
  --output results/offline_metrics.json
```

#### 5.2 在线评估 (`evaluation/game_simulator.py`)

**游戏模拟**:

运行100局完整游戏，收集:

1. **胜率** (Win Rate):
   ```
   达到2048方块的游戏比例
   目标: ≥ 70% (接近Expectimax的80%)
   ```

2. **平均分数**:
   ```
   100局的平均最终分数
   目标: ≥ 15,000
   ```

3. **最大方块分布**:
   ```
   1024: 15局 (15%)
   2048: 70局 (70%)
   4096: 12局 (12%)
   8192: 3局 (3%)
   ```

4. **平均步数**:
   ```
   每局平均移动次数
   预期: 800-1200步
   ```

5. **生存率**:
   ```
   超过500步的游戏比例
   目标: ≥ 80%
   ```

**运行命令**:
```bash
python evaluation/game_simulator.py \
  --model checkpoints/transformer/best_model.pth \
  --num-games 100 \
  --output results/game_simulations/transformer_results.json \
  --visualize \
  --seed 42
```

**输出示例**:
```json
{
  "model": "transformer",
  "num_games": 100,
  "win_rate": 0.73,
  "avg_score": 16234.5,
  "max_tile_distribution": {
    "512": 2,
    "1024": 12,
    "2048": 73,
    "4096": 11,
    "8192": 2
  },
  "avg_moves": 1043.2,
  "survival_rate": 0.85,
  "games": [...]
}
```

#### 5.3 对比分析

**性能对比表**:

| 指标 | Transformer | Expectimax | Monte Carlo | 启发式 |
|-----|------------|-----------|-------------|--------|
| **离线指标** |
| 测试准确率 | ? % | N/A | N/A | N/A |
| Top-2准确率 | ? % | N/A | N/A | N/A |
| **在线指标** |
| 胜率 | ? % | 80% | 70% | 60% |
| 平均分数 | ? | ~17000 | ~14000 | ~11000 |
| 达到4096 | ? % | 15% | 8% | 3% |
| 平均步数 | ? | ~1100 | ~950 | ~750 |
| **性能指标** |
| 推理时间 | ? ms | 5-10ms | 50-100ms | <1ms |
| 模型大小 | ~3MB | N/A | N/A | N/A |

#### 5.4 可视化分析

**训练曲线**:
```python
# 绘制损失曲线
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('results/loss_curve.png')
```

**准确率曲线**:
```python
plt.plot(epochs, train_accs, label='Train Accuracy')
plt.plot(epochs, val_accs, label='Val Accuracy')
plt.axhline(y=80, color='r', linestyle='--', label='Target (80%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig('results/accuracy_curve.png')
```

**注意力可视化** (高级):
```python
# 提取某个棋盘的注意力权重
attention_weights = model.get_attention(board_tensor)
# 可视化16×16注意力矩阵
plt.imshow(attention_weights, cmap='viridis')
plt.title('Self-Attention Heatmap')
plt.savefig('results/attention_heatmap.png')
```

#### 5.5 失败案例分析

**收集低性能游戏**:
- 分数 < 5000 的游戏
- 未达到1024的游戏
- 早期失败 (步数 < 100)

**分析方向**:
1. 是否有共同的棋盘模式？
2. 哪些动作被误判？
3. 是否缺乏某类训练数据？

**改进策略**:
- 针对性数据增强
- 调整损失权重
- 添加困难样本

---

## 🗂️ 文件清单

### 创建的文件

```
models/transformer/
├── __init__.py                      # 模块导出
├── transformer_policy.py            # 主Transformer模型
├── positional_encoding.py           # 2D位置编码
└── base_model.py                    # 工具函数

training/
├── __init__.py                      # 模块导出
├── dataset.py                       # Game2048Dataset类
├── augmentation.py                  # 数据增强
├── train_transformer.py             # 训练脚本
└── utils.py                         # 训练工具函数

evaluation/
├── __init__.py                      # 模块导出
├── offline_eval.py                  # 测试集评估
├── game_simulator.py                # 游戏模拟
└── visualize_results.py             # 结果可视化

checkpoints/transformer/             # 训练检查点（自动生成）
├── best_model.pth
├── transformer_epoch25.pth
├── transformer_epoch50.pth
└── ...

results/
├── training_curves/transformer/     # TensorBoard日志
├── game_simulations/                # 游戏结果
│   └── transformer_results.json
├── offline_metrics.json             # 测试集指标
├── loss_curve.png                   # 训练曲线图
└── accuracy_curve.png               # 准确率图
```

---

## ⚙️ 超参数配置表

| 参数类别 | 参数名 | 值 | 说明 |
|---------|-------|---|------|
| **模型架构** |
| | embed_dim | 128 | 嵌入维度 |
| | num_heads | 8 | 注意力头数 |
| | num_layers | 4 | Transformer层数 |
| | dim_feedforward | 512 | 前馈网络维度 |
| | dropout | 0.1 | Encoder dropout |
| | head_dropout | 0.2 | 策略头dropout |
| **训练配置** |
| | batch_size | 64 | 批次大小 |
| | epochs | 200 | 最大epoch数 |
| | learning_rate | 0.0003 | 初始学习率 |
| | weight_decay | 0.01 | L2正则化 |
| | gradient_clip | 1.0 | 梯度裁剪 |
| **调度器** |
| | warmup_epochs | 10 | 预热epoch数 |
| | scheduler_T0 | 20 | 余弦周期 |
| | eta_min | 1e-6 | 最小学习率 |
| **早停** |
| | patience | 25 | 等待epoch数 |
| | min_delta | 0.0001 | 最小改进 |
| **数据** |
| | train_ratio | 0.70 | 训练集比例 |
| | val_ratio | 0.15 | 验证集比例 |
| | test_ratio | 0.15 | 测试集比例 |
| | augmentation_prob | 0.5 | 增强概率 |
| | num_workers | 4 | 数据加载进程 |

---

## 📊 预期结果

### 训练过程

**Epoch 1-10 (预热阶段)**:
- 损失: 1.3 → 0.8
- 准确率: 40% → 65%
- 学习率: 0.00003 → 0.0003

**Epoch 10-50 (快速学习)**:
- 损失: 0.8 → 0.5
- 准确率: 65% → 80%
- 学习率: 0.0003 → 0.00015

**Epoch 50-100 (精细调整)**:
- 损失: 0.5 → 0.42
- 准确率: 80% → 83%
- 学习率: 0.00015 → 0.00008

**Epoch 100+ (收敛)**:
- 损失: 0.42 → 0.38
- 准确率: 83% → 85%
- 可能提前停止

### 最终性能预测

**保守估计** (基于500局数据):
- 测试准确率: 75-80%
- 游戏胜率: 60-70%
- 平均分数: 13,000-16,000

**理想情况** (模型充分学习):
- 测试准确率: 80-85%
- 游戏胜率: 70-80%
- 平均分数: 16,000-20,000

**突破性能** (接近/超越Expectimax):
- 测试准确率: 85%+
- 游戏胜率: 80%+
- 平均分数: 20,000+

---

## 🚀 快速开始

### 1. 验证数据
```bash
python -c "
import json
count = 0
with open('data/training_games.jsonl') as f:
    for line in f:
        game = json.loads(line)
        count += len(game['moves'])
print(f'总游戏数: {500}')
print(f'总步数: {count}')
print(f'增强后: {count * 8}')
"
```

### 2. 实现数据集
```bash
# 先实现并测试dataset.py
python training/dataset.py --test
```

### 3. 实现模型
```bash
# 测试模型前向传播
python models/transformer/transformer_policy.py --test
```

### 4. 开始训练
```bash
# 先小批次测试
python training/train_transformer.py --debug

# 确认无误后全量训练
python training/train_transformer.py
```

### 5. 监控进度
```bash
# 终端1: 运行训练
python training/train_transformer.py

# 终端2: 启动TensorBoard
tensorboard --logdir results/training_curves
```

---

## ⏱️ 时间规划

### 开发阶段
- **第1-2天**: 数据基础设施（dataset, augmentation）
- **第3-4天**: Transformer模型实现
- **第5-6天**: 训练循环与工具
- **第7天**: 集成测试与调试

### 训练阶段
- **第8天**: 启动训练（后台运行10-20小时）
- **第9天**: 监控训练，中期评估

### 评估阶段
- **第10天**: 离线评估，在线模拟
- **第11天**: 结果分析，可视化

**总计**: ~11天（包含训练等待时间）

---

## 🎯 成功标准

### 最低要求 ✓
- [ ] 测试准确率 ≥ 70%
- [ ] 胜率 ≥ 50%
- [ ] 平均分数 ≥ 10,000
- [ ] 推理时间 < 50ms

### 目标性能 ⭐
- [ ] 测试准确率 ≥ 80%
- [ ] 胜率 ≥ 70%
- [ ] 平均分数 ≥ 15,000
- [ ] 推理时间 < 20ms

### 挑战目标 🏆
- [ ] 测试准确率 ≥ 85%
- [ ] 胜率 ≥ 80% (匹敌Expectimax)
- [ ] 达到4096: ≥ 10%
- [ ] 平均分数 ≥ 20,000

---

## 🔧 调试技巧

### 数据问题
```python
# 检查数据分布
python -c "
from training.dataset import Game2048Dataset
dataset = Game2048Dataset('data/training_games.jsonl')
print(f'样本数: {len(dataset)}')
print(f'动作分布: {dataset.get_action_distribution()}')
"
```

### 模型问题
```python
# 检查模型输出
python -c "
import torch
from models.transformer import TransformerPolicy
model = TransformerPolicy()
dummy_input = torch.randn(2, 1, 4, 4)
output = model(dummy_input)
print(f'输出形状: {output.shape}')  # 应为 (2, 4)
print(f'参数量: {sum(p.numel() for p in model.parameters())}')
"
```

### 训练问题
```bash
# 单步调试
python -c "
from training.train_transformer import train_one_epoch
# ... 运行单个epoch查看梯度
"
```

---

## 📚 参考资料

### 论文
- **Attention Is All You Need** (Vaswani et al., 2017) - Transformer原理
- **Playing Atari with Deep RL** (Mnih et al., 2013) - DQN基础
- **Mastering Chess and Shogi by Self-Play** (Silver et al., 2017) - AlphaZero

### 代码参考
- PyTorch Transformer官方教程
- Hugging Face Transformers库
- MinGPT (Karpathy) - 简化Transformer实现

---

## 📝 下一步行动

1. ✅ **已完成**: 生成500局训练数据
2. ⏳ **进行中**: 创建训练计划文档
3. 🔜 **下一步**: 实现 `training/dataset.py`

**准备开始编码时请告知！** 🚀
