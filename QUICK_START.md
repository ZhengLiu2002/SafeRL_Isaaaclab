# Galileo SafeRL 快速开始指南

## 前置条件
确保已安装 IsaacLab 和 galileo_parkour 扩展。

## 安装 galileo_parkour 扩展
```bash
cd /home/lz/Project/IsaacLab
# 扩展应该已经存在于 source/extensions/galileo_parkour/
```

## 训练示例

### 1. CPO (Constrained Policy Optimization)
```bash
cd /home/lz/Project/IsaacLab/Isaaclab_SafeRL/CPO
python scripts/train.py \
    --task Isaac-Velocity-Galileo-CPO-v0 \
    --num_envs 8192 \
    --max_iterations 10000 \
    --headless \
    --log_interval 10
```

### 2. FOCPO (First-Order CPO)
```bash
cd /home/lz/Project/IsaacLab/Isaaclab_SafeRL/FOCPO
python scripts/train.py \
    --task Isaac-Velocity-Galileo-FOCPO-v0 \
    --num_envs 2048 \
    --max_iterations 1000 \
    --headless \
    --log_interval 1
```

### 3. PCPO (Projection-Based CPO)
```bash
cd /home/lz/Project/IsaacLab/Isaaclab_SafeRL/PCPO
python scripts/train.py \
    --task Isaac-Velocity-Galileo-PCPO-v0 \
    --num_envs 2048 \
    --max_iterations 1000 \
    --headless
```

### 4. PPOLag (PPO-Lagrange)
```bash
cd /home/lz/Project/IsaacLab/Isaaclab_SafeRL/PPOLag
python scripts/train.py \
    --task Isaac-Velocity-Galileo-PPOLag-v0 \
    --num_envs 2048 \
    --max_iterations 1000 \
    --headless \
    --log_interval 1
```

## 测试训练好的模型

### 播放模型
```bash
# 示例：播放 CPO 模型
cd /home/lz/Project/IsaacLab/Isaaclab_SafeRL/CPO
python scripts/play.py \
    --task Isaac-Velocity-Galileo-CPO-v0 \
    --num_envs 50 \
    --model_path logs/CPO/
```

## 配置文件位置

### 环境配置
- **CPO**: `CPO/config/galileo_env_cfg.py`
- **FOCPO**: `FOCPO/config/galileo_env_cfg.py`
- **PCPO**: `PCPO/config/galileo_env_cfg.py`
- **PPOLag**: `PPOLag/config/galileo_env_cfg.py`

### 算法配置
- **CPO**: `CPO/config/galileo_cpo_cfg.py`
- **FOCPO**: `FOCPO/config/galileo_focpo_cfg.py`
- **PCPO**: `PCPO/config/galileo_pcpo_cfg.py`
- **PPOLag**: `PPOLag/config/galileo_ppolag_cfg.py`

## 常用参数

### 训练参数
- `--num_envs`: 并行环境数量 (默认: 2048)
- `--max_iterations`: 最大训练迭代次数 (默认: 1000)
- `--seed`: 随机种子 (默认: 0)
- `--steps_per_env`: 每个环境的步数 (默认: 16)
- `--batch_size`: 批次大小 (默认: 32768)
- `--log_interval`: 日志记录间隔 (默认: 10)
- `--save_interval`: 模型保存间隔 (默认: 100)

### 测试参数
- `--num_envs`: 环境数量 (默认: 1)
- `--num_steps`: 最大步数 (默认: 1000)
- `--stochastic`: 使用随机策略而非确定性策略

## 日志和模型保存位置
```
Isaaclab_SafeRL/
├── CPO/
│   └── logs/
│       └── CPO/
│           └── Isaac-Velocity-Galileo-CPO-v0/
│               └── [timestamp]/
│                   ├── model_0.pt
│                   ├── model_100.pt
│                   └── tb/  # TensorBoard 日志
├── FOCPO/
│   └── logs/...
├── PCPO/
│   └── logs/...
└── PPOLag/
    └── logs/...
```

## 查看训练进度
```bash
# 启动 TensorBoard（建议在根目录下运行以查看所有算法的日志）
# 确保在 Isaaclab_SafeRL 目录下运行，或者调整路径
tensorboard --logdir . --port 6008

# 查看特定算法的日志
tensorboard --logdir CPO/logs --port 6008
```

## TensorBoard 内容说明
训练日志包含以下主要部分：
- **Episode/**: 回合奖励、长度、成本
- **Train/**: 训练过程中的平均奖励、成本、KL散度等
- **Loss/**: 各种损失函数值
- **RewardTerms/**: 各项奖励的具体数值（如跟踪速度、惩罚项等）
- **CostTerms/**: 各项安全约束的具体数值（如速度限制、碰撞等）
- **Perf/**: 训练性能指标（FPS等）

## 自定义配置

### 修改奖励权重
编辑相应的 `config/galileo_env_cfg.py` 文件中的 `RewardsCfg` 类。

### 修改算法参数
编辑相应的 `config/galileo_[algo]_cfg.py` 文件中的 `ALGO_CFG` 字典。

### 修改机器人参数
编辑 `source/extensions/galileo_parkour/galileo_parkour/assets/galileo.py` 文件。

## 性能建议

1. **GPU 利用率**: 使用 `nvidia-smi` 监控 GPU 使用情况
2. **环境数量**: 根据 GPU 内存调整 `--num_envs`
3. **批次大小**: 通常设置为 `num_envs * steps_per_env`
4. **存储设备**: 如果 GPU 内存紧张，使用 `--storage_device cpu`

## 更多信息
详细的迁移信息请参考 `MIGRATION_SUMMARY.md`。




