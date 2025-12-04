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
    --num_envs 2048 \
    --max_iterations 1000
```

### 2. FOCPO (First-Order CPO)
```bash
cd /home/lz/Project/IsaacLab/Isaaclab_SafeRL/FOCPO
python scripts/train.py \
    --task Isaac-Velocity-Galileo-FOCPO-v0 \
    --num_envs 2048 \
    --max_iterations 1000
```

### 3. PCPO (Projection-Based CPO)
```bash
cd /home/lz/Project/IsaacLab/Isaaclab_SafeRL/PCPO
python scripts/train.py \
    --task Isaac-Velocity-Galileo-PCPO-v0 \
    --num_envs 2048 \
    --max_iterations 1000
```

### 4. PPOLag (PPO-Lagrange)
```bash
cd /home/lz/Project/IsaacLab/Isaaclab_SafeRL/PPOLag
python scripts/train.py \
    --task Isaac-Velocity-Galileo-PPOLag-v0 \
    --num_envs 2048 \
    --max_iterations 1000
```

## 测试训练好的模型

### 播放模型
```bash
# 示例：播放 CPO 模型
cd /home/lz/Project/IsaacLab/Isaaclab_SafeRL/CPO
python scripts/play.py \
    --task Isaac-Velocity-Galileo-CPO-v0 \
    --model_path logs/CPO/Isaac-Velocity-Galileo-CPO-v0/20250101_120000/model_1000.pt \
    --num_envs 4
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
# 启动 TensorBoard
tensorboard --logdir Isaaclab_SafeRL/[ALGO]/logs/[ALGO]/Isaac-Velocity-Galileo-[ALGO]-v0/

# 例如：查看 CPO 训练进度
tensorboard --logdir Isaaclab_SafeRL/CPO/logs/CPO/Isaac-Velocity-Galileo-CPO-v0/
```

## 故障排除

### 问题：找不到 galileo_parkour 模块
**解决方案**:
```bash
# 确保在 IsaacLab 环境中
cd /home/lz/Project/IsaacLab
# 重新设置环境
source isaaclab.sh
```

### 问题：找不到任务 ID
**解决方案**: 确保使用正确的任务 ID：
- CPO: `Isaac-Velocity-Galileo-CPO-v0`
- FOCPO: `Isaac-Velocity-Galileo-FOCPO-v0`
- PCPO: `Isaac-Velocity-Galileo-PCPO-v0`
- PPOLag: `Isaac-Velocity-Galileo-PPOLag-v0`

### 问题：GPU 内存不足
**解决方案**: 减少并行环境数量
```bash
python scripts/train.py --task Isaac-Velocity-Galileo-CPO-v0 --num_envs 1024
```

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

