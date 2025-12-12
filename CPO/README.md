# Constrained Policy Optimization (CPO)

Independent implementation of CPO for Isaac Lab.

## Structure
- `algorithm.py`: CPO algorithm implementation.
- `modules.py`: Actor-Critic network.
- `rewards.py`: Reward functions.
- `storage.py`: Rollout storage.
- `config/`: Environment and training configurations.

```bash
cd /home/lz/Project/IsaacLab/Isaaclab_SafeRL/CPO
conda activate isaaclab
```

## Usage
Train:
```bash
cd /home/lz/Project/IsaacLab/Isaaclab_SafeRL/CPO
python scripts/train.py \
    --task Isaac-Velocity-Galileo-CPO-v0 \
    --num_envs 10000 \
    --max_iterations 50000 \
    --headless \
    --log_interval 1
```
*Check config/galileo_cpo_cfg.py for task ID.*

Play:
```bash
# 示例：播放 CPO 模型
cd /home/lz/Project/IsaacLab/Isaaclab_SafeRL/CPO
python scripts/play.py \
    --task Isaac-Velocity-Galileo-CPO-v0 \
    --num_envs 50 \
    --model_path logs/CPO/
```

### 高程图传感器开关
- 在 `config/galileo/galileo_env_cfg.py` 中设置 `use_height_map=True` 启用高程图输入；`height_map_sensor_name` 默认为 `height_scanner`，`height_map_obs_key` 为观测字典中的键名。
- 启用后，`constraint_com_height` 与 `reward_base_height` 使用“相对地面高度”（基于高程图/地形高度均值）；未开启时保持绝对高度逻辑。
- 约束/奖励参数可在同一文件中调整：`height_map_target_height`（期望相对高度）、`height_map_min_height`/`height_map_max_height`（相对高度上下界）。

Tensorboard

```bash
tensorboard --logdir ./logs/CPO/Isaac-Velocity-Galileo-CPO-v0/
```

### TensorBoard 指标说明与评估指南

**常见标量含义**
- `Perf/Reward`、`Perf/Episode_Reward`：步均奖励与单回合总回报，反映任务完成度。
- `Perf/Episode_Length`：平均存活步数；与回报结合判断是“靠存活”还是“完成目标”。
- `Constraints/Cost_vs_Limit (Estimated_Cost, Limit)`：约束成本与阈值对比；持续高于 Limit 表示未满足安全约束。
- `Constraints/Cost_Per_Step`、`Violations_Per_Episode`：成本/违规的密度与单回合累计，区分“偶发大罚”还是“持续小罚”。
- `Costs/ByTerm/*`：按约束项拆分（如 collision、torque 等），定位主要违规来源，优先调整该项权重或阈值。
- `Episode/Real_Done_Rate`、`Episode/Timeout_Rate`：终止原因占比；Real_Done 高多因失败/碰撞，Timeout 高说明策略更保守。
- `Policy/KL`：更新间 KL；过高可能策略崩溃，过低可能学习停滞，可调 learning rate 或信赖域步长。
- `Loss/Policy`、`Loss/Value`、`Loss/Cost_Value`：策略、价值、成本价值损失；持续增大或振荡意味着估计器不稳。
- `Entropy`：策略熵；过低可能过早收敛，适当保持探索可防止陷入局部最优。
- `LR`、`Grad_Norm`：学习率与梯度范数；剧烈尖峰提示需检查数值稳定性或裁剪设置。
- `Meta/Update_Skipped_No_Complete_Ep`：当迭代没有完整 episode 时置 1；频繁出现需增大 `steps_per_env` 或调整终止逻辑。

**优先关注什么**
- 性能：`Perf/Reward` 与 `Perf/Episode_Reward` 的上升趋势与平滑度。
- 安全：`Constraints/Cost_vs_Limit` 是否长期低于或贴近 0；`Costs/ByTerm/*` 是否有主导项。
- 稳定：`Policy/KL`、`Loss/*` 是否无尖峰；`Grad_Norm` 是否受控；`Entropy` 是否过早塌陷。
- 行为：`Episode_Length` 与终止占比，判断是“成功完成”还是“躲避但不干活”。

**如何用指标评估训练效果**
1) 性能-安全双轴：优先看回报曲线是否上升，同时成本曲线是否低于 Limit。若回报升但成本超限，需收紧步长/约束权重；反之成本低但回报平，适当放宽探索或增大学习率。  
2) 稳定性：监控 `Policy/KL` 与 `Loss/*`。出现尖峰时，可降低学习率、启用梯度裁剪或缩小信赖域步长。  
3) 行为合理性：`Episode_Length`、`Real_Done_Rate/Timeout_Rate` 能揭示策略是“冲但易撞”还是“苟但不做事”；结合 `Costs/ByTerm/*` 定位主要风险源。  
4) 收敛判定：回报进入平台期且成本稳定在阈值下、KL 波动小、Loss 平稳，即可认为接近收敛；若同时熵很低，可尝试小幅增大熵系数验证是否仍有提升空间。

杀死所有进程

```bash
pkill -f tensorboard
```


