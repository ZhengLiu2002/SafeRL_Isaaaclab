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

### TensorBoard 指标说明与评估指南（通俗版）

训练好不好，最直观就看 TensorBoard 的曲线是否“像一个在进步的孩子”：  
**回报在涨、违规在降、更新在发生、损失在稳。**  
下面按类别把每条曲线都解释清楚，你可以照着逐项体检。

#### 1) 任务性能（机器人“干活”干得怎么样）

- `Perf/Reward`（每一步平均奖励）  
  含义：这次迭代 rollout 里**每步的平均 reward**。  
  怎么看：  
  - 早期应该从负或很小的值**逐步上升**。  
  - 上升到某个水平后进入平台期是正常的（说明接近收敛）。  
  - 如果长期在 0 附近抖动，常见原因是**策略没更新、指令太小或奖励饱和**。  
  - 如果突然掉到很低并长时间不恢复，多半是策略崩了（更新太猛/数值不稳）。

- `Perf/Episode_Reward`（每回合总回报）  
  含义：所有环境里**完整 episode 的总 reward 的平均值**。  
  怎么看：  
  - 比 `Perf/Reward` 更贴近“你真正在乎的成绩”。  
  - 关注趋势即可，不必苛求平滑：episode 结束是稀疏事件，本来就噪。  
  - 如果 `Perf/Reward` 在升但 `Episode_Reward` 不升，说明可能是**“小步小奖”涨了，但大目标没完成**（比如只会站稳不会走）。

- `Perf/Episode_Length` / `Episode/Length`（回合长度）  
  含义：平均一回合走了多少步/多长时间。  
  怎么看：  
  - 早期若长度很短（频繁摔/碰撞），正常。能逐步变长说明更稳。  
  - **长度变长但回报不涨** → 可能只学会“苟活”，没学会跟踪/前进。  
  - **长度变短同时回报掉** → 策略退化或扰动太强。

- `Debug/Cmd_LinVel_XY`、`Debug/Cmd_AngVel_Z`（指令速度幅值）  
  含义：课程/命令采样出来的期望线速度、角速度大小。  
  怎么看：  
  - 如果这两个长期接近 0，等于让机器人“没作业”，回报很难涨。  
  - 指令应随课程逐步上升到目标范围。若不上升，优先查 command curriculum。

- `Debug/Act_LinVel_XY`、`Debug/Act_AngVel_Z`（实际速度幅值）  
  含义：机器人真实执行出来的速度大小。  
  怎么看：  
  - 应该逐步**贴近 Cmd 曲线**。  
  - 如果 Act 曲线总是明显低于 Cmd，说明“推不动/不敢动”，可能是奖励惩罚太重或动作尺度太小。  

- `Debug/LinVel_XY_Error`、`Debug/AngVel_Z_Error`（跟踪误差）  
  含义：期望速度和实际速度的差距。  
  怎么看：  
  - **越低越好**；训练有效时应持续下降。  
  - 如果误差不降但回报在涨，说明奖励里有其它项在主导（比如存活、对称性），需要调平衡。

- `Rewards/Components/*`（各奖励子项）  
  含义：每个 reward term 的平均值（已乘权重）。  
  怎么看：  
  - 为了减少噪声：**当前权重为 0 的 reward term 会被自动省略**（不会在控制台打印、也不会写入该 run 的 TensorBoard）。  
  - 先看与你目标最相关的（`track_lin_vel_xy`、`track_ang_vel_z`）。它们应该上升或保持较高。  
  - 惩罚项（如 `collision`、`torques`、`action_rate`）通常是**负值**，训练有效时它们的绝对值会变小（更少罚）。  
  - 如果某个惩罚项负得越来越大，说明策略在“用坏方式换回报”，需要提高该惩罚权重或收紧阈值。

- `Diagnostics/*`（诊断量，如 `error_vel_xy`）  
  含义：不直接计入奖励，但反映状态好坏。  
  怎么看：  
  - 与 Error 曲线一致：越低越好；反弹常意味着策略退化。

#### 2) 安全/约束（有没有违规、违得多不多）

- `Constraints/Cost_vs_Limit/Estimated_Cost` vs `Limit`  
  含义：估计的**每回合成本**和安全阈值。  
  怎么看：  
  - `Estimated_Cost` 应该**低于或贴近 Limit**。  
  - 若长期高于 Limit，说明安全没学到（或 Limit 太紧）。  
  - 若远低于 Limit 且回报不涨，说明策略过于保守（可以放宽课程/探索）。

- `Constraints/Cost_Per_Step`（每步成本）  
  含义：成本密度。  
  怎么看：  
  - 能区分“偶发大违规”还是“持续小违规”。  
  - `Cost_Per_Step` 很低但 `Estimated_Cost` 高，多半是回合太长（走得久累积多）。  

- `Constraints/Violations_Per_Episode`（每回合累计违规）  
  含义：本轮完整 episode 中的平均累计成本。  
  怎么看：  
  - 和 `Estimated_Cost` 应一致；若差很大，可能是 episode 太少导致估计噪声。

- `Costs/ByTerm/*`（按约束项拆分）  
  含义：每个 cost term 的平均违规强度。  
  怎么看：  
  - 找最大的那一项，就是主要安全风险源。  
  - 例如 `joint_torque_limit` 高 → 动作过猛；`collision` 高 → 步态不稳/落脚粗暴。

- `Constraints/Lagrange_Multiplier_Nu`（CPO 的 ν）  
  含义：约束违反时的“惩罚系数/拉格朗日乘子”。  
  怎么看：  
  - ν 上升代表算法在更用力压成本；下降代表约束轻松满足。  
  - ν 一直为 0 也可能是成本过低（约束没起作用）。

#### 3) 学习动态（策略有没有在更新、更新稳不稳）

- `Policy/KL`（接受更新后的 KL）  
  含义：本次策略更新前后分布差距。  
  怎么看：  
  - 理想情况：在 `target_kl` 附近小幅波动（既学得动又不发疯）。  
  - **长期接近 0** → 更新几乎被拒绝/学习停滞。  
  - **频繁尖峰** → 更新过大，容易崩溃。

- `Constraints/LineSearch_Step`（线搜索接受的步长比例）  
  含义：1 表示全步接受，0 表示完全拒绝。  
  怎么看：  
  - 正常训练时大多数迭代应该 >0（哪怕 0.1 也算在学）。  
  - 长期为 0 就是“原地踏步”，优先查 KL/成本判据和数值稳定。

- `Policy/Update_Accepted`（是否接受更新）  
  含义：`LineSearch_Step>0` 记为 1，否则 0。  
  怎么看：  
  - 一眼判断“每次迭代是不是在学”。  

- `Loss/Policy_Surr_Reward`（奖励 surrogate）  
  含义：CPO/TRPO 要最大化的奖励代理目标（期望优势）。  
  怎么看：  
  - 训练有效时，它通常逐步上升或保持正值。  
  - 长期为负且不改善，说明策略更新方向被噪声/惩罚主导。  

- `Loss/Policy_Surr_Cost`（成本 surrogate）  
  含义：对应约束方向的代理目标。  
  怎么看：  
  - 若成本一直远低于 Limit，这个量应接近 0；  
  - 若违规多，它会变大，算法会用它来调整 ν。

- `Policy/Action_Std`、`Policy/Action_Std_Min`、`Policy/Action_Std_Max`（动作噪声尺度）  
  含义：策略分布的标准差，决定探索强弱。  
  怎么看：  
  - 早期较大正常；随训练应逐步下降到稳定水平。  
  - **std 很大且回报不上升** → 乱探索；可减小初始 std。  
  - **std 很小且回报很早平台** → 过早收敛；适当增加探索或课程难度。

- `Policy/Entropy`（策略熵）  
  含义：分布的不确定性，越大越随机。  
  怎么看：  
  - 和 std 类似，但更直观。  
  - 熵骤降到极低 → 策略“死板”，可能卡局部最优。

- `Meta/Update_Skipped_No_Complete_Ep`  
  含义：本轮 rollout 没收集到完整 episode 时为 1。  
  怎么看：  
  - 偶尔为 1 没事；频繁为 1 会让 CPO 的回合成本估计很不准。  
  - 解决：增大 `steps_per_env` 或让 episode 更容易结束。

#### 4) 价值网络（critic 学得准不准）

- `Loss/Value`（奖励价值损失）  
  含义：critic 预测回报的 MSE。  
  怎么看：  
  - 初期较大正常，随后应下降并稳定。  
  - **持续上升/剧烈震荡** → critic 跟不上策略，可能 lr 太大或回报尺度变了。

- `Loss/Cost_Value`（成本价值损失）  
  含义：成本 critic 的 MSE。  
  怎么看：  
  - 规律同上；若成本稀疏/噪声大，会比 `Loss/Value` 更抖。

#### 5) 课程/环境难度（是不是在“升学”）

- `Curriculum/Terrain_Level`  
  含义：当前最大地形等级/难度。  
  怎么看：  
  - 难度应随训练慢慢上升；回报掉一点是正常的“升学阵痛”。  
  - 难度升但回报完全崩 → 升太快，需要放缓课程。

- `Curriculum/Cost_Limit`  
  含义：课程对成本上限的倍率调整后的实际 Limit。  
  怎么看：  
  - 你会看到 Limit 随 stage 变化，这是预期行为。

#### 6) Debug 健康度（有没有数值异常）

- `Debug/Action_Abs_Mean`、`Debug/Action_Abs_Max`  
  含义：动作绝对值均值/最大值。  
  怎么看：  
  - 过大说明动作饱和、可能导致碰撞/抖动。  
  - 长期接近 0 说明策略“不会动”或动作尺度太小。

- `Debug/Obs_Abs_Mean`、`Debug/Obs_Abs_Max`  
  含义：观测幅值健康检查。  
  怎么看：  
  - 出现突然巨大尖峰 → 观测里可能有异常值（NaN/Inf 或尺度错），会直接毁训练。

- `Perf/FPS`  
  含义：训练速度指标，不代表学得好坏，但可用来判断是否掉帧/卡顿。

#### 7) 终止与失败模式

- `Episode/Real_Done_Rate`、`Episode/Timeout_Rate`  
  含义：真实失败终止 vs 超时终止占比。  
  怎么看：  
  - Real_Done 高 → 常摔/碰撞；  
  - Timeout 高 → 更稳但可能不够“积极”。  
  - 最理想是 Timeout 为主、同时回报高。

- `Terminations/*`（如 `base_contact`、`time_out`）  
  含义：各终止原因的细分比例。  
  怎么看：  
  - 用来精准定位“为什么结束”：是底盘碰地、关节爆限还是时间到。

---

**一套最实用的看图顺序（建议照抄）**
1. 先看 `Policy/Update_Accepted`、`Constraints/LineSearch_Step`、`Policy/KL`：确认“算法在学”。  
2. 再看 `Perf/Episode_Reward` + `Debug/*Error`：确认“学的是对的（能跟踪）”。  
3. 最后看 `Constraints/Cost_vs_Limit` + `Costs/ByTerm/*`：确认“学得安全”。  
4. 如果某一步不通过，就回到上一步找原因。

杀死所有进程

```bash
pkill -f tensorboard
```
