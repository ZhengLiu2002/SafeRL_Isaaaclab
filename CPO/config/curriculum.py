from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CurriculumStage:
    name: str
    description: str
    iteration_start: int
    iteration_end: Optional[int] = None
    
    terrain_complexity: float = 0.0
    obstacle_density: float = 0.0
    obstacle_height_range: tuple = (0.0, 0.0)
    
    max_lin_vel: float = 1.0
    max_ang_vel: float = 1.0
    min_command_norm: float = 0.0
    command_resample_time_s: Optional[float] = None
    
    reward_weight_multipliers: Dict[str, float] = None
    
    cost_limit_multiplier: float = 1.0

    push_interval_range_s: Optional[Tuple[float, float]] = None
    push_velocity_range: Optional[Tuple[float, float]] = None
    friction_range: Optional[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.reward_weight_multipliers is None:
            self.reward_weight_multipliers = {}
        if self.push_interval_range_s is not None and len(self.push_interval_range_s) != 2:
            raise ValueError("push_interval_range_s 应为 (min, max)")
        if self.push_velocity_range is not None and len(self.push_velocity_range) != 2:
            raise ValueError("push_velocity_range 应为 (min, max)")
        if self.friction_range is not None and len(self.friction_range) != 2:
            raise ValueError("friction_range 应为 (min, max)")


DEFAULT_CURRICULUM = [
    CurriculumStage(
        name="stage_1_flat",
        description="平地热身，低速全向与姿态稳定",
        iteration_start=0,
        iteration_end=800,
        terrain_complexity=0.0,
        obstacle_density=0.0,
        obstacle_height_range=(0.0, 0.05),
        max_lin_vel=0.6,
        max_ang_vel=0.4,
        min_command_norm=0.05,
        command_resample_time_s=8.0,
        reward_weight_multipliers={
            "track_lin_vel_xy": 1.2,
            "track_ang_vel_z": 1.1,
            "orientation": 1.2,
            "feet_air_time": 0.8,
            "torques": 0.6,
            "action_rate": 0.8,
        },
        cost_limit_multiplier=1.2,
        friction_range=(0.9, 1.1),
    ),
    CurriculumStage(
        name="stage_2_gentle",
        description="轻微起伏/低台阶，逐步放开指令范围",
        iteration_start=800,
        iteration_end=1800,
        terrain_complexity=0.25,
        obstacle_density=0.2,
        obstacle_height_range=(0.0, 0.12),
        max_lin_vel=0.8,
        max_ang_vel=0.6,
        min_command_norm=0.08,
        command_resample_time_s=9.0,
        reward_weight_multipliers={
            "track_lin_vel_xy": 1.1,
            "track_ang_vel_z": 1.0,
            "orientation": 1.1,
            "feet_air_time": 0.9,
            "torques": 0.8,
            "action_rate": 0.9,
        },
        cost_limit_multiplier=1.05,
        push_interval_range_s=(22.0, 28.0),
        push_velocity_range=(0.6, 1.0),
        friction_range=(0.6, 1.1),
    ),
    CurriculumStage(
        name="stage_3_moderate",
        description="中等粗糙度/小台阶，加入外推与摩擦扰动",
        iteration_start=1800,
        iteration_end=3200,
        terrain_complexity=0.5,
        obstacle_density=0.45,
        obstacle_height_range=(0.05, 0.2),
        max_lin_vel=1.0,
        max_ang_vel=0.8,
        min_command_norm=0.10,
        command_resample_time_s=10.0,
        reward_weight_multipliers={
            "track_lin_vel_xy": 1.0,
            "track_ang_vel_z": 1.0,
            "orientation": 1.1,
            "feet_air_time": 1.0,
            "collision": 1.1,
            "torques": 1.0,
            "action_rate": 1.0,
        },
        cost_limit_multiplier=0.95,
        push_interval_range_s=(16.0, 22.0),
        push_velocity_range=(1.0, 1.4),
        friction_range=(0.5, 1.2),
    ),
    CurriculumStage(
        name="stage_4_rough",
        description="粗糙坡面/离散块，提升扰动强度与角速度",
        iteration_start=3200,
        iteration_end=5200,
        terrain_complexity=0.75,
        obstacle_density=0.65,
        obstacle_height_range=(0.1, 0.3),
        max_lin_vel=1.1,
        max_ang_vel=1.0,
        min_command_norm=0.12,
        command_resample_time_s=10.0,
        reward_weight_multipliers={
            "track_lin_vel_xy": 0.95,
            "track_ang_vel_z": 1.05,
            "orientation": 1.15,
            "feet_air_time": 1.05,
            "collision": 1.2,
            "torques": 1.1,
            "action_rate": 1.05,
        },
        cost_limit_multiplier=0.9,
        push_interval_range_s=(12.0, 18.0),
        push_velocity_range=(1.2, 1.6),
        friction_range=(0.45, 1.2),
    ),
    CurriculumStage(
        name="stage_5_extreme",
        description="高难度混合地形，收紧成本限制",
        iteration_start=5200,
        iteration_end=None,
        terrain_complexity=1.0,
        obstacle_density=0.85,
        obstacle_height_range=(0.15, 0.4),
        max_lin_vel=1.2,
        max_ang_vel=1.2,
        min_command_norm=0.15,
        command_resample_time_s=8.0,
        reward_weight_multipliers={
            "track_lin_vel_xy": 0.9,
            "track_ang_vel_z": 1.1,
            "orientation": 1.2,
            "feet_air_time": 1.05,
            "collision": 1.25,
            "torques": 1.25,
            "action_rate": 1.1,
        },
        cost_limit_multiplier=0.85,
        push_interval_range_s=(10.0, 14.0),
        push_velocity_range=(1.4, 1.8),
        friction_range=(0.4, 1.25),
    ),
]


def get_current_stage(iteration: int, curriculum: list = None) -> Optional[CurriculumStage]:
    if curriculum is None:
        curriculum = DEFAULT_CURRICULUM
    
    for stage in curriculum:
        if iteration >= stage.iteration_start:
            if stage.iteration_end is None or iteration < stage.iteration_end:
                return stage
    
    if curriculum:
        return curriculum[-1]
    return None


def apply_curriculum_to_config(
    base_config: Dict[str, Any],
    stage: CurriculumStage,
) -> Dict[str, Any]:
    config = base_config.copy()
    
    if "rewards" in config and stage.reward_weight_multipliers:
        for reward_name, multiplier in stage.reward_weight_multipliers.items():
            if hasattr(config["rewards"], reward_name):
                reward_term = getattr(config["rewards"], reward_name)
                if hasattr(reward_term, "weight"):
                    if not hasattr(reward_term, "_base_weight"):
                        reward_term._base_weight = reward_term.weight
                    reward_term.weight = reward_term._base_weight * multiplier
    
    if "cost_limit" in config:
        base_limit = config.get("_base_cost_limit", config.get("cost_limit"))
        if base_limit is not None:
            config["_base_cost_limit"] = base_limit
            config["cost_limit"] = base_limit * stage.cost_limit_multiplier
    
    return config


def _find_reward_term_cfg(rew_mgr, reward_name: str):
    if not hasattr(rew_mgr, "get_term_cfg"):
        return None, None
    try:
        term_cfg = rew_mgr.get_term_cfg(reward_name)
        if term_cfg is not None:
            return term_cfg, reward_name
    except Exception:
        term_cfg = None
    for key in getattr(rew_mgr, "terms", {}).keys():
        if str(key) == reward_name or str(key).endswith(reward_name):
            try:
                term_cfg = rew_mgr.get_term_cfg(key)
                if term_cfg is not None:
                    return term_cfg, key
            except Exception:
                continue
    return None, None


def apply_stage_runtime(base_env, agent, stage: CurriculumStage, verbose: bool = True):
    if stage is None:
        return

    prefix = "[Curriculum]"

    def _log(msg: str):
        if verbose:
            print(f"  {prefix} {msg}")

    cmd_mgr = getattr(base_env, "command_manager", None)
    if cmd_mgr and hasattr(cmd_mgr, "get_term"):
        term = cmd_mgr.get_term("base_velocity")
        cfg = getattr(term, "cfg", None) if term else None
        ranges = getattr(cfg, "ranges", None) if cfg else None
        try:
            if ranges:
                ranges.lin_vel_x = (-stage.max_lin_vel, stage.max_lin_vel)
                if hasattr(ranges, "lin_vel_y"):
                    ranges.lin_vel_y = (-stage.max_lin_vel, stage.max_lin_vel)
                ranges.ang_vel_z = (-stage.max_ang_vel, stage.max_ang_vel)
                _log(f"命令范围 -> lin ±{stage.max_lin_vel:.2f}, yaw ±{stage.max_ang_vel:.2f}")
            if cfg and stage.min_command_norm > 0.0 and hasattr(cfg, "min_command_norm"):
                cfg.min_command_norm = stage.min_command_norm
                _log(f"最小指令幅值 -> {stage.min_command_norm:.2f}")
            if cfg and stage.command_resample_time_s is not None and hasattr(cfg, "resampling_time_s"):
                cfg.resampling_time_s = stage.command_resample_time_s
                _log(f"指令重采样周期 -> {stage.command_resample_time_s:.1f}s")
        except Exception as e:
            _log(f"[warn] 命令课程更新失败: {e}")
    else:
        _log("[warn] command_manager 未找到，跳过命令课程")

    if agent is not None and hasattr(agent, "cost_limit"):
        if not hasattr(agent, "_base_cost_limit"):
            agent._base_cost_limit = agent.cost_limit
        agent.cost_limit = agent._base_cost_limit * stage.cost_limit_multiplier
        _log(f"成本上限 -> {agent.cost_limit:.3f}")

    rew_mgr = getattr(base_env, "reward_manager", None)
    if rew_mgr:
        active_names = list(getattr(rew_mgr, "active_terms", []))
        for name, mult in stage.reward_weight_multipliers.items():
            term_cfg, resolved_key = _find_reward_term_cfg(rew_mgr, name)
            if term_cfg is None:
                _log(f"[warn] 奖励项 '{name}' 未找到，可用: {active_names}")
                continue
            try:
                base_w = getattr(term_cfg, "_base_weight", term_cfg.weight)
                term_cfg._base_weight = base_w
                term_cfg.weight = base_w * mult
                _log(f"奖励权重 {resolved_key} -> {term_cfg.weight:.3f}")
            except Exception as e:
                _log(f"[warn] 奖励项 '{name}' 更新失败: {e}")
    else:
        _log("[warn] reward_manager 未找到，跳过奖励课程")

    terrain_obj = getattr(getattr(base_env, "scene", None), "terrain", None)
    scene_cfg = getattr(getattr(base_env, "cfg", None), "scene", None)
    terrain_cfg = getattr(scene_cfg, "terrain", None) if scene_cfg else None
    try:
        max_level = getattr(terrain_cfg, "num_difficulty_levels", None)
        if max_level is None:
            gen = getattr(terrain_cfg, "terrain_generator", None) if terrain_cfg else None
            if gen and hasattr(gen, "num_rows") and hasattr(gen, "num_cols"):
                max_level = max(gen.num_rows, gen.num_cols)
        if max_level is None:
            max_level = 1
        target_level = max(0, min(int(round(stage.terrain_complexity * max(max_level - 1, 1))), max_level))
        if terrain_obj and hasattr(terrain_obj, "cfg"):
            terrain_obj.cfg.max_init_terrain_level = target_level
            _log(f"地形等级 -> {target_level}")
        gen_cfg = getattr(terrain_cfg, "terrain_generator", None) if terrain_cfg else None
        if gen_cfg:
            if hasattr(gen_cfg, "obstacle_density"):
                gen_cfg.obstacle_density = stage.obstacle_density
            if hasattr(gen_cfg, "obstacle_height_range"):
                gen_cfg.obstacle_height_range = stage.obstacle_height_range
            if hasattr(gen_cfg, "terrain_complexity"):
                gen_cfg.terrain_complexity = stage.terrain_complexity
    except Exception as terrain_err:
        _log(f"[warn] 地形课程更新失败: {terrain_err}")

    evt_mgr = getattr(base_env, "event_manager", None)
    if evt_mgr and hasattr(evt_mgr, "get_term_cfg"):
        if stage.push_interval_range_s or stage.push_velocity_range:
            try:
                push_cfg = evt_mgr.get_term_cfg("push_robot")
                if push_cfg:
                    if stage.push_interval_range_s is not None:
                        push_cfg.interval_range_s = stage.push_interval_range_s
                        _log(f"推力间隔 -> {stage.push_interval_range_s}")
                    if stage.push_velocity_range is not None:
                        v_min, v_max = stage.push_velocity_range
                        v_mag = max(abs(v_min), abs(v_max))
                        params = push_cfg.params if isinstance(push_cfg.params, dict) else {}
                        vel_range = params.get("velocity_range", {})
                        vel_range["x"] = (-v_mag, v_mag)
                        vel_range["y"] = (-v_mag, v_mag)
                        params["velocity_range"] = vel_range
                        push_cfg.params = params
                        _log(f"推力幅值 -> ±{v_mag:.2f} m/s")
            except Exception as e:
                _log(f"[warn] 推力课程更新失败: {e}")

        if stage.friction_range is not None:
            try:
                fric_cfg = evt_mgr.get_term_cfg("randomize_friction")
                if fric_cfg and isinstance(fric_cfg.params, dict):
                    fric_cfg.params["static_friction_range"] = stage.friction_range
                    fric_cfg.params["dynamic_friction_range"] = stage.friction_range
                    _log(f"摩擦系数范围 -> {stage.friction_range}")
            except Exception as e:
                _log(f"[warn] 摩擦课程更新失败: {e}")
    else:
        if stage.push_interval_range_s or stage.push_velocity_range or stage.friction_range:
            _log("[warn] event_manager 未找到，跳过事件课程")
