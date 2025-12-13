from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from terrain_utils import get_ground_height

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
else:
    ManagerBasedRLEnv = Any

def reward_alive(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def reward_track_lin_vel_xy(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)[:, :2]
    error = commands - asset.data.root_lin_vel_b[:, :2]
    return torch.exp(-torch.sum(error * error, dim=1) / (std**2 + 1e-6))


def reward_track_ang_vel_z(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)[:, 2]
    error = commands - asset.data.root_ang_vel_b[:, 2]
    return torch.exp(-torch.square(error) / (std**2 + 1e-6))

def reward_base_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_height_map: bool = False,
    height_sensor_cfg: SceneEntityCfg | None = None,
    clamp_error: float | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    base_z = asset.data.root_pos_w[:, 2]

    if use_height_map:
        ground_z = get_ground_height(env, height_sensor_cfg)
        base_z = base_z - ground_z

    err = base_z - target_height
    if clamp_error is not None:
        err = torch.clamp(err, min=-clamp_error, max=clamp_error)
    return torch.square(err)


def reward_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def reward_ang_vel_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def reward_lin_vel_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def reward_dof_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    default_pos = getattr(asset.data, "default_joint_pos", None)
    if default_pos is None:
        default_pos = torch.zeros_like(asset.data.joint_pos)
    idx = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    return torch.sum(torch.square(asset.data.joint_pos[:, idx] - default_pos[:, idx]), dim=1)

def reward_torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    idx = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    torques = asset.data.applied_torque[:, idx]
    torques = torques.clamp(min=-200.0, max=200.0)
    return torch.sum(torch.square(torques), dim=1)

def reward_foot_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_scale: float = 0.12,
) -> torch.Tensor:
    """鼓励左右成对脚的高度一致，抑制单腿高抬/蹦跳步态。"""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_ids = asset.find_bodies(".*_foot")[0]
    if foot_ids is None or len(foot_ids) < 2:
        return torch.ones(env.num_envs, device=env.device)

    foot_pos = asset.data.body_state_w[:, foot_ids, 2]
    if foot_pos.shape[1] >= 4:
        pairs = [(0, 1), (2, 3)]
    else:
        pairs = [(0, 1)]

    diffs = []
    for a, b in pairs:
        a = min(a, foot_pos.shape[1] - 1)
        b = min(b, foot_pos.shape[1] - 1)
        diffs.append(torch.abs(foot_pos[:, a] - foot_pos[:, b]))

    total_diff = torch.stack(diffs, dim=1).sum(dim=1)
    reward = torch.exp(-total_diff / (height_scale + 1e-6))
    return reward

def reward_hip_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*_hip_joint"),
) -> torch.Tensor:
    """专门约束髋关节位置，避免过大摆动影响步态对称性。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    default_pos = getattr(asset.data, "default_joint_pos", None)
    if default_pos is None:
        default_pos = torch.zeros_like(asset.data.joint_pos)
    return torch.sum(
        torch.square(asset.data.joint_pos[:, joint_ids] - default_pos[:, joint_ids]), dim=1
    )
class reward_action_rate(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.term_name = cfg.params.get("action_term", "joint_pos")
        action_term = env.action_manager.get_term(self.term_name)
        action_dim = action_term.raw_actions.shape[-1]
        self.previous_actions = torch.zeros(env.num_envs, action_dim, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.previous_actions.zero_()
            return

        if isinstance(env_ids, slice):
            self.previous_actions[env_ids] = 0.0
            return

        ids = torch.as_tensor(env_ids, device=self.previous_actions.device, dtype=torch.long)
        self.previous_actions[ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        action_term = env.action_manager.get_term(self.term_name)
        current_actions = action_term.raw_actions
        diff = current_actions - self.previous_actions
        self.previous_actions.copy_(current_actions)
        return torch.norm(diff, dim=1)
class reward_dof_acc(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.dt = env.cfg.decimation * env.cfg.sim.dt
        self.previous_joint_vel = torch.zeros(env.num_envs, asset.num_joints, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.previous_joint_vel.zero_()
            return

        if isinstance(env_ids, slice):
            self.previous_joint_vel[env_ids] = 0.0
            return

        ids = torch.as_tensor(env_ids, device=self.previous_joint_vel.device, dtype=torch.long)
        self.previous_joint_vel[ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        vel = asset.data.joint_vel
        acc = (vel - self.previous_joint_vel) / self.dt
        self.previous_joint_vel.copy_(vel)
        return torch.sum(torch.square(acc), dim=1)
    
class reward_collision(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.threshold = cfg.params.get("threshold", 0.1)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]

        if self.sensor_cfg.body_names is not None:
            self.body_ids, _ = self.contact_sensor.find_bodies(self.sensor_cfg.body_names)
        elif self.sensor_cfg.body_ids is not None:
            self.body_ids = self.sensor_cfg.body_ids
        else:
            self.body_ids = None

        if self.body_ids is None or len(self.body_ids) == 0:
            for cand in ["base_link", "base", ".*base.*"]:
                try:
                    self.body_ids, _ = self.contact_sensor.find_bodies(cand)
                    break
                except Exception:
                    continue
        if self.body_ids is None or len(self.body_ids) == 0:
            raise ValueError(f"collision: Could not resolve body IDs for sensor '{self.sensor_cfg.name}'")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        threshold: float = 0.1,
    ) -> torch.Tensor:
        th = threshold if threshold is not None else self.threshold
        net_contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.body_ids]
        collisions = torch.norm(net_contact_forces, dim=-1) > th
        return torch.any(collisions, dim=-1).float()


def reward_ground_impact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """惩罚足端接触力跃变，鼓励柔和落地。

    返回非负 penalty（越大越糟），建议在奖励配置中用负权重相加。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    cfg_body_ids = sensor_cfg.body_ids
    need_resolve = False
    if cfg_body_ids is None:
        need_resolve = True
    elif not isinstance(cfg_body_ids, slice) and len(cfg_body_ids) == 0:
        need_resolve = True

    if need_resolve and sensor_cfg.body_names is not None:
        body_ids, _ = contact_sensor.find_bodies(sensor_cfg.body_names)
    else:
        body_ids = cfg_body_ids

    if body_ids is None:
        return torch.zeros(env.num_envs, device=env.device)
    if not isinstance(body_ids, slice) and len(body_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    curr_forces = contact_sensor.data.net_forces_w_history[:, 0, body_ids]
    prev_forces = contact_sensor.data.net_forces_w_history[:, 1, body_ids]
    delta = curr_forces - prev_forces
    # ||f_t - f_{t-1}||^2，按足端求均值
    penalty = torch.mean(torch.sum(delta * delta, dim=-1), dim=1)
    return penalty


def reward_feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if not hasattr(contact_sensor.data, "last_air_time"):
        if not hasattr(env, "_warned_air_time"):
            print(f"[Warning] 'last_air_time' missing in sensor {sensor_cfg.name}; feet_air_time returns zeros.")
            env._warned_air_time = True
        return torch.zeros(env.num_envs, device=env.device)

    cfg_body_ids = sensor_cfg.body_ids
    need_resolve = False
    if cfg_body_ids is None:
        need_resolve = True
    elif not isinstance(cfg_body_ids, slice) and len(cfg_body_ids) == 0:
        need_resolve = True

    if need_resolve and sensor_cfg.body_names is not None:
        body_ids, _ = contact_sensor.find_bodies(sensor_cfg.body_names)
    else:
        body_ids = cfg_body_ids

    if body_ids is None:
        return torch.zeros(env.num_envs, device=env.device)
    if not isinstance(body_ids, slice) and len(body_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    curr_forces = contact_sensor.data.net_forces_w_history[:, 0, body_ids]
    prev_forces = contact_sensor.data.net_forces_w_history[:, 1, body_ids]
    is_contact = torch.norm(curr_forces, dim=-1) > 1.0
    was_contact = torch.norm(prev_forces, dim=-1) > 1.0
    first_contact = is_contact & (~was_contact)

    air_time = contact_sensor.data.last_air_time[:, body_ids]
    reward = torch.sum((air_time.clamp(max=1.0) - threshold) * first_contact.float(), dim=1)

    commands = env.command_manager.get_command(command_name)
    move_mask = torch.norm(commands[:, :2], dim=1) > 0.1
    return torch.clamp(reward * move_mask.float(), min=0.0)
