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
    
def constraint_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin: float = 0.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    idx = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    pos = asset.data.joint_pos[:, idx]

    if hasattr(asset.data, "soft_joint_pos_limits"):
        lower = asset.data.soft_joint_pos_limits[:, idx, 0] - margin
        upper = asset.data.soft_joint_pos_limits[:, idx, 1] + margin
    else:
        lower = torch.full_like(pos, -torch.inf)
        upper = torch.full_like(pos, torch.inf)

    violation = (pos < lower) | (pos > upper)
    return violation.float().mean(dim=1)


def constraint_joint_vel(
    env: ManagerBasedRLEnv,
    limit: float = 20.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    idx = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    vel = asset.data.joint_vel[:, idx]
    violation = torch.abs(vel) > limit
    return violation.float().mean(dim=1)


def constraint_joint_torque(
    env: ManagerBasedRLEnv,
    limit: float = 40.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    idx = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    torques = asset.data.applied_torque[:, idx]
    violation = torch.abs(torques) > limit
    return violation.float().mean(dim=1)


def constraint_body_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
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

    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, body_ids]
    collisions = torch.norm(net_contact_forces, dim=-1) > threshold
    return torch.any(collisions, dim=-1).float()


def constraint_com_height(
    env: ManagerBasedRLEnv,
    min_height: float = 0.25,
    max_height: float = 0.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_height_map: bool = False,
    height_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_z = asset.data.root_pos_w[:, 2]

    if use_height_map:
        ground_z = get_ground_height(env, height_sensor_cfg)
        rel_height = root_z - ground_z
        violation = (rel_height < min_height) | (rel_height > max_height)
    else:
        violation = (root_z < min_height) | (root_z > max_height)
    return violation.float()


def constraint_com_orientation(
    env: ManagerBasedRLEnv,
    max_angle_rad: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    grav_xy = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    limit = torch.sin(torch.tensor(max_angle_rad, device=env.device))
    violation = grav_xy > limit
    return violation.float()
