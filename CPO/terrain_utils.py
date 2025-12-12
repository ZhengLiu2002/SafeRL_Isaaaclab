from __future__ import annotations

from typing import Any, Optional

import torch

try:
    from isaaclab.managers import SceneEntityCfg
except Exception:  # pragma: no cover - fallback when isaaclab is not available at import time
    SceneEntityCfg = Any  # type: ignore


def _resolve_sensor(env: Any, height_sensor_cfg: Any) -> Any:
    """Resolve a height sensor from the scene using a SceneEntityCfg or sensor name."""
    scene = getattr(env, "scene", None)
    sensor_name: Optional[str] = None

    if height_sensor_cfg is None:
        return None
    if isinstance(height_sensor_cfg, SceneEntityCfg):
        sensor_name = height_sensor_cfg.name
    elif isinstance(height_sensor_cfg, str):
        sensor_name = height_sensor_cfg

    if scene is None or sensor_name is None:
        return None

    sensors = getattr(scene, "sensors", None)
    if isinstance(sensors, dict) and sensor_name in sensors:
        return sensors[sensor_name]

    if hasattr(scene, sensor_name):
        return getattr(scene, sensor_name)

    return None


def get_height_map(env: Any, height_sensor_cfg: Any = None) -> Optional[torch.Tensor]:
    """Best-effort height-map retrieval from sensor or terrain.

    Returns:
        torch.Tensor: shape (num_envs, N) or (num_envs, H, W) when available.
        None: if no height information could be resolved.
    """
    sensor = _resolve_sensor(env, height_sensor_cfg)
    if sensor is not None:
        data = getattr(sensor, "data", None)
        for key in ("height_map", "heights", "height_samples", "samples", "height"):
            hm = getattr(data, key, None) if data is not None else None
            if hm is not None:
                return hm

    scene = getattr(env, "scene", None)
    terrain = getattr(scene, "terrain", None) if scene is not None else None
    if terrain is not None:
        for key in ("height_samples", "heights", "height_map"):
            hm = getattr(terrain, key, None)
            if hm is not None:
                return torch.as_tensor(hm, device=getattr(env, "device", None))

    return None


def get_ground_height(env: Any, height_sensor_cfg: Any = None) -> torch.Tensor:
    """Estimate ground height per environment.

    Uses height map if available, otherwise falls back to env origins' z or zeros.
    """
    height_map = get_height_map(env, height_sensor_cfg)
    if height_map is not None:
        if height_map.ndim == 1:
            return height_map
        if height_map.ndim == 2:
            return height_map.mean(dim=1)
        if height_map.ndim >= 3:
            return height_map.reshape(height_map.shape[0], -1).mean(dim=1)

    scene = getattr(env, "scene", None)
    origins = getattr(scene, "env_origins", None) if scene is not None else None
    if origins is not None:
        try:
            return origins[:, 2]
        except Exception:
            pass

    device = getattr(env, "device", None) or torch.device("cpu")
    num_envs = getattr(env, "num_envs", 1)
    return torch.zeros(num_envs, device=device)

