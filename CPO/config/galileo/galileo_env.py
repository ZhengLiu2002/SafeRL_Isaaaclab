
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, quat_apply
try:
    from terrain_utils import get_height_map
except Exception:
    from ...terrain_utils import get_height_map  # type: ignore

class ConstraintManager:

    def __init__(self, cfg: Any, env: ManagerBasedRLEnv):
        self.cfg = cfg
        self.env = env
        self._term_cfgs: Dict[str, RewardTermCfg] = {}
        self._term_objs: Dict[str, Optional[ManagerTermBase]] = {}

        if cfg is None:
            return

        for name, term in vars(cfg).items():
            if not isinstance(term, RewardTermCfg):
                continue
            self._validate_term_binding(name, term)
            self._term_cfgs[name] = term

            func = term.func
            if isinstance(func, type) and issubclass(func, ManagerTermBase):
                self._term_objs[name] = func(term, env)
            else:
                self._term_objs[name] = None

    def _validate_term_binding(self, term_name: str, term_cfg: RewardTermCfg) -> None:
        params = getattr(term_cfg, "params", {}) or {}

        for key in ("asset_cfg", "sensor_cfg"):
            entity_cfg = params.get(key)
            if isinstance(entity_cfg, SceneEntityCfg):
                self._validate_scene_entity(term_name, entity_cfg)

    def _validate_scene_entity(self, term_name: str, entity_cfg: SceneEntityCfg) -> None:
        scene = getattr(self.env, "scene", None)
        if scene is None:
            raise ValueError("ConstraintManager requires env.scene to resolve constraint bindings.")

        entity_name = entity_cfg.name
        if str(entity_name) == "0":
            raise ValueError(
                f"Constraint '{term_name}' has an invalid scene entity name '0'. "
                "Check the constraint config for a misconfigured SceneEntityCfg."
            )
        if entity_name is None:
            return

        if hasattr(scene, "sensors") and isinstance(scene.sensors, dict) and entity_name in scene.sensors:
            target = scene.sensors[entity_name]
        elif hasattr(scene, "keys") and entity_name in scene.keys():
            target = scene[entity_name]
        else:
            try:
                target = scene[entity_name]
            except KeyError as exc:
                available = []
                if hasattr(scene, "keys"):
                    try:
                        available = list(scene.keys())
                    except Exception:
                        available = []
                raise ValueError(
                    f"Constraint '{term_name}' references missing scene entity '{entity_name}'. "
                    f"Available: {available}"
                ) from exc
            except Exception as exc:
                raise ValueError(
                    f"Constraint '{term_name}' could not resolve scene entity '{entity_name}'."
                ) from exc

        body_names = getattr(entity_cfg, "body_names", None)
        body_ids = getattr(entity_cfg, "body_ids", None)
        if body_names:
            finder = getattr(target, "find_bodies", None)
            if finder is None:
                raise ValueError(
                    f"Constraint '{term_name}' references bodies {body_names} on '{entity_name}', "
                    f"but the target does not support body resolution."
                )
            try:
                ids, _ = finder(body_names)
            except Exception as exc:
                raise ValueError(
                    f"Constraint '{term_name}' could not resolve bodies {body_names} on '{entity_name}'."
                ) from exc
            if ids is None or len(ids) == 0:
                raise ValueError(f"Constraint '{term_name}' could not resolve bodies {body_names} on '{entity_name}'.")
        elif body_ids is not None:
            if isinstance(body_ids, slice):
                if body_ids == slice(None):
                    return
                if hasattr(target, "num_bodies"):
                    stop = target.num_bodies if body_ids.stop is None else body_ids.stop
                    max_idx = stop - 1
                    if max_idx >= target.num_bodies:
                        raise ValueError(
                            f"Constraint '{term_name}' uses invalid body_ids slice ({body_ids}) for '{entity_name}'"
                        )
                return

            ids_seq = body_ids.tolist() if isinstance(body_ids, torch.Tensor) else body_ids
            if hasattr(target, "num_bodies") and isinstance(ids_seq, (list, tuple)) and len(ids_seq) > 0:
                max_idx = max(ids_seq)
                if max_idx >= target.num_bodies:
                    raise ValueError(
                        f"Constraint '{term_name}' uses invalid body_ids (max {max_idx}) for '{entity_name}'"
                    )

    def reset(self, env_ids: Optional[Iterable[int]] = None) -> None:
        if env_ids is None:
            target_env_ids = None
        elif isinstance(env_ids, torch.Tensor):
            target_env_ids = env_ids.to(device=self.env.device, dtype=torch.long)
        else:
            target_env_ids = torch.as_tensor(list(env_ids), device=self.env.device, dtype=torch.long)

        for term_obj in self._term_objs.values():
            if isinstance(term_obj, ManagerTermBase):
                term_obj.reset(target_env_ids)

    def compute(self) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for name, term_cfg in self._term_cfgs.items():
            term_obj = self._term_objs.get(name)
            if isinstance(term_obj, ManagerTermBase):
                val = term_obj(self.env, **term_cfg.params)
            else:
                val = term_cfg.func(self.env, **term_cfg.params)
            outputs[name] = val
        return outputs


class GalileoCPOEnv(ManagerBasedRLEnv):

    def __init__(self, cfg, render_mode: Optional[str] = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.constraint_manager = ConstraintManager(getattr(self.cfg, "constraints", None), self)
        self._debug_scene_logged = False
        self._retarget_base_contact_termination()
        
        self.max_terrain_level = 9
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.use_height_map = getattr(self.cfg, "use_height_map", False)
        self.height_map_obs_key = getattr(self.cfg, "height_map_obs_key", "height_map")
        self.height_map_sensor_cfg = getattr(self.cfg, "height_map_sensor_cfg", None)
        
        if hasattr(self.scene, "terrain") and hasattr(self.scene.terrain, "env_origins"):
            self.terrain_origins = self.scene.terrain.env_origins
            try:
                num_cols = self.terrain_origins.shape[1]
                cols = torch.arange(self.num_envs, device=self.device) % num_cols
                self.scene.env_origins = self.terrain_origins[self.terrain_levels, cols]
            except Exception:
                pass

    def _retarget_base_contact_termination(self):
        term_mgr = getattr(self, "termination_manager", None)
        if term_mgr is None or not hasattr(term_mgr, "get_term_cfg"):
            return
        try:
            cfg = term_mgr.get_term_cfg("base_contact")
        except Exception:
            cfg = None
        try:
            term = term_mgr.get_term("base_contact")
        except Exception:
            term = None

        if cfg is None:
            return

        sensor_cfg = None
        if isinstance(getattr(cfg, "params", None), dict):
            sensor_cfg = cfg.params.get("sensor_cfg", None)
        if sensor_cfg is None:
            return

        sensor_cfg.body_names = "base_link"

        try:
            contact_sensor = self.scene.sensors[sensor_cfg.name]
            body_ids, _ = contact_sensor.find_bodies(sensor_cfg.body_names)
            if term is not None:
                if hasattr(term, "body_ids"):
                    term.body_ids = body_ids
                if hasattr(term, "sensor_cfg"):
                    term.sensor_cfg.body_ids = body_ids
            sensor_cfg.body_ids = body_ids
        except Exception:
            pass

    def reset(self, *args, **kwargs):
        env_ids = kwargs.get("env_ids", None)
        
        if env_ids is not None:
            self._update_terrain_curriculum(env_ids)
        
        obs, info = super().reset(*args, **kwargs)

        obs = self._maybe_attach_height_map_obs(obs)
        
        
        if hasattr(self, "constraint_manager"):
            self.constraint_manager.reset(env_ids)
            
        return obs, info

    def _update_terrain_curriculum(self, env_ids):
        if not hasattr(self.scene, "terrain"):
            return

        robot = self.scene["robot"]
        root_states = robot.data.root_state_w
        
        distance = torch.norm(root_states[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        
        cmd_mgr = getattr(self, "command_manager", None)
        cmd_norm = torch.zeros_like(distance)
        if cmd_mgr:
            cmd = cmd_mgr.get_command("base_velocity")
            if cmd is not None:
                cmd_norm = torch.norm(cmd[env_ids, :2], dim=1)

        terrain_length = 8.0
        max_ep_len_s = self.max_episode_length_s
        
        move_up = distance > (terrain_length / 2.0)
        move_down = (distance < cmd_norm * max_ep_len_s * 0.5) & (~move_up)
        
        old_levels = self.terrain_levels[env_ids].clone()
        self.terrain_levels[env_ids] += 1 * move_up.long() - 1 * move_down.long()
        
        at_max = self.terrain_levels[env_ids] >= self.max_terrain_level
        if torch.any(at_max):
            self.terrain_levels[env_ids] = torch.where(
                at_max,
                torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                self.terrain_levels[env_ids]
            )
        
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0, self.max_terrain_level)
        
        if hasattr(self, "terrain_origins"):
            num_cols = self.terrain_origins.shape[1]
            cols = env_ids % num_cols
            new_origins = self.terrain_origins[self.terrain_levels[env_ids], cols]
            self.scene.env_origins[env_ids] = new_origins

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)

        obs = self._maybe_attach_height_map_obs(obs)

        self._apply_deadband_logic()

        try:
            if getattr(getattr(self, "cfg", None), "only_positive_rewards", False):
                rew = torch.clamp(torch.as_tensor(rew, device=self.device), min=0.0)
        except Exception:
            pass

        if hasattr(self, "constraint_manager"):
            constraint_vals = self.constraint_manager.compute()

            if not isinstance(info, dict):
                info = {}

            log_dst = info.setdefault("log", {})
            extras = info.setdefault("extras", {})
            extras_log = extras.setdefault("log", {})

            try:
                scene = getattr(self, "scene", None)
                robot = None
                if scene is not None:
                    if hasattr(scene, "__getitem__") and "robot" in scene:
                        robot = scene["robot"]
                    elif hasattr(scene, "robot"):
                        robot = scene.robot
                    elif hasattr(scene, "articulations") and len(getattr(scene, "articulations", [])) > 0:
                        robot = scene.articulations[0]
                if robot is None and hasattr(self, "robot"):
                    robot = self.robot
                data = getattr(robot, "data", None) if robot is not None else None
                if data is not None:
                    if hasattr(data, "root_lin_vel_b"):
                        log_dst["base_lin_vel_x"] = data.root_lin_vel_b[:, 0]
                        log_dst["base_lin_vel_y"] = data.root_lin_vel_b[:, 1]
                        log_dst["act_lin_vel_xy"] = torch.norm(data.root_lin_vel_b[:, :2], dim=1)
                    if hasattr(data, "root_ang_vel_b"):
                        log_dst["base_ang_vel_z"] = data.root_ang_vel_b[:, 2]
                        log_dst["act_ang_vel_z"] = data.root_ang_vel_b[:, 2]
                    if hasattr(data, "root_lin_vel_w") and "base_lin_vel_x" not in log_dst:
                        log_dst["base_lin_vel_x"] = data.root_lin_vel_w[:, 0]
                        log_dst["base_lin_vel_y"] = data.root_lin_vel_w[:, 1]
                        log_dst["act_lin_vel_xy"] = torch.norm(data.root_lin_vel_w[:, :2], dim=1)
                    if hasattr(data, "root_ang_vel_w") and "base_ang_vel_z" not in log_dst:
                        log_dst["base_ang_vel_z"] = data.root_ang_vel_w[:, 2]
                        log_dst["act_ang_vel_z"] = data.root_ang_vel_w[:, 2]
            except Exception:
                pass

            for name, val in constraint_vals.items():
                log_dst[name] = val
                extras_log[name] = val

        if not self._debug_scene_logged:
            try:
                scene = getattr(self, "scene", None)
                scene_keys = list(scene.keys()) if hasattr(scene, "keys") else []
                art_names = []
                if hasattr(scene, "articulations"):
                    try:
                        art_names = [getattr(a, "name", str(i)) for i, a in enumerate(scene.articulations)]
                    except Exception:
                        pass
                info.setdefault("log", {})["debug_scene_keys"] = scene_keys
                info["log"]["debug_articulations"] = art_names
            except Exception as e:
                pass
            self._debug_scene_logged = True

        try:
            cmd_mgr = getattr(self, "command_manager", None)
            if cmd_mgr and hasattr(cmd_mgr, "get_command"):
                cmd = cmd_mgr.get_command("base_velocity")
                if cmd is not None:
                    log_dst["command_lin_vel_x"] = cmd[:, 0]
                    log_dst["command_lin_vel_y"] = cmd[:, 1]
                    log_dst["command_ang_vel_z"] = cmd[:, 2]
                    if cmd.shape[1] > 3:
                         log_dst["command_heading"] = cmd[:, 3]

            scene = getattr(self, "scene", None)
            robot = None
            if scene is not None:
                if hasattr(scene, "__getitem__") and "robot" in scene:
                    robot = scene["robot"]
                elif hasattr(scene, "robot"):
                    robot = scene.robot
                elif hasattr(scene, "articulations") and len(getattr(scene, "articulations")) > 0:
                    robot = scene.articulations[0]

            data = getattr(robot, "data", None) if robot is not None else None
            if data is not None:
                if hasattr(data, "root_lin_vel_b"):
                    log_dst["base_lin_vel_x"] = data.root_lin_vel_b[:, 0]
                    log_dst["base_lin_vel_y"] = data.root_lin_vel_b[:, 1]
                    log_dst["act_lin_vel_xy"] = torch.norm(data.root_lin_vel_b[:, :2], dim=1)
                if hasattr(data, "root_ang_vel_b"):
                    log_dst["base_ang_vel_z"] = data.root_ang_vel_b[:, 2]
                    log_dst["act_ang_vel_z"] = data.root_ang_vel_b[:, 2]
        except Exception:
            pass

        return obs, rew, terminated, truncated, info

    def _maybe_attach_height_map_obs(self, obs):
        if not self.use_height_map:
            return obs

        try:
            hm = get_height_map(self, self.height_map_sensor_cfg)
        except Exception:
            hm = None

        if hm is None:
            hm = torch.zeros(self.num_envs, 1, device=self.device)
        else:
            if hm.ndim >= 3:
                hm = hm.reshape(hm.shape[0], -1)
            elif hm.ndim == 1:
                hm = hm.unsqueeze(-1)

        key = self.height_map_obs_key or "height_map"
        if isinstance(obs, dict):
            obs[key] = hm

        return obs

    def _apply_deadband_logic(self):
        cmd_mgr = getattr(self, "command_manager", None)
        if not cmd_mgr: return
        
        term = cmd_mgr.get_term("base_velocity")
        if not term: return
        
        cmds = getattr(term, "command", None)
        if cmds is None: return

        lin_norm = torch.norm(cmds[:, :2], dim=1)
        # 与命令采样逻辑保持一致，小于 0.2 m/s 的平移指令置零
        deadband_mask = lin_norm < 0.2
        cmds[deadband_mask, :2] = 0.0
        
