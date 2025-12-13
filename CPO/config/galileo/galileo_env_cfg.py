from pathlib import Path
import sys

import gymnasium as gym

from isaaclab.managers import EventTermCfg, RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

DEFAULT_COST_KEYS = (
    "joint_pos_limit",
    "joint_vel_limit",
    "joint_torque_limit",
    # "body_collision",
    # "com_height",
    "com_orientation",
)

try:
    from ... import rewards as cpo_rewards
except ImportError:
    import rewards as cpo_rewards

try:
    from ... import constraints as cpo_constraints
except ImportError:
    import constraints as cpo_constraints

def _load_galileo_asset_cfg():
    try:
        from ...assets.galileo import GALILEO_CFG
        return GALILEO_CFG
    except Exception:
        try:
            from assets.galileo import GALILEO_CFG
            return GALILEO_CFG
        except Exception:
            pass

    repo_root = Path(__file__).resolve().parents[4]
    ext_dir = repo_root / "source" / "extensions"
    if ext_dir.exists():
        ext_path = str(ext_dir)
        if ext_path not in sys.path:
            sys.path.append(ext_path)
        try:
            from galileo_parkour.assets.galileo import GALILEO_CFG
            return GALILEO_CFG
        except Exception:
            pass

    raise ImportError("无法导入 Galileo 资源配置，请确认本地 assets 或扩展可用。")


GALILEO_ASSET_CFG = _load_galileo_asset_cfg()

TASK_ID = "Isaac-Velocity-Galileo-CPO-v0"

@configclass
class SafetyConstraintsCfg:

    joint_pos_limit = RewTerm(
        func=cpo_constraints.constraint_joint_pos,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "margin": 0.3},
    )

    joint_vel_limit = RewTerm(
        func=cpo_constraints.constraint_joint_vel,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit": 50.0},
    )

    joint_torque_limit = RewTerm(
        func=cpo_constraints.constraint_joint_torque,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit": 100.0},
    )

    # body_collision = RewTerm(
    #     func=cpo_constraints.constraint_body_contact,
    #     weight=0.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*thigh", ".*calf","base_link"]),
    #         "threshold": 150.0,
    #     },
    # )

    # com_height = RewTerm(
    #     func=cpo_constraints.constraint_com_height,
    #     weight=0.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "min_height": 0.2,
    #         "max_height": 0.8,
    #     },
    # )

    com_orientation = RewTerm(
        func=cpo_constraints.constraint_com_orientation,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_angle_rad": 0.35},
    )


@configclass
class LocomotionRewardsCfg:
    track_lin_vel_xy = RewTerm(
        func=cpo_rewards.reward_track_lin_vel_xy,
        weight=2,
        params={"command_name": "base_velocity", "std": 0.5, 
                "asset_cfg": SceneEntityCfg("robot")}
    )
    track_ang_vel_z = RewTerm(
        func=cpo_rewards.reward_track_ang_vel_z,
        weight=1.,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    alive = RewTerm(
        func=cpo_rewards.reward_alive,
        weight=.5
    )
    feet_air_time = RewTerm(
        func=cpo_rewards.reward_feet_air_time,
        weight=3.0,
        params={
            # Use regex to match feet reliably across Galileo variants.
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    base_height = RewTerm(
        func=cpo_rewards.reward_base_height,
        weight=-1.0,
        params={"target_height": 0.4, "asset_cfg": SceneEntityCfg("robot")},
    )
    orientation = RewTerm(
        func=cpo_rewards.reward_orientation, 
        weight=-0.5, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    foot_symmetry = RewTerm(
        func=cpo_rewards.reward_foot_symmetry,
        weight=.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "height_scale": 0.12},
    )
    hip_pos = RewTerm(
        func=cpo_rewards.reward_hip_pos,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    ang_vel_xy = RewTerm(
        func=cpo_rewards.reward_ang_vel_xy, 
        weight=-0.1
    )

    lin_vel_z = RewTerm(
        func=cpo_rewards.reward_lin_vel_z, 
        weight=-1.0, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    action_rate = RewTerm(
        func=cpo_rewards.reward_action_rate, 
        weight=-0.01, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    collision = RewTerm(
        func=cpo_rewards.reward_collision,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*thigh", ".*calf","base_link"]),
            "threshold": 10.0,
        },
    )
    ground_impact = RewTerm(
        func=cpo_rewards.reward_ground_impact,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    torques = RewTerm(
        func=cpo_rewards.reward_torques, 
        weight=-1.0e-6, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    dof_acc = RewTerm(
        func=cpo_rewards.reward_dof_acc, 
        weight=-2.5e-7, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    dof_error = RewTerm(
        func=cpo_rewards.reward_dof_error,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class LocomotionEventCfg:

    base_external_force_torque = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (-1.0, 1.0),
            "torque_range": (-1.0, 1.0),
        },
    )

    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.7, 1.3), "velocity_range": (0.0, 0.0)},
    )

    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.55, 0.65), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(15.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
        },
    )

    randomize_friction = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )


@configclass
class GalileoOmniEnvCfg(LocomotionVelocityRoughEnvCfg):

    rewards: LocomotionRewardsCfg = LocomotionRewardsCfg()
    constraints: SafetyConstraintsCfg = SafetyConstraintsCfg()
    events: LocomotionEventCfg = LocomotionEventCfg()

    # 高程图/高度相对约束接口
    use_height_map: bool = True
    height_map_sensor_name: str = "height_scanner"
    height_map_obs_key: str = "height_map"
    height_map_target_height: float = 0.4  # 相对地面的期望高度
    height_map_min_height: float = 0.12   # 相对高度下限
    height_map_max_height: float = 0.90   # 相对高度上限

    # Deadband for tiny linear-velocity commands (m/s). Set to 0 to disable.
    deadband_lin_vel_threshold: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        # 允许负向奖励生效，便于策略从姿态/高度惩罚中学会稳定
        self.only_positive_rewards = False

        self.scene.robot = GALILEO_ASSET_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if getattr(self.terminations, "base_contact", None) is not None:
            self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"
        if getattr(self.events, "base_external_force_torque", None) is not None:
            self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        if getattr(self.events, "add_base_mass", None) is not None:
            self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        if getattr(self.scene, "height_scanner", None) is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
            # 按需开启高程图扫描
            if hasattr(self.scene.height_scanner, "enable"):
                self.scene.height_scanner.enable = self.use_height_map
        if getattr(self.events, "reset_base", None) is not None:
            self.events.reset_base.params = {
                "pose_range": {
                    "x": (-0.1, 0.1), 
                    "y": (-0.1, 0.1), 
                    "z": (0.0, 0.0), 
                    "yaw": (-3.14, 3.14)
                },
                "velocity_range": {
                    "x": (-0.0, 0.0),
                    "y": (-0.0, 0.0),
                    "z": (-0.0, 0.0),
                    "roll": (-0.0, 0.0),
                    "pitch": (-0.0, 0.0),
                    "yaw": (-0.0, 0.0),
                },
            }

        legs_actuator = getattr(getattr(self.scene, "robot", None), "actuators", {}).get("legs")
        if legs_actuator is not None:
            # 提高阻尼并适度增加刚度，减小落地振荡
            legs_actuator.stiffness = 100.0
            legs_actuator.damping = 4.0

        if hasattr(self.actions, "joint_pos"):
            # 缩小关节位置动作幅度，降低大幅动作导致的失稳
            self.actions.joint_pos.scale = 0.4

        self._apply_omni_command_ranges()

        self.cost_keys = list(DEFAULT_COST_KEYS)

        # 高程图观测/约束/奖励开关与参数绑定
        self.height_map_sensor_cfg = SceneEntityCfg(self.height_map_sensor_name)
        if self.use_height_map:
            if hasattr(self.rewards, "base_height"):
                self.rewards.base_height.params.update(
                    {
                        "use_height_map": True,
                        "height_sensor_cfg": self.height_map_sensor_cfg,
                        "target_height": self.height_map_target_height,
                        "clamp_error": 0.5,
                    }
                )

            if hasattr(self.constraints, "com_height"):
                self.constraints.com_height.params.update(
                    {
                        "use_height_map": True,
                        "height_sensor_cfg": self.height_map_sensor_cfg,
                        "min_height": self.height_map_min_height,
                        "max_height": self.height_map_max_height,
                    }
                )

    def _apply_omni_command_ranges(self):
        commands = getattr(self, "commands", None)
        if commands is None or not hasattr(commands, "base_velocity"):
            return

        base_velocity = commands.base_velocity
        if hasattr(base_velocity, "ranges"):
            # 收紧采样范围，与 legged_gym 默认命令方案保持一致
            base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
            base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
            base_velocity.ranges.ang_vel_z = (-.1, .1)
            base_velocity.ranges.heading = (-3.14, 3.14)
            
        base_velocity.heading_command = True
        if hasattr(base_velocity, "min_command_norm"):
            base_velocity.min_command_norm = 0.0
        if hasattr(base_velocity, "resampling_time_s"):
            base_velocity.resampling_time_s = 10.0


BaseRewardsCfg = LocomotionRewardsCfg
BaseEnvCfg = GalileoOmniEnvCfg


def register_task():
    try:
        gym.spec(TASK_ID)
        return
    except gym.error.Error:
        pass

    gym.register(
        id=TASK_ID,
        entry_point="config.galileo.galileo_env:GalileoCPOEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}:BaseEnvCfg"},
    )
