import gymnasium as gym

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from galileo_parkour.config.galileo_saferl_cfg import GalileoRoughEnvCfg
try:
    from .. import rewards as focpo_rewards
except ImportError:
    import rewards as focpo_rewards

TASK_ID = "Isaac-Velocity-Galileo-FOCPO-v0"
COST_KEYS = ["lin_vel_z_l2", "ang_vel_xy_l2", "flat_orientation_l2", "action_rate_l2"]


@configclass
class FOCPORewardsCfg:
    """Reward shaping dedicated to FOCPO using custom reward functions only."""

    track_lin_vel_xy_exp = RewTerm(
        func=focpo_rewards.track_lin_vel_xy_exp,
        weight=1.7,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    track_ang_vel_z_exp = RewTerm(
        func=focpo_rewards.track_ang_vel_z_exp,
        weight=0.8,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    alive = RewTerm(func=focpo_rewards.reward_alive, weight=2.0)
    feet_air_time = RewTerm(
        func=focpo_rewards.feet_air_time,
        weight=0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.3,
        },
    )
    base_height_l2 = RewTerm(
        func=focpo_rewards.base_height_l2,
        weight=-1.0,
        params={"target_height": 0.35, "asset_cfg": SceneEntityCfg("robot")},
    )
    ang_vel_xy_l2 = RewTerm(func=focpo_rewards.ang_vel_xy_l2, weight=-0.05)
    lin_vel_z_l2 = RewTerm(func=focpo_rewards.lin_vel_z_l2, weight=-1.2, params={"asset_cfg": SceneEntityCfg("robot")})
    action_rate_l2 = RewTerm(func=focpo_rewards.reward_action_rate, weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot")})
    dof_torques_l2 = RewTerm(func=focpo_rewards.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot")})
    dof_acc_l2 = RewTerm(func=focpo_rewards.reward_dof_acc, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot")})
    dof_pos_limits = RewTerm(func=focpo_rewards.joint_pos_limits, weight=1.0e-6, params={"asset_cfg": SceneEntityCfg("robot")})
    dof_error = RewTerm(
        func=focpo_rewards.reward_dof_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    flat_orientation_l2 = RewTerm(func=focpo_rewards.flat_orientation_l2, weight=-0.6, params={"asset_cfg": SceneEntityCfg("robot")})
    undesired_contacts = RewTerm(
        func=focpo_rewards.reward_collision,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*thigh", ".*calf"]), "threshold": 1.0},
    )


@configclass
class GalileoFOCPOEnvCfg(GalileoRoughEnvCfg):
    """Rough-terrain Galileo config dedicated to FOCPO."""

    rewards: FOCPORewardsCfg = FOCPORewardsCfg()

    def __post_init__(self):
        # Backup the undesired_contacts configuration because the parent class sets it to None
        undesired_contacts = self.rewards.undesired_contacts
        super().__post_init__()
        # Restore the undesired_contacts configuration
        self.rewards.undesired_contacts = undesired_contacts
        # Fix body names for Galileo/Go2
        if self.rewards.undesired_contacts is not None:
            self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*thigh", ".*calf"]

        # Slightly tighter control for first-order safe method.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.1, 1.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.events.push_robot = None
        self.rewards.lin_vel_z_l2.weight = -1.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.6
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 0.3
        if self.rewards.undesired_contacts is not None:
            self.rewards.undesired_contacts.weight = -2.0


def register_task():
    """Register a unique Gym ID for the FOCPO-specific env."""
    try:
        gym.spec(TASK_ID)
        return
    except gym.error.Error:
        pass

    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}:GalileoFOCPOEnvCfg"},
    )
