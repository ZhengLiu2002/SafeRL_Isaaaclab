import gymnasium as gym

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from galileo_parkour.config.galileo_saferl_cfg import GalileoRoughEnvCfg
try:
    from .. import rewards as ppolag_rewards
except ImportError:
    import rewards as ppolag_rewards

# Unique identifiers and default cost taps for PPOLag.
TASK_ID = "Isaac-Velocity-Galileo-PPOLag-v0"
COST_KEYS = ["lin_vel_z_l2", "ang_vel_xy_l2", "action_rate_l2"]


@configclass
class PPOLagRewardsCfg:
    """Reward shaping isolated for PPOLag to decouple from other algorithms."""

    track_lin_vel_xy_exp = RewTerm(
        func=ppolag_rewards.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    track_ang_vel_z_exp = RewTerm(
        func=ppolag_rewards.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    alive = RewTerm(func=ppolag_rewards.reward_alive, weight=2.0)
    feet_air_time = RewTerm(
        func=ppolag_rewards.feet_air_time,
        weight=0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.3,
        },
    )
    base_height_l2 = RewTerm(
        func=ppolag_rewards.base_height_l2,
        weight=-1.0,
        params={"target_height": 0.35, "asset_cfg": SceneEntityCfg("robot")},
    )
    ang_vel_xy_l2 = RewTerm(func=ppolag_rewards.ang_vel_xy_l2, weight=-0.02)
    lin_vel_z_l2 = RewTerm(func=ppolag_rewards.lin_vel_z_l2, weight=-0.75, params={"asset_cfg": SceneEntityCfg("robot")})
    action_rate_l2 = RewTerm(func=ppolag_rewards.reward_action_rate, weight=-0.006, params={"asset_cfg": SceneEntityCfg("robot")})
    dof_torques_l2 = RewTerm(func=ppolag_rewards.joint_torques_l2, weight=-1.2e-5, params={"asset_cfg": SceneEntityCfg("robot")})
    dof_acc_l2 = RewTerm(func=ppolag_rewards.reward_dof_acc, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot")})
    dof_pos_limits = RewTerm(func=ppolag_rewards.joint_pos_limits, weight=1.0e-6, params={"asset_cfg": SceneEntityCfg("robot")})
    dof_error = RewTerm(
        func=ppolag_rewards.reward_dof_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    flat_orientation_l2 = RewTerm(func=ppolag_rewards.flat_orientation_l2, weight=-0.4, params={"asset_cfg": SceneEntityCfg("robot")})
    undesired_contacts = RewTerm(
        func=ppolag_rewards.reward_collision,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*thigh", ".*calf"]), "threshold": 1.0},
    )


@configclass
class GalileoPPOLagEnvCfg(GalileoRoughEnvCfg):
    """Rough-terrain Galileo env config dedicated to PPOLag runs."""

    rewards: PPOLagRewardsCfg = PPOLagRewardsCfg()

    def __post_init__(self):
        # Backup the undesired_contacts configuration because the parent class sets it to None
        undesired_contacts = self.rewards.undesired_contacts
        super().__post_init__()
        # Restore the undesired_contacts configuration
        self.rewards.undesired_contacts = undesired_contacts
        # Fix body names for Galileo/Go2
        if self.rewards.undesired_contacts is not None:
            self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*thigh", ".*calf"]

        # Pin down spacing and command noise for PPOLag experiments.
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        # Encourage fewer flips while keeping exploration.
        self.rewards.lin_vel_z_l2.weight = -0.75
        self.rewards.ang_vel_xy_l2.weight = -0.02
        # Slightly higher reward for staying on the commanded heading.
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        # Make air-time reward more pronounced for gaits.
        self.rewards.feet_air_time.weight = 0.4
        # Keep torques/action smoothness gentle.
        self.rewards.action_rate_l2.weight = -0.006
        self.rewards.dof_torques_l2.weight = -1.2e-5
        if self.rewards.undesired_contacts is not None:
            self.rewards.undesired_contacts.weight = -2.0


def register_task():
    """Register a unique Gym ID for PPOLag so configs stay isolated."""
    try:
        gym.spec(TASK_ID)
        return
    except gym.error.Error:
        pass

    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}:GalileoPPOLagEnvCfg"},
    )
