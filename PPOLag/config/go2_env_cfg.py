import gymnasium as gym

from isaaclab.utils import configclass
from galileo_parkour.config.galileo_saferl_cfg import GalileoRoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg as BaseRewardsCfg

# Unique identifiers and default cost taps for PPOLag.
TASK_ID = "Isaac-Velocity-Galileo-PPOLag-v0"
COST_KEYS = ["lin_vel_z_l2", "ang_vel_xy_l2", "action_rate_l2"]


@configclass
class PPOLagRewardsCfg(BaseRewardsCfg):
    """Reward shaping isolated for PPOLag to decouple from other algorithms."""

    def __post_init__(self):
        # Keep the base structure but bias slightly toward velocity tracking and smoother actions.
        self.track_lin_vel_xy_exp.weight = 2.0
        self.track_ang_vel_z_exp.weight = 1.0
        self.lin_vel_z_l2.weight = -0.75
        self.ang_vel_xy_l2.weight = -0.02
        self.action_rate_l2.weight = -0.006
        self.dof_torques_l2.weight = -1.2e-5
        self.feet_air_time.weight = 0.4


@configclass
class GalileoPPOLagEnvCfg(GalileoRoughEnvCfg):
    """Rough-terrain Galileo env config dedicated to PPOLag runs."""

    rewards: PPOLagRewardsCfg = PPOLagRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
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

