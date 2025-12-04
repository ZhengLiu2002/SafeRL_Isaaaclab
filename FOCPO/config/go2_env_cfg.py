import gymnasium as gym

from isaaclab.utils import configclass
from galileo_parkour.config.galileo_saferl_cfg import GalileoRoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg as BaseRewardsCfg

TASK_ID = "Isaac-Velocity-Galileo-FOCPO-v0"
COST_KEYS = ["lin_vel_z_l2", "ang_vel_xy_l2", "flat_orientation_l2", "action_rate_l2"]


@configclass
class FOCPORewardsCfg(BaseRewardsCfg):
    """Reward shaping dedicated to FOCPO."""

    def __post_init__(self):
        self.track_lin_vel_xy_exp.weight = 1.7
        self.track_ang_vel_z_exp.weight = 0.8
        self.lin_vel_z_l2.weight = -1.2
        self.ang_vel_xy_l2.weight = -0.05
        self.flat_orientation_l2.weight = -0.6
        self.action_rate_l2.weight = -0.01
        self.feet_air_time.weight = 0.3


@configclass
class GalileoFOCPOEnvCfg(GalileoRoughEnvCfg):
    """Rough-terrain Galileo config dedicated to FOCPO."""

    rewards: FOCPORewardsCfg = FOCPORewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Slightly tighter control for first-order safe method.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.1, 1.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.events.push_robot = None
        self.rewards.lin_vel_z_l2.weight = -1.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.6
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 0.3


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

