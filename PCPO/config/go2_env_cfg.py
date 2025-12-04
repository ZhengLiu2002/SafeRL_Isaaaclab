import gymnasium as gym

from isaaclab.utils import configclass
from galileo_parkour.config.galileo_saferl_cfg import GalileoRoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg as BaseRewardsCfg

TASK_ID = "Isaac-Velocity-Galileo-PCPO-v0"
COST_KEYS = ["lin_vel_z_l2", "ang_vel_xy_l2", "dof_torques_l2"]


@configclass
class PCPORewardsCfg(BaseRewardsCfg):
    """Reward shaping dedicated to PCPO."""

    def __post_init__(self):
        self.track_lin_vel_xy_exp.weight = 1.8
        self.track_ang_vel_z_exp.weight = 0.9
        self.lin_vel_z_l2.weight = -1.0
        self.ang_vel_xy_l2.weight = -0.04
        self.dof_torques_l2.weight = -1.5e-5
        self.action_rate_l2.weight = -0.008
        self.feet_air_time.weight = 0.35


@configclass
class GalileoPCPOEnvCfg(GalileoRoughEnvCfg):
    """Rough-terrain Galileo config dedicated to PCPO."""

    rewards: PCPORewardsCfg = PCPORewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-1.2, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.9, 0.9)
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.04
        self.rewards.action_rate_l2.weight = -0.008
        self.rewards.dof_torques_l2.weight = -1.5e-5
        self.rewards.feet_air_time.weight = 0.35


def register_task():
    """Register a unique Gym ID for the PCPO-specific env."""
    try:
        gym.spec(TASK_ID)
        return
    except gym.error.Error:
        pass

    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}:GalileoPCPOEnvCfg"},
    )

