import gymnasium as gym

from isaaclab.utils import configclass
from galileo_parkour.config.galileo_saferl_cfg import GalileoRoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg as BaseRewardsCfg

TASK_ID = "Isaac-Velocity-Galileo-CPO-v0"
COST_KEYS = ["lin_vel_z_l2", "ang_vel_xy_l2", "flat_orientation_l2", "action_rate_l2"]


@configclass
class CPORewardsCfg(BaseRewardsCfg):
    """Reward shaping tuned for CPO with stronger safety penalties."""

    def __post_init__(self):
        self.track_lin_vel_xy_exp.weight = 1.6
        self.track_ang_vel_z_exp.weight = 0.8
        self.lin_vel_z_l2.weight = -1.5
        self.ang_vel_xy_l2.weight = -0.06
        self.action_rate_l2.weight = -0.012
        self.flat_orientation_l2.weight = -0.4
        self.undesired_contacts.weight = -2.0


@configclass
class GalileoCPOEnvCfg(GalileoRoughEnvCfg):
    """Rough-terrain Galileo config dedicated to CPO."""

    rewards: CPORewardsCfg = CPORewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Tighter command ranges to keep the robot conservative.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        # Reduce randomized pushes for stabler cost estimation.
        self.events.push_robot = None
        # Safety-focused penalties.
        self.rewards.lin_vel_z_l2.weight = -1.5
        self.rewards.ang_vel_xy_l2.weight = -0.06
        self.rewards.flat_orientation_l2.weight = -0.4
        self.rewards.action_rate_l2.weight = -0.012
        self.rewards.undesired_contacts.weight = -2.0


def register_task():
    """Register a unique Gym ID for the CPO-specific env."""
    try:
        gym.spec(TASK_ID)
        return
    except gym.error.Error:
        pass

    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}:GalileoCPOEnvCfg"},
    )

