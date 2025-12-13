from pathlib import Path

from .galileo_env_cfg import TASK_ID

ALGO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRAIN_CFG = {
    "seed": 42,
    "num_envs": 2048,
    "steps_per_env": 48,
    "batch_size": 98304,
    "max_iterations": 3000,
    "log_interval": 10,
    "save_interval": 100,
    "torch_threads": 4,
    "storage_device": None,
}

ALGO_CFG = {
    "gamma": 0.99,
    "lam": 0.95,
    "cost_gamma": 0.99,
    "cost_lam": 0.95,
    "cost_limit": 400.0,
    "target_kl": 0.02,
    "damping_coeff": 0.1,
    "backtrack_coeff": 0.8,
    "max_backtracks": 15,
    "entropy_coef": 0.0,
    "fvp_sample_size": 4096,
    "critic_lr": 1e-3,
    "cost_critic_lr": 1e-3,
    "value_epochs": 5,
    "value_mini_batches": 4,
    "value_clip_grad_norm": 1.0,
    "center_reward_adv_only": False,
}

DEFAULT_TASK = TASK_ID
