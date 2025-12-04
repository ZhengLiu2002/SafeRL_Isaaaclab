from pathlib import Path

from .galileo_env_cfg import TASK_ID, COST_KEYS

ALGO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRAIN_CFG = {
    "seed": 0,
    "num_envs": 4096,
    "steps_per_env": 24,
    "batch_size": 65536,
    "max_iterations": 1000,
    "log_interval": 10,
    "save_interval": 100,
    "torch_threads": 4,
    "storage_device": "cpu",
}

ALGO_CFG = {
    "gamma": 0.99,
    "lam": 0.95,
    "cost_gamma": 0.99,
    "cost_lam": 0.95,
    "cost_limit": 25.0,
    "target_kl": 0.01,
    "entropy_coef": 0.0,
    "fvp_sample_size": 1024,
}

DEFAULT_TASK = TASK_ID
DEFAULT_COST_KEYS = COST_KEYS

