from pathlib import Path

from .galileo_env_cfg import TASK_ID, COST_KEYS

# Base directory for PPOLag runs (used by scripts for logs/checkpoints).
ALGO_ROOT = Path(__file__).resolve().parents[1]

# Training loop defaults (can still be overridden via CLI flags).
DEFAULT_TRAIN_CFG = {
    "seed": 0,
    "num_envs": 4096,
    "steps_per_env": 24,
    "max_iterations": 1000,
    "batch_size": 65536,  # 4096 envs * 16 steps per env
    "log_interval": 10,
    "save_interval": 100,
    "torch_threads": 4,
}

# Algorithm-specific hyperparameters for PPOLagrange.
ALGO_CFG = {
    "num_epochs": 5,
    "learning_rate": 1.0e-3,
    "gamma": 0.99,
    "lam": 0.95,
    "cost_gamma": 0.99,
    "cost_lam": 0.95,
    "clip_param": 0.2,
    "entropy_coef": 0.01,
    "value_loss_coef": 1.0,
    "cost_value_loss_coef": 1.0,
    "max_grad_norm": 1.0,
    "cost_limit": 25.0,
    "pid_kp": 0.1,
    "pid_ki": 0.01,
    "pid_kd": 0.01,
}

DEFAULT_TASK = TASK_ID
DEFAULT_COST_KEYS = COST_KEYS

