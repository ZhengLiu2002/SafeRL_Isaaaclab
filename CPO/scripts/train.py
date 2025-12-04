import argparse
import inspect
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repo root and algorithm root are on PYTHONPATH.
SCRIPT_DIR = Path(__file__).resolve()
ALGO_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (ALGO_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from isaaclab.app import AppLauncher
from algorithm import CPO
from modules import ActorCritic
from utils.tensor_utils import extract_obs

ALGO_NAME = "CPO"


def _format_time(seconds: float) -> str:
    """Return hh:mm:ss string for readability."""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    parser = argparse.ArgumentParser(description="Train a CPO agent in Isaac Lab (standalone entry).")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--cost_keys", nargs="+", default=None, help="Cost keys from info (overrides config defaults).")
    parser.add_argument("--max_iterations", type=int, default=None, help="Max training iterations.")
    parser.add_argument("--steps_per_env", type=int, default=None, help="Rollout horizon per environment.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for updates.")
    parser.add_argument("--storage_device", type=str, default=None, help="Device for rollout storage (e.g., cpu to save GPU memory).")
    parser.add_argument("--log_interval", type=int, default=None, help="Iterations between console logs.")
    parser.add_argument("--save_interval", type=int, default=None, help="Iterations between checkpoints.")
    parser.add_argument("--run_name", type=str, default=None, help="Name of this run; defaults to timestamp.")
    parser.add_argument("--no_tensorboard", action="store_true", help="Disable TensorBoard logging (enabled by default).")

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch App
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Delayed import: requires SimulationApp to be initialized by AppLauncher
    from isaaclab_tasks.utils import parse_env_cfg
    import isaaclab_tasks
    
    from config.galileo_env_cfg import TASK_ID, COST_KEYS, register_task
    from config.galileo_cpo_cfg import ALGO_CFG, DEFAULT_TRAIN_CFG

    # Resolve defaults
    task_name = args.task if args.task is not None else TASK_ID
    num_envs = args.num_envs if args.num_envs is not None else DEFAULT_TRAIN_CFG["num_envs"]
    seed = args.seed if args.seed is not None else DEFAULT_TRAIN_CFG["seed"]
    max_iterations = args.max_iterations if args.max_iterations is not None else DEFAULT_TRAIN_CFG["max_iterations"]
    steps_per_env = args.steps_per_env if args.steps_per_env is not None else DEFAULT_TRAIN_CFG["steps_per_env"]
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_TRAIN_CFG["batch_size"]
    storage_device = args.storage_device if args.storage_device is not None else DEFAULT_TRAIN_CFG.get("storage_device")
    log_interval = args.log_interval if args.log_interval is not None else DEFAULT_TRAIN_CFG["log_interval"]
    save_interval = args.save_interval if args.save_interval is not None else DEFAULT_TRAIN_CFG["save_interval"]
    cost_keys = args.cost_keys if args.cost_keys is not None else COST_KEYS

    # Seed and threading
    torch.manual_seed(seed)
    torch.set_num_threads(DEFAULT_TRAIN_CFG.get("torch_threads", torch.get_num_threads()))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")
    device = torch.device(getattr(args, "device", "cuda:0"))

    # Make sure the task is registered to use the algorithm-local config.
    register_task()

    # Build env config from registry and override device/num_envs
    env_cfg = parse_env_cfg(task_name, device=str(device), num_envs=num_envs)
    env = gym.make(task_name, cfg=env_cfg)
    print(f"[INFO] Environment created: {env}")

    # Init Model
    init_obs, _ = env.reset()
    init_obs_tensor = extract_obs(init_obs, device=device)
    num_obs = init_obs_tensor.shape[-1]

    # Handle vectorized environments where action_space might be (num_envs, action_dim)
    action_space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)

    if action_space is not None:
        if hasattr(action_space, "shape"):
            if len(action_space.shape) > 1 and action_space.shape[0] == num_envs:
                num_actions = action_space.shape[-1]
            else:
                num_actions = action_space.shape[0]
        else:
            num_actions = init_obs_tensor.shape[-1]
    else:
        num_actions = init_obs_tensor.shape[-1]

    print(f"[INFO] Num Obs: {num_obs}, Num Actions: {num_actions}")

    actor_critic = ActorCritic(num_obs, num_actions).to(device)

    # Init Algorithm
    algo_params = ALGO_CFG.copy()
    algo_class = CPO

    print(f"[INFO] Initializing {ALGO_NAME}...")

    effective_storage = storage_device or str(device)
    print(f"[INFO] Steps/Env: {steps_per_env}, Batch Size: {batch_size}, Storage: {effective_storage}")

    algo_kwargs = {
        "env": env,
        "actor_critic": actor_critic,
        "device": str(device),
        "num_envs": num_envs,
        "num_steps_per_env": steps_per_env,
        "batch_size": batch_size,
        "cost_keys": cost_keys,
        "storage_device": storage_device,
    }
    algo_kwargs.update(algo_params)

    allowed_keys = set()
    for cls in algo_class.mro():
        if "__init__" in cls.__dict__:
            sig = inspect.signature(cls.__init__)
            for name, param in sig.parameters.items():
                if name == "self" or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                allowed_keys.add(name)
    filtered_kwargs = algo_kwargs if not allowed_keys else {k: v for k, v in algo_kwargs.items() if k in allowed_keys}
    ignored_keys = [] if not allowed_keys else [k for k in algo_kwargs if k not in allowed_keys]
    if ignored_keys:
        print(f"[WARN] Ignoring unsupported algo config keys for {ALGO_NAME}: {ignored_keys}")

    agent = algo_class(**filtered_kwargs)

    # Training Loop
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ALGO_ROOT / "logs" / ALGO_NAME / task_name / run_name
    os.makedirs(log_dir, exist_ok=True)
    writer = None
    if not args.no_tensorboard:
        tb_dir = os.path.join(log_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)

    print(f"[INFO] Starting Training Loop... Logs at {log_dir}")
    steps_per_iter = num_envs * steps_per_env
    log_interval = max(1, log_interval)
    start_time = time.perf_counter()

    # Moving average buffers for console logging
    lenbuffer = []
    rewbuffer = []

    for it in range(max_iterations):
        iter_start = time.perf_counter()
        logs = agent.step()
        iter_time = time.perf_counter() - iter_start
        fps = steps_per_iter / max(iter_time, 1e-6)

        # Detailed Logging
        if it % log_interval == 0:
            mean_reward = logs.get("mean_reward", 0.0)
            mean_cost = logs.get("mean_cost", 0.0)
            ep_rew = logs.get("episode/reward")
            ep_len = logs.get("episode/length")

            elapsed = time.perf_counter() - start_time
            total_steps = (it + 1) * steps_per_iter

            # Update buffers
            if ep_rew is not None:
                rewbuffer.append(ep_rew)
                if len(rewbuffer) > 100:
                    rewbuffer.pop(0)
            if ep_len is not None:
                lenbuffer.append(ep_len)
                if len(lenbuffer) > 100:
                    lenbuffer.pop(0)

            # TensorBoard Logging (Parkour Style Grouping)
            if writer:
                # Episode
                if ep_rew is not None:
                    writer.add_scalar("Episode/reward", ep_rew, total_steps)
                if ep_len is not None:
                    writer.add_scalar("Episode/length", ep_len, total_steps)
                if logs.get("episode/cost") is not None:
                    writer.add_scalar("Episode/cost", logs.get("episode/cost"), total_steps)

                # Loss
                writer.add_scalar("Loss/value_function", logs.get("value_loss", 0.0), total_steps)
                writer.add_scalar("Loss/surrogate", logs.get("loss", logs.get("reward_surrogate", 0.0)), total_steps)
                writer.add_scalar("Loss/cost_value_function", logs.get("cost_value_loss", 0.0), total_steps)
                writer.add_scalar("Loss/learning_rate", algo_params.get("learning_rate", 1e-3), total_steps)  # Approximate if not scheduled

                # Perf
                writer.add_scalar("Perf/total_fps", fps, total_steps)
                writer.add_scalar("Perf/collection_time", logs.get("collection_time", 0.0), total_steps)
                writer.add_scalar("Perf/learning_time", logs.get("learning_time", 0.0), total_steps)

                # Train
                writer.add_scalar("Train/mean_reward", mean_reward, total_steps)
                writer.add_scalar("Train/mean_cost", mean_cost, total_steps)
                writer.add_scalar("Train/entropy", logs.get("entropy", 0.0), total_steps)
                writer.add_scalar("Train/approx_kl", logs.get("approx_kl", 0.0), total_steps)
                writer.add_scalar("Train/clip_frac", logs.get("clip_frac", 0.0), total_steps)
                writer.add_scalar("Train/penalty", logs.get("penalty", 0.0), total_steps)

                # Diagnostics / Terms
                for key, val in logs.items():
                    if key.endswith("_term"):
                        clean_key = key.replace("_term", "")
                        if "cost" in key or "penalty" in key:
                            writer.add_scalar(f"CostTerms/{clean_key}", val, total_steps)
                        else:
                            writer.add_scalar(f"RewardTerms/{clean_key}", val, total_steps)

                writer.flush()

            # Console Logging (Parkour Style)
            width = 80
            pad = 35
            str_art = f" \033[1m {ALGO_NAME} iteration {it}/{max_iterations} \033[0m "

            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_art.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {logs.get('collection_time', 0.0):.3f}s, learning {logs.get('learning_time', 0.0):.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {logs.get('value_loss', 0.0):.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {logs.get('loss', 0.0):.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {logs.get('mean_action_std', 0.0):.4f}\n"""
                f"""{'Mean reward:':>{pad}} {mean_reward:.4f}\n"""
                f"""{'Mean cost:':>{pad}} {mean_cost:.4f}\n"""
                f"""{'Mean penalty:':>{pad}} {logs.get('penalty', 0.0):.4f}\n"""
            )

            if len(lenbuffer) > 0:
                avg_rew = sum(rewbuffer) / len(rewbuffer)
                avg_len = sum(lenbuffer) / len(lenbuffer)
                log_string += f"""{'Mean episode reward:':>{pad}} {avg_rew:.4f}\n"""
                log_string += f"""{'Mean episode length:':>{pad}} {avg_len:.4f}\n"""

            log_string += f"\n{'Reward Terms:':>{pad}}\n"
            for key, val in logs.items():
                if key.endswith("_term") and "cost" not in key and "penalty" not in key:
                    clean_key = key.replace("_term", "")
                    log_string += f"""{clean_key:>{pad}} {val:.4f}\n"""

            log_string += f"\n{'Cost Terms:':>{pad}}\n"
            for key, val in logs.items():
                if key.endswith("_term") and ("cost" in key or "penalty" in key):
                    clean_key = key.replace("_term", "")
                    log_string += f"""{clean_key:>{pad}} {val:.4f}\n"""

            log_string += (
                f"""{'-' * width}\n"""
                f"""{'Total steps:':>{pad}} {(it + 1) * steps_per_iter}\n"""
                f"""{'Time elapsed:':>{pad}} {_format_time(elapsed)}\n"""
            )

            if mean_reward == 0.0 and it > 5:
                log_string += f"\n\033[93m[WARN] Mean reward is EXACTLY 0.0! Check if rewards are defined/active.\033[0m\n"

            print(log_string)

        # Save
        if it % max(1, save_interval) == 0:
            torch.save(actor_critic.state_dict(), os.path.join(log_dir, f"model_{it}.pt"))

    if writer:
        writer.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
