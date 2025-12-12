import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve()
ALGO_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (ALGO_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher
from modules import ActorCritic
from utils.tensor_utils import extract_obs

def main():
    parser = argparse.ArgumentParser(description="Play/Evaluate a CPO agent in Isaac Lab.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model (.pt).")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of vectorized environments to run.")
    parser.add_argument("--num_steps", type=int, default=100000, help="Maximum steps to run.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions (with policy std) instead of deterministic mean.")
    parser.add_argument("--log_interval", type=int, default=200, help="How often to print rollout diagnostics.")

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    device = torch.device(getattr(args, "device", "cuda:0"))
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from config.galileo.galileo_env_cfg import TASK_ID as DEFAULT_TASK, register_task

    task_name = args.task or DEFAULT_TASK
    register_task()

    from isaaclab_tasks.utils import parse_env_cfg
    import isaaclab_tasks

    env_cfg = parse_env_cfg(task_name, device=str(device), num_envs=args.num_envs)
    env = gym.make(task_name, cfg=env_cfg)

    init_obs, _ = env.reset()
    init_obs_tensor = extract_obs(init_obs, device=device)
    num_obs = init_obs_tensor.shape[-1]

    action_space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)
    if hasattr(action_space, "shape") and len(action_space.shape) > 1 and action_space.shape[0] == args.num_envs:
        num_actions = action_space.shape[-1]
    else:
        num_actions = action_space.shape[0]

    actor = ActorCritic(num_obs, num_actions).to(device)

    print(f"[INFO] Loading model from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()

    obs = init_obs_tensor

    step_count = 0
    log_interval = max(1, args.log_interval)
    while simulation_app.is_running() and step_count < args.num_steps:
        with torch.no_grad():
            if args.stochastic:
                action = actor.act(obs)
            else:
                action = actor.act_inference(obs)

        obs, rew, terminated, truncated, info = env.step(action)
        obs = extract_obs(obs, device=device)
        step_count += 1

        if step_count % log_interval == 0:
            a_mean = action.mean().item()
            a_std = action.std().item()
            rew_mean = torch.as_tensor(rew).mean().item() if hasattr(torch, "as_tensor") else float(rew.mean())
            done_any = (terminated | truncated).any().item() if hasattr(torch, "as_tensor") else bool((terminated or truncated))
            print(f"[INFO] step {step_count}: action mean {a_mean:+.3f}, std {a_std:.3f}, reward {rew_mean:+.3f}, any_done={done_any}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
