import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve()
ALGO_ROOT = SCRIPT_DIR.parents[1]
if str(ALGO_ROOT) not in sys.path:
    sys.path.append(str(ALGO_ROOT))

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from isaaclab.app import AppLauncher
from config.curriculum import get_current_stage, apply_stage_runtime


def _unwrap_env(env):
    if hasattr(env, "unwrapped"):
        base = env.unwrapped
        if hasattr(base, "reward_manager") and hasattr(base, "scene"):
            return base

    cur = env
    visited = set()
    while cur is not None and cur not in visited:
        visited.add(cur)

        if hasattr(cur, "reward_manager"):
            return cur

        if hasattr(cur, "env") and cur.env is not cur:
            cur = cur.env
            continue
        if hasattr(cur, "unwrapped") and cur.unwrapped is not cur:
            cur = cur.unwrapped
            continue
        break

    return cur


def main():
    parser = argparse.ArgumentParser(description="Train CPO Agent (optimized logging)")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--cost_keys", nargs="+", default=None, help="Constraint keys (default: constraint_collision)")
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--log_interval", type=int, default=None, help="Console logging interval")
    parser.add_argument("--save_interval", type=int, default=None, help="Checkpoint interval")
    parser.add_argument("--torch_threads", type=int, default=None, help="Torch intra-op threads")
    parser.add_argument(
        "--center_reward_adv_only",
        action="store_true",
        help="Only mean-center reward advantages (no variance norm), for ablations",
    )

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab_tasks.utils import parse_env_cfg
    from config.galileo.galileo_env_cfg import TASK_ID, register_task
    from config.galileo.galileo_cpo_cfg import ALGO_CFG, DEFAULT_TRAIN_CFG
    from algorithm import CPO
    from modules import ActorCritic
    from utils.tensor_utils import extract_obs, sanitize_term_key

    register_task()
    task_name = args.task or TASK_ID

    num_envs = args.num_envs or DEFAULT_TRAIN_CFG["num_envs"]
    seed = args.seed if args.seed is not None else DEFAULT_TRAIN_CFG["seed"]
    max_iterations = args.max_iterations or DEFAULT_TRAIN_CFG["max_iterations"]
    log_interval = args.log_interval or DEFAULT_TRAIN_CFG["log_interval"]
    save_interval = args.save_interval or DEFAULT_TRAIN_CFG["save_interval"]
    torch_threads = args.torch_threads or DEFAULT_TRAIN_CFG.get("torch_threads")
    if torch_threads:
        torch.set_num_threads(int(torch_threads))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    storage_device = DEFAULT_TRAIN_CFG["storage_device"]
    if str(storage_device).startswith("cuda") and not torch.cuda.is_available():
        storage_device = "cpu"

    env_cfg = parse_env_cfg(task_name, device=str(device), num_envs=num_envs)
    env_cfg.seed = seed
    env = gym.make(task_name, cfg=env_cfg)
    base_env = _unwrap_env(env)
    print(f"[Info] Base Env resolved to: {type(base_env)}")

    if args.cost_keys:
        cost_keys = args.cost_keys
    elif hasattr(env_cfg, "cost_keys"):
        cost_keys = list(getattr(env_cfg, "cost_keys"))
    else:
        cost_keys = ["undesired_contacts"]

    init_obs, _ = env.reset()
    init_obs_tensor = extract_obs(init_obs, device=device)
    num_obs = init_obs_tensor.shape[-1]

    action_space = env.single_action_space if hasattr(env, "single_action_space") else env.action_space
    num_actions = action_space.shape[-1]

    actor_critic = ActorCritic(num_obs, num_actions).to(device)

    # Keep a small allowlist of non-reward diagnostics we still want to log/plot.
    extra_key_groups = {
        "diagnostics": ("error_vel_xy", "error_vel_yaw"),
        "terminations": ("base_contact", "time_out"),
        "curriculum": ("terrain_levels", "command_ang_vel_z", "command_lin_vel_x", "command_lin_vel_y"),
    }
    extra_log_keys = [k for keys in extra_key_groups.values() for k in keys]

    def _active_reward_term_names(env_obj):
        try:
            rew_mgr = getattr(env_obj, "reward_manager", None)
            active = getattr(rew_mgr, "active_terms", None) if rew_mgr is not None else None
            if isinstance(active, dict):
                out = []
                for names in active.values():
                    out.extend(list(names))
                return list(dict.fromkeys(out))
            if isinstance(active, (list, tuple)):
                return list(dict.fromkeys(active))
            term_names = list(getattr(rew_mgr, "_term_names", []) or []) if rew_mgr is not None else []
            return list(dict.fromkeys(term_names))
        except Exception:
            return []

    reward_term_names = _active_reward_term_names(base_env)
    reward_log_keys = list(dict.fromkeys([*reward_term_names, *extra_log_keys])) if reward_term_names else None

    cpo_kwargs = ALGO_CFG.copy()
    cpo_kwargs.update(
        {
            "env": env,
            "actor_critic": actor_critic,
            "device": str(device),
            "num_envs": num_envs,
            "num_steps_per_env": DEFAULT_TRAIN_CFG["steps_per_env"],
            "cost_keys": cost_keys,
            "storage_device": storage_device,
            "init_obs": init_obs,
            "reward_log_keys": reward_log_keys,
            "center_reward_adv_only": args.center_reward_adv_only or ALGO_CFG.get("center_reward_adv_only", False),
        }
    )

    agent = CPO(**cpo_kwargs)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ALGO_ROOT / "logs" / "CPO" / task_name / run_name
    writer = None if args.no_tensorboard else SummaryWriter(log_dir / "tb")

    extra_reward_keys = {f"reward_{sanitize_term_key(k)}" for k in extra_log_keys}
    reward_diagnostic_keys = {f"reward_{sanitize_term_key(k)}" for k in extra_key_groups["diagnostics"]}
    reward_termination_keys = {f"reward_{sanitize_term_key(k)}" for k in extra_key_groups["terminations"]}
    reward_curriculum_keys = {f"reward_{sanitize_term_key(k)}" for k in extra_key_groups["curriculum"]}

    reward_component_names = {sanitize_term_key(n) for n in (reward_term_names or [])}
    reward_component_keys = {f"reward_{n}" for n in reward_component_names}
    allowed_reward_keys = (reward_component_keys | extra_reward_keys) if reward_component_keys else None

    def _collect_reward_weights(env_obj, term_names):
        weights = {}
        rew_mgr_local = getattr(env_obj, "reward_manager", None)
        if not term_names or rew_mgr_local is None or not hasattr(rew_mgr_local, "get_term_cfg"):
            return weights
        for term_name in term_names:
            try:
                term_cfg = rew_mgr_local.get_term_cfg(term_name)
            except Exception:
                term_cfg = None
            w = getattr(term_cfg, "weight", None) if term_cfg is not None else None
            if w is None:
                continue
            weights[sanitize_term_key(term_name).lower()] = float(w)
        return weights

    reward_weights = _collect_reward_weights(base_env, reward_term_names)

    def _is_zero_weight_reward(name: str) -> bool:
        w = reward_weights.get(sanitize_term_key(name).lower())
        return w is not None and abs(w) <= 1e-12

    print(f"[INFO] Training started. Logs: {log_dir}")

    steps_per_iter = num_envs * DEFAULT_TRAIN_CFG["steps_per_env"]

    current_stage = ""

    for it in range(max_iterations):
        stage = get_current_stage(it)
        if stage and stage.name != current_stage:
            print(f"\n[Curriculum] Stage -> {stage.name}: {stage.description}")
            apply_stage_runtime(base_env, agent, stage)
            reward_weights = _collect_reward_weights(base_env, reward_term_names)
            current_stage = stage.name

        iter_start = time.perf_counter()
        logs = agent.step()
        step_time = time.perf_counter() - iter_start
        fps = steps_per_iter / max(step_time, 1e-8)

        # Refresh reward weights for accurate filtering (weights can be modified by curriculum/runtime).
        if writer or (it % log_interval == 0):
            reward_weights = _collect_reward_weights(base_env, reward_term_names)

        if it % log_interval == 0:
            total_cost = logs.get("Meta/estimated_ep_cost", 0.0)
            act_abs_mean = logs.get("Debug/action_abs_mean", 0.0)
            act_abs_max = logs.get("Debug/action_abs_max", 0.0)
            obs_abs_mean = logs.get("Debug/obs_abs_mean", 0.0)
            ep_len = logs.get("Episode/len", logs.get("Perf/Episode_Length", 0.0))
            cost_per_step = logs.get("Meta/cost_per_step", 0.0)
            cmd_lin_xy = logs.get("Debug/cmd_lin_xy", 0.0)
            act_lin_xy = logs.get("Debug/act_lin_xy", 0.0)
            cmd_ang_z = logs.get("Debug/cmd_ang_vel_z", 0.0)
            act_ang_z = logs.get("Debug/act_ang_vel_z", 0.0)
            lin_xy_err = logs.get("Debug/lin_xy_err", 0.0)
            ang_vel_err = logs.get("Debug/ang_vel_z_err", 0.0)
            sigma_mean = logs.get("Policy/mean_action_std", 0.0)
            policy_entropy = logs.get("policy_entropy", 0.0)
            num_cmd = logs.get("Debug/num_cmd_samples", 0)
            num_act = logs.get("Debug/num_act_samples", 0)
            real_done_rate = logs.get("Episode/real_done_rate", None)
            timeout_rate = logs.get("Episode/timeout_rate", None)

            width = 80
            sep = "=" * width
            header = f" Learning Iteration {it} / {max_iterations} "

            def fmt(label, value):
                return f"{label:<45}{value:>15}"

            def _to_float(val):
                try:
                    return float(val)
                except Exception:
                    return val

            reward_terms = []
            diagnostic_terms = []
            termination_terms = []
            curriculum_terms = []
            other_reward_like = []

            for k, v in logs.items():
                if not k.startswith("reward_"):
                    continue
                if allowed_reward_keys is not None and k not in allowed_reward_keys:
                    continue

                pretty = k.replace("reward_", "", 1)
                if k in reward_component_keys and _is_zero_weight_reward(pretty):
                    continue
                value = _to_float(v)

                if k in reward_component_keys or (allowed_reward_keys is None and k not in extra_reward_keys):
                    reward_terms.append((pretty, value))
                elif k in reward_diagnostic_keys:
                    diagnostic_terms.append((pretty, value))
                elif k in reward_termination_keys:
                    termination_terms.append((pretty, value))
                elif k in reward_curriculum_keys:
                    curriculum_terms.append((pretty, value))
                else:
                    other_reward_like.append((pretty, value))

            reward_terms.sort(key=lambda x: x[0])
            diagnostic_terms.sort(key=lambda x: x[0])
            termination_terms.sort(key=lambda x: x[0])
            curriculum_terms.sort(key=lambda x: x[0])
            other_reward_like.sort(key=lambda x: x[0])

            mean_reward = logs.get("Meta/mean_reward", 0.0)
            kl_val = logs.get("kl", 0.0)
            max_step_cost = logs.get("Meta/max_step_cost", 0.0)
            cpo_nu = logs.get("cpo/nu", 0.0)
            cpo_lam = logs.get("cpo/lam", 0.0)
            cpo_step = logs.get("cpo/step_frac", 0.0)
            value_cost_loss = logs.get("cost_value_loss", 0.0)

            log_lines = [
                sep,
                header.center(width, " "),
                sep,
                "Performance:",
                f"    {fmt('Computation:', f'{fps:.0f} steps/s')}",
                f"    {fmt('Collection time:', f'{step_time:.3f} s')}",
                "",
                "Train:",
                f"    {fmt('Mean reward:', f'{mean_reward:.4f}')}",
                f"    {fmt('Mean action noise std:', f'{sigma_mean:.2f}')}",
                f"    {fmt('KL divergence:', f'{kl_val:.4f}')}",
                f"    {fmt('Policy entropy:', f'{policy_entropy:.3f}')}",
                f"    {fmt('Episode length:', f'{ep_len:.2f}')}",
            ]
            if real_done_rate is not None and timeout_rate is not None:
                log_lines.append(f"    {fmt('Done real / timeout:', f'{real_done_rate:.2f} / {timeout_rate:.2f}')}")
            log_lines += [
                "",
                "CPO (Constraint):",
                f"    {fmt('Mean cost (Est / Limit):', f'{total_cost:.2f} / {agent.cost_limit:.2f}')}",
                f"    {fmt('Cost per step:', f'{cost_per_step:.4f}')}",
                f"    {fmt('Max step cost:', f'{max_step_cost:.2f}')}",
                f"    {fmt('CPO nu / lam / step:', f'{cpo_nu:.4f} / {cpo_lam:.4f} / {cpo_step:.2f}')}",
                f"    {fmt('Value loss (Cost):', f'{value_cost_loss:.2f}')}",
                "",
            ]
            log_lines.append("Rewards:")
            if reward_terms:
                for name, val in reward_terms:
                    log_lines.append(f"    {fmt(f'{name}:', f'{val:.3f}')}")
            else:
                log_lines.append("    (no reward terms logged)")

            if diagnostic_terms:
                log_lines.append("")
                log_lines.append("Diagnostics:")
                for name, val in diagnostic_terms:
                    log_lines.append(f"    {fmt(f'{name}:', f'{val:.3f}')}")

            if termination_terms:
                log_lines.append("")
                log_lines.append("Terminations:")
                for name, val in termination_terms:
                    log_lines.append(f"    {fmt(f'{name}:', f'{val:.3f}')}")

            if curriculum_terms:
                log_lines.append("")
                log_lines.append("Curriculum:")
                for name, val in curriculum_terms:
                    log_lines.append(f"    {fmt(f'{name}:', f'{val:.3f}')}")

            if other_reward_like:
                log_lines.append("")
                log_lines.append("Other reward-like:")
                for name, val in other_reward_like:
                    log_lines.append(f"    {fmt(f'{name}:', f'{val:.3f}')}")
            log_lines.append(sep)
            print("\n".join(log_lines))

        if writer:
            global_step = it + 1

            episode_len = logs.get("Episode/len", 0.0)
            if episode_len == 0.0:
                try:
                    episode_len = float(agent.ep_len_buf.mean().item())
                except Exception:
                    episode_len = 0.0

            writer.add_scalar("Perf/Reward", logs.get("Meta/mean_reward", 0.0), global_step)
            writer.add_scalar("Perf/Episode_Reward", logs.get("Episode/reward", 0.0), global_step)
            writer.add_scalar("Perf/Episode_Length", episode_len, global_step)
            writer.add_scalar("Episode/Length", episode_len, global_step)
            writer.add_scalar("Perf/FPS", logs.get("Meta/fps", fps), global_step)

            writer.add_scalars(
                "Constraints/Cost_vs_Limit",
                {
                    "Estimated_Cost": logs.get("Meta/estimated_ep_cost", 0.0),
                    "Limit": agent.cost_limit,
                },
                global_step,
            )
            writer.add_scalar("Constraints/Lagrange_Multiplier_Nu", logs.get("cpo/nu", 0.0), global_step)
            writer.add_scalar("Constraints/Trust_Region_Lam", logs.get("cpo/lam", 0.0), global_step)
            writer.add_scalar("Constraints/LineSearch_Step", logs.get("cpo/step_frac", 0.0), global_step)
            writer.add_scalar("Constraints/Cost_Per_Step", logs.get("Meta/cost_per_step", 0.0), global_step)
            writer.add_scalar("Constraints/Violations_Per_Episode", logs.get("Meta/violations_per_ep", 0.0), global_step)
            writer.add_scalar("Curriculum/Cost_Limit", agent.cost_limit, global_step)

            try:
                terrain_level = None
                cfg_scene = getattr(base_env, "cfg", None).scene if hasattr(base_env, "cfg") else None
                terrain_cfg = getattr(cfg_scene, "terrain", None)
                if terrain_cfg and hasattr(terrain_cfg, "max_init_terrain_level"):
                    terrain_level = terrain_cfg.max_init_terrain_level
                elif hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "max_init_terrain_level"):
                    terrain_level = env.scene.terrain.max_init_terrain_level
                if terrain_level is not None:
                    writer.add_scalar("Curriculum/Terrain_Level", terrain_level, global_step)
            except Exception:
                pass

            writer.add_scalar("Policy/KL", logs.get("kl", 0.0), global_step)
            writer.add_scalar("Policy/Action_Std", logs.get("Policy/mean_action_std", 0.0), global_step)
            writer.add_scalar("Policy/Entropy", logs.get("policy_entropy", 0.0), global_step)
            writer.add_scalar("Policy/Action_Std_Min", logs.get("policy_std_min", 0.0), global_step)
            writer.add_scalar("Policy/Action_Std_Max", logs.get("policy_std_max", 0.0), global_step)
            writer.add_scalar(
                "Policy/Update_Accepted", float(logs.get("cpo/step_frac", 0.0) > 0.0), global_step
            )
            writer.add_scalar("Loss/Policy_Surr_Reward", logs.get("policy_surr_reward", 0.0), global_step)
            writer.add_scalar("Loss/Policy_Surr_Cost", logs.get("policy_surr_cost", 0.0), global_step)
            writer.add_scalar("Loss/Value", logs.get("value_loss", 0.0), global_step)
            writer.add_scalar("Loss/Cost_Value", logs.get("cost_value_loss", 0.0), global_step)
            writer.add_scalar("Episode/Real_Done_Rate", logs.get("Episode/real_done_rate", 0.0), global_step)
            writer.add_scalar("Episode/Timeout_Rate", logs.get("Episode/timeout_rate", 0.0), global_step)
            writer.add_scalar(
                "Meta/Update_Skipped_No_Complete_Ep", logs.get("Meta/update_skipped_no_complete_ep", 0.0), global_step
            )
            writer.add_scalar("Debug/Action_Abs_Mean", logs.get("Debug/action_abs_mean", 0.0), global_step)
            writer.add_scalar("Debug/Action_Abs_Max", logs.get("Debug/action_abs_max", 0.0), global_step)
            writer.add_scalar("Debug/Obs_Abs_Mean", logs.get("Debug/obs_abs_mean", 0.0), global_step)
            writer.add_scalar("Debug/Obs_Abs_Max", logs.get("Debug/obs_abs_max", 0.0), global_step)
            writer.add_scalar("Debug/Cmd_LinVel_XY", logs.get("Debug/cmd_lin_xy", 0.0), global_step)
            writer.add_scalar("Debug/Act_LinVel_XY", logs.get("Debug/act_lin_xy", 0.0), global_step)
            writer.add_scalar("Debug/LinVel_XY_Error", logs.get("Debug/lin_xy_err", 0.0), global_step)
            writer.add_scalar("Debug/Cmd_AngVel_Z", logs.get("Debug/cmd_ang_vel_z", 0.0), global_step)
            writer.add_scalar("Debug/Act_AngVel_Z", logs.get("Debug/act_ang_vel_z", 0.0), global_step)
            writer.add_scalar("Debug/AngVel_Z_Error", logs.get("Debug/ang_vel_z_err", 0.0), global_step)

            for key, val in logs.items():
                if key.startswith("reward_"):
                    if allowed_reward_keys is not None and key not in allowed_reward_keys:
                        continue
                    pretty = key.replace("reward_", "", 1)
                    if key in reward_component_keys and _is_zero_weight_reward(pretty):
                        continue
                    if key in reward_component_keys or (allowed_reward_keys is None and key not in extra_reward_keys):
                        tag = f"Rewards/Components/{pretty}"
                    elif key in reward_diagnostic_keys:
                        tag = f"Diagnostics/{pretty}"
                    elif key in reward_termination_keys:
                        tag = f"Terminations/{pretty}"
                    elif key in reward_curriculum_keys:
                        tag = f"Curriculum/{pretty}"
                    else:
                        tag = f"Rewards/Other/{pretty}"
                    writer.add_scalar(tag, val, global_step)
                elif key.startswith("constraint_"):
                    pretty = key.replace("constraint_", "", 1)
                    writer.add_scalar(f"Costs/ByTerm/{pretty}", val, global_step)

            writer.flush()

        if it > 0 and it % save_interval == 0:
            torch.save(actor_critic.state_dict(), log_dir / f"model_{it}.pt")

    torch.save(actor_critic.state_dict(), log_dir / "model_final.pt")
    print(f"[INFO] Final model saved to {log_dir / 'model_final.pt'}")

    if writer:
        writer.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
