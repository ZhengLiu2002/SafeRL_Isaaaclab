import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.optim as optim

try:
    from .modules import ActorCritic
    from .storage import RolloutStorage
    from .utils.tensor_utils import (
        extract_costs,
        extract_log_dict,
        extract_obs,
        sanitize_term_key,
        to_device_tensor,
    )
except ImportError:
    from modules import ActorCritic
    from storage import RolloutStorage
    from utils.tensor_utils import (
        extract_costs,
        extract_log_dict,
        extract_obs,
        sanitize_term_key,
        to_device_tensor,
    )


class CPO:
    """
    约束策略优化（CPO）训练器，在最大化回报的同时满足成本上限，通过 rollout、优势计算、价值更新和约束更新闭环。
    """
    def __init__(
        self,
        env,
        actor_critic: ActorCritic,
        device: str = "cpu",
        num_envs: int = 1,
        num_steps_per_env: int = 24,
        gamma: float = 0.99,
        lam: float = 0.95,
        cost_gamma: float = 0.99,
        cost_lam: float = 0.95,
        cost_limit: float = 25.0,
        target_kl: float = 0.01,
        damping_coeff: float = 0.1,
        backtrack_coeff: float = 0.8,
        max_backtracks: int = 10,
        entropy_coef: float = 0.0,
        critic_lr: float = 1e-3,
        cost_critic_lr: float = 1e-3,
        value_epochs: int = 5,
        value_mini_batches: int = 4,
        value_clip_grad_norm: float = 1.0,
        cost_keys: Optional[List[str]] = None,
        storage_device: Optional[str] = None,
        fvp_sample_size: Optional[int] = None,
        reward_log_keys: Optional[List[str]] = None,
        init_obs=None,
        center_reward_adv_only: bool = False,
        **kwargs,
    ):
        self.env = env
        self.actor_critic = actor_critic
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device) if storage_device else self.device

        self.num_envs = num_envs
        self.num_steps_per_env = num_steps_per_env
        self.gamma, self.lam = gamma, lam
        self.cost_gamma, self.cost_lam = cost_gamma, cost_lam
        self.cost_limit = cost_limit
        self.target_kl = target_kl
        self.damping_coeff = damping_coeff
        self.backtrack_coeff, self.max_backtracks = backtrack_coeff, max_backtracks
        self.entropy_coef = entropy_coef
        self.critic_lr = critic_lr
        self.cost_critic_lr = cost_critic_lr
        self.value_epochs = value_epochs
        self.value_mini_batches = value_mini_batches
        self.value_clip_grad_norm = value_clip_grad_norm
        self.fvp_sample_size = fvp_sample_size
        self.cost_keys = cost_keys or []
        self.cost_ema = None
        self.cost_ema_alpha = 0.9
        self.center_reward_adv_only = center_reward_adv_only
        self._base_env = getattr(env, "unwrapped", env)

        self._sanitized_cost_keys = {sanitize_term_key(k).lower() for k in self.cost_keys}
        if reward_log_keys:
            self._reward_log_allowlist = {sanitize_term_key(k).lower() for k in reward_log_keys}
        else:
            self._reward_log_allowlist = None

        if init_obs is None:
            init_obs, _ = self.env.reset()
        self.current_obs = extract_obs(init_obs, device=self.device)
        self.num_envs = getattr(env, "num_envs", self.current_obs.shape[0] if self.current_obs.dim() > 1 else 1)
        obs_shape = self.current_obs.shape[1:] if self.current_obs.dim() > 1 else self.current_obs.shape

        if hasattr(env, "single_action_space"):
            action_shape = env.single_action_space.shape
        elif hasattr(env, "action_space") and len(env.action_space.shape) > 1 and env.action_space.shape[0] == self.num_envs:
            action_shape = env.action_space.shape[1:]
        else:
            action_shape = env.action_space.shape

        pin_memory = self.storage_device.type == "cpu" and self.device.type == "cuda"
        self.storage = RolloutStorage(
            self.num_envs, num_steps_per_env, obs_shape, action_shape, self.storage_device, pin_memory=pin_memory
        )

        self.critic_optimizer = optim.Adam(self.actor_critic.critic.parameters(), lr=self.critic_lr)
        self.cost_critic_optimizer = optim.Adam(self.actor_critic.cost_critic.parameters(), lr=self.cost_critic_lr)

        # 用于在多个环境间累积训练回合统计的缓冲区。
        self.ep_rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.ep_cost_buf = torch.zeros(self.num_envs, device=self.device)
        self.ep_len_buf = torch.zeros(self.num_envs, device=self.device)

        self.global_ep_stats = defaultdict(float)
        self.global_ep_count = 0

    def step(self) -> Dict[str, float]:
        self.actor_critic.train()
        t_start = time.perf_counter()

        log_sums = defaultdict(float)
        log_counts = defaultdict(int)
        last_actions = None
        real_done_count = 0
        timeout_count = 0

        # Debug aggregates for command/velocity magnitudes (avoid mean-cancellation of signed commands).
        debug_cmd_lin_xy_sum = 0.0
        debug_cmd_ang_z_sum = 0.0
        debug_act_lin_xy_sum = 0.0
        debug_act_ang_z_sum = 0.0
        debug_lin_xy_err_sum = 0.0
        debug_ang_z_err_sum = 0.0
        debug_metric_count = 0

        # 在每个环境中按固定步数运行当前策略并收集过渡样本。
        for _ in range(self.num_steps_per_env):
            if torch.isnan(self.current_obs).any() or torch.isinf(self.current_obs).any():
                print("[CPO][warn] current_obs contained NaN/Inf; sanitized before policy forward.")
                self.current_obs = torch.nan_to_num(self.current_obs, nan=0.0, posinf=0.0, neginf=0.0)

            with torch.no_grad():
                actions = self.actor_critic.act(self.current_obs)
                act_log_prob, values, cost_values, _ = self.actor_critic.evaluate(self.current_obs, actions)
                mu = self.actor_critic.safe_actor_mean(self.current_obs, context="rollout")
                sigma = (torch.nn.functional.softplus(self.actor_critic.std) + 1e-5).expand_as(mu)

            next_obs, rew, term, trunc, infos = self.env.step(actions)
            next_obs_tensor = extract_obs(next_obs, device=self.device)

            rew = to_device_tensor(rew, self.device).view(-1)
            costs = extract_costs(infos, self.cost_keys, self.device, self.num_envs)
            costs = costs.clamp(min=0.0, max=50.0)

            real_dones = to_device_tensor(term, self.device, dtype=torch.bool)
            time_outs = to_device_tensor(trunc, self.device, dtype=torch.bool)

            self.ep_rew_buf += rew
            self.ep_cost_buf += costs
            self.ep_len_buf += 1

            if torch.any(time_outs):
                with torch.no_grad():
                    _, next_vals, next_c_vals, _ = self.actor_critic.evaluate(next_obs_tensor, actions)
                bootstrap_mask = time_outs & (~real_dones)
                if torch.any(bootstrap_mask):
                    rew[bootstrap_mask] += self.gamma * next_vals[bootstrap_mask].view(-1)
                    costs[bootstrap_mask] += self.cost_gamma * next_c_vals[bootstrap_mask].view(-1)

            dones = torch.logical_or(real_dones, time_outs)
            real_done_count += int(real_dones.sum().item())
            timeout_count += int(time_outs.sum().item())

            done_idx = dones.nonzero(as_tuple=False).flatten()
            if len(done_idx) > 0:
                self.global_ep_stats["reward"] += self.ep_rew_buf[done_idx].sum().item()
                self.global_ep_stats["cost"] += self.ep_cost_buf[done_idx].sum().item()
                self.global_ep_stats["len"] += self.ep_len_buf[done_idx].sum().item()
                self.global_ep_count += len(done_idx)

                self.ep_rew_buf[done_idx] = 0
                self.ep_cost_buf[done_idx] = 0
                self.ep_len_buf[done_idx] = 0

            # 将当前过渡存入 rollout 缓冲区，后续用于优势与回报计算。
            self.storage.add_transitions(
                self.current_obs, actions, rew, costs, dones, values, cost_values, act_log_prob, mu, sigma
            )
            self.current_obs = next_obs_tensor
            last_actions = actions

            log_src = {}
            extracted = extract_log_dict(infos)
            if extracted:
                log_src.update(extracted)
            if isinstance(infos, dict):
                if isinstance(infos.get("log"), dict):
                    log_src.update(infos["log"])
                extras = infos.get("extras")
                if isinstance(extras, dict) and isinstance(extras.get("log"), dict):
                    log_src.update(extras["log"])

            # Extract signed command/velocity logs to compute magnitudes for debugging.
            try:
                cmd_x = log_src.get("command_lin_vel_x", None)
                cmd_y = log_src.get("command_lin_vel_y", None)
                cmd_ang = log_src.get("command_ang_vel_z", None)
                act_lin_xy = log_src.get("act_lin_vel_xy", None)
                act_ang = log_src.get("act_ang_vel_z", None)

                if cmd_x is not None and cmd_y is not None:
                    cmd_x_t = to_device_tensor(cmd_x, self.device).view(-1)[: self.num_envs]
                    cmd_y_t = to_device_tensor(cmd_y, self.device).view(-1)[: self.num_envs]
                    cmd_lin_xy_t = torch.norm(torch.stack([cmd_x_t, cmd_y_t], dim=-1), dim=-1)
                    debug_cmd_lin_xy_sum += cmd_lin_xy_t.mean().item()
                else:
                    cmd_lin_xy_t = None

                if cmd_ang is not None:
                    cmd_ang_t = to_device_tensor(cmd_ang, self.device).view(-1)[: self.num_envs]
                    debug_cmd_ang_z_sum += cmd_ang_t.abs().mean().item()
                else:
                    cmd_ang_t = None

                if act_lin_xy is not None:
                    act_lin_xy_t = to_device_tensor(act_lin_xy, self.device).view(-1)[: self.num_envs]
                    debug_act_lin_xy_sum += act_lin_xy_t.mean().item()
                else:
                    act_lin_xy_t = None

                if act_ang is not None:
                    act_ang_t = to_device_tensor(act_ang, self.device).view(-1)[: self.num_envs]
                    debug_act_ang_z_sum += act_ang_t.abs().mean().item()
                else:
                    act_ang_t = None

                if cmd_lin_xy_t is not None and act_lin_xy_t is not None:
                    debug_lin_xy_err_sum += (cmd_lin_xy_t - act_lin_xy_t).abs().mean().item()
                if cmd_ang_t is not None and act_ang_t is not None:
                    debug_ang_z_err_sum += (cmd_ang_t - act_ang_t).abs().mean().item()

                debug_metric_count += 1
            except Exception:
                pass

            # 汇总环境可能输出的数值指标（奖励或成本）。
            for k, v in log_src.items():
                k_clean = sanitize_term_key(k)
                k_lower = k_clean.lower()

                is_cost = k_lower in self._sanitized_cost_keys or "constraint" in k_lower or "cost" in k_lower
                is_reward_allowed = self._reward_log_allowlist is None or k_lower in self._reward_log_allowlist
                if not is_cost and not is_reward_allowed:
                    continue

                try:
                    val = to_device_tensor(v, self.device).mean().item()
                except Exception:
                    continue

                prefix = "constraint" if is_cost else "reward"
                log_sums[f"{prefix}_{k_clean}"] += val
                log_counts[f"{prefix}_{k_clean}"] += 1

        # Rollout 结束后，用当前策略估计最后状态值以便计算回报与优势。
        with torch.no_grad():
            if last_actions is None:
                last_actions = self.actor_critic.act(self.current_obs)
            _, last_val, last_cost_val, _ = self.actor_critic.evaluate(self.current_obs, last_actions)

        # 计算奖励与成本的折扣回报以及广义优势估计。
        self.storage.compute_returns(last_val, last_cost_val, self.gamma, self.lam, self.cost_gamma, self.cost_lam)

        est_ep_cost = None
        # 维护回合成本的指数移动平均值，为 CPO 约束提供目标参考。
        if self.global_ep_count > 0:
            est_ep_cost = self.global_ep_stats["cost"] / self.global_ep_count
            if self.cost_ema is None:
                self.cost_ema = est_ep_cost
            else:
                self.cost_ema = self.cost_ema_alpha * self.cost_ema + (1 - self.cost_ema_alpha) * est_ep_cost
        elif self.cost_ema is not None:
            est_ep_cost = self.cost_ema

        mean_cost = self.storage.costs.mean().item()
        max_step_cost = self.storage.costs.max().item()
        if max_step_cost > 100.0:
            print(f"[Warning] Large cost spike detected: {max_step_cost:.2f}")

        if est_ep_cost is None:
            update_res = {
                "value_loss": 0.0,
                "cost_value_loss": 0.0,
                "kl": 0.0,
                "cpo/nu": 0.0,
                "cpo/lam": 0.0,
                "cpo/step_frac": 0.0,
                "cpo/recovery": 0.0,
                "policy_surr_reward": 0.0,
                "policy_surr_cost": 0.0,
                "policy_entropy": 0.0,
                "policy_std_min": 0.0,
                "policy_std_max": 0.0,
                "Meta/update_skipped_no_complete_ep": 1.0,
            }
        else:
            update_res = self.update(est_ep_cost)
            update_res["Meta/update_skipped_no_complete_ep"] = 0.0

        fps = (self.num_envs * self.num_steps_per_env) / max(1e-8, (time.perf_counter() - t_start))

        results = {
            "Meta/mean_reward": self.storage.rewards.mean().item(),
            "Meta/mean_cost": mean_cost,
            "Meta/max_step_cost": max_step_cost,
            "Meta/estimated_ep_cost": est_ep_cost if est_ep_cost is not None else 0.0,
            "Meta/cost_per_step": mean_cost,
            "Meta/fps": fps,
            "Policy/mean_action_std": self.storage.sigma.mean().item(),
        }

        if debug_metric_count > 0:
            results["Debug/cmd_lin_xy"] = debug_cmd_lin_xy_sum / debug_metric_count
            results["Debug/cmd_ang_vel_z"] = debug_cmd_ang_z_sum / debug_metric_count
            results["Debug/act_lin_xy"] = debug_act_lin_xy_sum / debug_metric_count
            results["Debug/act_ang_vel_z"] = debug_act_ang_z_sum / debug_metric_count
            results["Debug/lin_xy_err"] = debug_lin_xy_err_sum / debug_metric_count
            results["Debug/ang_vel_z_err"] = debug_ang_z_err_sum / debug_metric_count
        else:
            results["Debug/cmd_lin_xy"] = 0.0
            results["Debug/cmd_ang_vel_z"] = 0.0
            results["Debug/act_lin_xy"] = 0.0
            results["Debug/act_ang_vel_z"] = 0.0
            results["Debug/lin_xy_err"] = 0.0
            results["Debug/ang_vel_z_err"] = 0.0
        results.update(update_res)

        for k, tot in log_sums.items():
            results[k] = tot / max(1, log_counts[k])

        try:
            flat_actions = self.storage.actions.flatten(0, 1)
            results["Debug/action_abs_mean"] = flat_actions.abs().mean().item()
            results["Debug/action_abs_max"] = flat_actions.abs().max().item()
        except Exception:
            results["Debug/action_abs_mean"] = 0.0
            results["Debug/action_abs_max"] = 0.0

        try:
            flat_obs = self.storage.observations.flatten(0, 1)
            results["Debug/obs_abs_mean"] = flat_obs.abs().mean().item()
            results["Debug/obs_abs_max"] = flat_obs.abs().max().item()
        except Exception:
            results["Debug/obs_abs_mean"] = 0.0
            results["Debug/obs_abs_max"] = 0.0

        if self.global_ep_count > 0:
            results["Episode/reward"] = self.global_ep_stats["reward"] / self.global_ep_count
            results["Episode/cost"] = self.global_ep_stats["cost"] / self.global_ep_count
            results["Episode/len"] = self.global_ep_stats["len"] / self.global_ep_count
            results["Meta/violations_per_ep"] = results["Episode/cost"]
            self.global_ep_stats.clear()
            self.global_ep_count = 0
        else:
            results["Meta/violations_per_ep"] = 0.0

        total_dones = max(1, real_done_count + timeout_count)
        results["Episode/real_done_rate"] = real_done_count / total_dones
        results["Episode/timeout_rate"] = timeout_count / total_dones

        self.storage.clear()
        return results

    def update(self, est_ep_cost: float) -> Dict[str, float]:
        """
        在满足最近一轮估计的成本约束的同时对策略进行优化。
        """
        self.actor_critic.eval()
        (
            obs,
            acts,
            rets,
            advs,
            vals,
            log_p,
            c_rets,
            c_advs,
            c_vals,
            mu,
            sigma,
        ) = self.storage.full_batch()

        # 防止无效观测（NaN/Inf）破坏优化过程。
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print("[CPO][warn] Invalid observations detected (NaN/Inf); skipping update to avoid contamination.")
            return {
                "value_loss": 0.0,
                "cost_value_loss": 0.0,
                "kl": 0.0,
                "cpo/nu": 0.0,
                "cpo/lam": 0.0,
                "cpo/step_frac": 0.0,
                "cpo/recovery": 0.0,
                "policy_surr_reward": 0.0,
                "policy_surr_cost": 0.0,
                "policy_entropy": 0.0,
                "policy_std_min": 0.0,
                "policy_std_max": 0.0,
            }

        obs, acts = obs.to(self.device), acts.to(self.device)
        rets, c_rets = rets.to(self.device).squeeze(-1), c_rets.to(self.device).squeeze(-1)
        advs, c_advs = advs.to(self.device).squeeze(-1), c_advs.to(self.device).squeeze(-1)
        log_p = log_p.to(self.device)
        mu = mu.to(self.device)
        sigma = torch.clamp(sigma.to(self.device), min=1e-5)

        v_loss_sum = 0.0
        c_loss_sum = 0.0
        num_updates = 0
        # 使用多个小批次训练值网络，以跟上策略快速变化带来的时序差异。
        for batch in self.storage.mini_batch_generator(
            num_mini_batches=self.value_mini_batches, num_epochs=self.value_epochs
        ):
            (
                b_obs,
                _b_actions,
                _b_values,
                _b_advs,
                b_returns,
                _b_log_probs,
                _b_mu,
                _b_sigma,
                b_cost_returns,
                _b_cost_advs,
                _b_cost_values,
            ) = batch

            b_obs = b_obs.to(self.device)
            b_returns = b_returns.to(self.device).squeeze(-1)
            b_cost_returns = b_cost_returns.to(self.device).squeeze(-1)

            cur_v, cur_c = self.actor_critic.get_values(b_obs)
            v_loss = (b_returns - cur_v.view(-1)).pow(2).mean()
            c_loss = (b_cost_returns - cur_c.view(-1)).pow(2).mean()

            self.critic_optimizer.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), max_norm=self.value_clip_grad_norm)
            self.critic_optimizer.step()

            self.cost_critic_optimizer.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.cost_critic.parameters(), max_norm=self.value_clip_grad_norm
            )
            self.cost_critic_optimizer.step()

            v_loss_sum += v_loss.item()
            c_loss_sum += c_loss.item()
            num_updates += 1

        # 在计算策略梯度前标准化优势，提升数值稳定性。
        if self.center_reward_adv_only:
            advs = advs - advs.mean()
        else:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        c_advs = c_advs - c_advs.mean()

        # 可选地对观测进行子采样，以降低 Fisher 向量乘积的估算成本。
        if self.fvp_sample_size is not None and self.fvp_sample_size < obs.shape[0]:
            fvp_idx = torch.randperm(obs.shape[0], device=self.device)[: self.fvp_sample_size]
            fvp_obs = obs[fvp_idx]
            fvp_mu = mu[fvp_idx]
            fvp_sigma = sigma[fvp_idx]
        else:
            fvp_obs, fvp_mu, fvp_sigma = obs, mu, sigma

        old_dist = torch.distributions.Normal(mu, sigma)
        cur_mu = self.actor_critic.safe_actor_mean(obs, context="update")
        cur_std = torch.nn.functional.softplus(self.actor_critic.std) + 1e-5
        cur_dist = torch.distributions.Normal(cur_mu, cur_std)

        # 计算 PPO 风格的策略 surrogate 与成本 surrogate。
        cur_log_p = self.actor_critic._squashed_log_prob(cur_dist, acts)
        ratio = torch.exp(cur_log_p - log_p)
        entropy = cur_dist.entropy().sum(-1)
        surr_reward = (ratio * advs).mean()
        surr_cost = (ratio * c_advs).mean()
        entropy_mean = entropy.mean()
        loss_pi = surr_reward + self.entropy_coef * entropy_mean
        loss_cost = surr_cost

        std_min = cur_std.min()
        std_max = cur_std.max()

        params = list(self.actor_critic.actor_mean.parameters()) + [self.actor_critic.std]
        grads_g = torch.autograd.grad(loss_pi, params, retain_graph=True)
        grads_b = torch.autograd.grad(loss_cost, params)
        flat_g = torch.cat([g.view(-1) for g in grads_g])
        flat_b = torch.cat([g.view(-1) for g in grads_b])

        # Fisher 向量乘积，用于近似自然梯度方向。
        def Fvp(v):
            fvp_old_dist = torch.distributions.Normal(fvp_mu, fvp_sigma)
            fvp_cur_mu = self.actor_critic.safe_actor_mean(fvp_obs, context="fvp")
            fvp_cur_std = torch.nn.functional.softplus(self.actor_critic.std) + 1e-5
            fvp_cur_dist = torch.distributions.Normal(fvp_cur_mu, fvp_cur_std)

            kl = torch.distributions.kl_divergence(fvp_old_dist, fvp_cur_dist).mean()
            g_kl = torch.autograd.grad(kl, params, create_graph=True)
            flat_g_kl = torch.cat([g.view(-1) for g in g_kl])
            kl_v = (flat_g_kl * v).sum()
            g_v = torch.autograd.grad(kl_v, params)
            return torch.cat([g.contiguous().view(-1) for g in g_v]) + v * self.damping_coeff

        # 通过共轭梯度求解奖励方向（q）与成本方向（r）对应的自然梯度。
        q = self.conjugate_gradient(Fvp, flat_g)
        r = self.conjugate_gradient(Fvp, flat_b)

        gHg = torch.dot(flat_g, q)
        bHr = torch.dot(flat_b, r)
        bHq = torch.dot(flat_b, q)

        # 计算估计回合成本与目标上限之间的差距。
        c_val = est_ep_cost - self.cost_limit

        # Solve CPO/TRPO quadratic subproblem (Achiam et al., 2017) using natural gradients.
        # Notation:
        #   B = g^T H^{-1} g = gHg
        #   A = b^T H^{-1} b = bHr
        #   C = b^T H^{-1} g = bHq
        #   c = J_c - d      = c_val
        # Trust-region: 0.5 * x^T H x <= target_kl  =>  2*target_kl is used in closed-form scaling.
        eps = 1e-8
        A = bHr
        B = gHg
        C = bHq
        E = 2.0 * self.target_kl
        is_recovery = False

        # If the cost direction is degenerate, fall back to unconstrained TRPO step.
        if torch.abs(A) < eps:
            nu = torch.tensor(0.0, device=self.device)
            lam_tensor = torch.sqrt(torch.clamp(B, min=eps) / max(E, eps))
            step_dir = (1.0 / (lam_tensor + eps)) * q
        else:
            # Case 1: constraint is satisfied and reward step also reduces cost -> TRPO step.
            if c_val <= 0.0 and C <= 0.0:
                nu = torch.tensor(0.0, device=self.device)
                lam_tensor = torch.sqrt(torch.clamp(B, min=eps) / max(E, eps))
                step_dir = (1.0 / (lam_tensor + eps)) * q
            else:
                # Case 2: use constrained solution.
                safe_denom = B - (C * C) / (A + eps)
                safe_denom = torch.clamp(safe_denom, min=eps)
                lam_tensor = torch.sqrt(safe_denom / max(E, eps))
                nu = torch.clamp((C + lam_tensor * c_val) / (A + eps), min=0.0)
                step_dir = (1.0 / (lam_tensor + eps)) * (q - nu * r)

                # If we're already violating the constraint and the constrained step is ill-conditioned,
                # fall back to a recovery step that reduces cost subject to the trust region.
                if c_val > 0.0 and not torch.isfinite(step_dir).all():
                    is_recovery = True
                    nu = torch.tensor(1.0, device=self.device)
                    lam_tensor = torch.sqrt(torch.clamp(A, min=eps) / max(E, eps))
                    step_dir = -(1.0 / (lam_tensor + eps)) * r

        lam = float(lam_tensor.item())

        if torch.isnan(step_dir).any() or torch.isinf(step_dir).any():
            print("[CPO] Update skipped due to NaN gradients")
            return {
                "value_loss": v_loss_sum / max(num_updates, 1),
                "cost_value_loss": c_loss_sum / max(num_updates, 1),
                "kl": 0.0,
                "cpo/nu": float(nu) if not isinstance(nu, float) else nu,
                "cpo/lam": lam,
                "cpo/step_frac": 0.0,
                "cpo/recovery": 1.0 if is_recovery else 0.0,
                "policy_surr_reward": float(surr_reward.item()),
                "policy_surr_cost": float(surr_cost.item()),
                "policy_entropy": float(entropy_mean.item()),
                "policy_std_min": float(std_min.item()),
                "policy_std_max": float(std_max.item()),
            }

        # 回退线搜索，确保在接受更新前满足 KL 与成本约束。
        search_res = self.line_search(step_dir, obs, acts, log_p, advs, c_advs, mu, sigma, c_val, is_recovery)

        return {
            "value_loss": v_loss_sum / max(num_updates, 1),
            "cost_value_loss": c_loss_sum / max(num_updates, 1),
            "kl": search_res.get("kl", 0.0),
            "cpo/nu": float(nu) if not isinstance(nu, float) else nu,
            "cpo/lam": lam,
            "cpo/step_frac": search_res.get("step_frac", 0.0),
            "cpo/recovery": 1.0 if is_recovery else 0.0,
            "policy_surr_reward": float(surr_reward.item()),
            "policy_surr_cost": float(surr_cost.item()),
            "policy_entropy": float(entropy_mean.item()),
            "policy_std_min": float(std_min.item()),
            "policy_std_max": float(std_max.item()),
        }

    def conjugate_gradient(self, fvp, b, nsteps: int = 10, tol: float = 1e-10):
        """
        使用仅需 Fisher 向量乘积的共轭梯度近似求解 H^{-1} b。
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)
        for _ in range(nsteps):
            Ap = fvp(p)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def line_search(self, step, obs, acts, old_log_p, adv, c_adv, mu, sigma, c_val, is_recovery=False):
        """
        使用回退线搜索在策略更新前验证奖励、KL 与成本约束。
        """
        if torch.isnan(step).any() or torch.isinf(step).any():
            print("[CPO][warn] line_search received invalid step; aborting update.")
            return {"kl": 0.0, "step_frac": 0.0}

        old_params = torch.cat([p.data.view(-1) for p in self._policy_params()])
        old_loss = -adv.mean()
        old_cost = c_adv.mean()
        step_frac = 1.0

        # 逐步缩小步长，直至满足约束或耗尽尝试次数。
        for _ in range(self.max_backtracks):
            self._set_policy_params(old_params + step_frac * step)
            with torch.no_grad():
                new_mu = self.actor_critic.safe_actor_mean(obs, context="line_search")
                new_std = torch.nn.functional.softplus(self.actor_critic.std) + 1e-5
                new_dist = torch.distributions.Normal(new_mu, new_std)
                new_log_p = self.actor_critic._squashed_log_prob(new_dist, acts)
                ratio = torch.exp(new_log_p - old_log_p)

                loss = -(ratio * adv).mean()
                c_surr = (ratio * c_adv).mean()
                kl = torch.distributions.kl_divergence(torch.distributions.Normal(mu, sigma), new_dist).mean()

            if torch.isnan(loss) or torch.isnan(kl) or torch.isnan(c_surr):
                step_frac *= 0.5
                continue

            reward_ok = loss <= old_loss + 1e-4
            kl_ok = kl <= self.target_kl * 1.5
            if is_recovery:
                cost_ok = c_surr < old_cost
                if cost_ok:
                    return {"kl": kl.item(), "step_frac": step_frac}
            else:
                if c_val > 0:
                    cost_ok = c_surr <= old_cost - c_val * 0.1
                else:
                    cost_ok = c_surr <= old_cost + max(abs(c_val), 1e-4) * 0.1

                if reward_ok and kl_ok and cost_ok:
                    return {"kl": kl.item(), "step_frac": step_frac}

            step_frac *= self.backtrack_coeff

        self._set_policy_params(old_params)
        return {"kl": 0.0, "step_frac": 0.0}

    def _policy_params(self):
        return list(self.actor_critic.actor_mean.parameters()) + [self.actor_critic.std]

    def _set_policy_params(self, flat_params):
        pointer = 0
        for param in self._policy_params():
            num_param = param.numel()
            param.data.copy_(flat_params[pointer : pointer + num_param].view(param.shape))
            pointer += num_param
