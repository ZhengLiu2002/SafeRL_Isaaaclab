import contextlib
import time
import torch
import torch.nn as nn
import torch.optim as optim

from .modules import ActorCritic
from .storage import RolloutStorage
from .utils.tensor_utils import extract_costs, extract_obs, to_device_tensor

class PPOLagrange:
    def __init__(self,
                 env,
                 actor_critic: ActorCritic,
                 device='cpu',
                 num_envs=1,
                 num_steps_per_env=24,
                 batch_size=128,
                 num_epochs=5,
                 learning_rate=1e-3,
                 gamma=0.99,
                 lam=0.95,
                 cost_gamma=0.99,
                 cost_lam=0.95,
                 clip_param=0.2,
                 entropy_coef=0.01,
                 value_loss_coef=1.0,
                 cost_value_loss_coef=1.0,
                 max_grad_norm=1.0,
                 cost_limit=25.0,
                 pid_kp=0.1,
                 pid_ki=0.01,
                 pid_kd=0.01,
                 cost_keys=[],
                 storage_device=None,
                 use_amp=False
                 ):
        
        self.env = env
        self.actor_critic = actor_critic
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device) if storage_device else self.device
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.num_envs = getattr(env, "num_envs", num_envs)
        self.num_steps_per_env = num_steps_per_env
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.gamma = gamma
        self.lam = lam
        self.cost_gamma = cost_gamma
        self.cost_lam = cost_lam
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.cost_value_loss_coef = cost_value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.cost_keys = cost_keys or []
        
        # PID Lagrangian
        self.cost_limit = cost_limit
        self.pid_kp = pid_kp
        self.pid_ki = pid_ki
        self.pid_kd = pid_kd
        self.pid_i = 0
        self.cost_penalty = 0.0

        # Init first obs before creating storage
        init_obs, _ = self.env.reset()
        self.current_obs = extract_obs(init_obs, device=self.device)
        self.num_envs = self.current_obs.shape[0] if self.current_obs.dim() > 1 else 1

        obs_shape = self.current_obs.shape[1:] if self.current_obs.dim() > 1 else self.current_obs.shape
        
        # Correctly handle action shape for vectorized envs
        if hasattr(env, "single_action_space"):
             action_shape = env.single_action_space.shape
        elif hasattr(env, "action_space") and len(env.action_space.shape) > 1 and env.action_space.shape[0] == self.num_envs:
             # Batched action space (num_envs, action_dim)
             action_shape = env.action_space.shape[1:]
        else:
             action_shape = env.action_space.shape

        pin_memory = self.storage_device.type == "cpu" and self.device.type == "cuda"
        self.storage = RolloutStorage(self.num_envs, num_steps_per_env, obs_shape, action_shape, self.storage_device, pin_memory=pin_memory)
        
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.scaler = self._make_grad_scaler(self.use_amp)
        
        # Enable faster cuDNN kernels when available
        torch.backends.cudnn.benchmark = True

        # Episode stats buffers
        self.ep_rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.ep_cost_buf = torch.zeros(self.num_envs, device=self.device)
        self.ep_len_buf = torch.zeros(self.num_envs, device=self.device)
        self.cur_ep_rew_total = 0
        self.cur_ep_cost_total = 0
        self.cur_ep_len_total = 0
        self.cur_ep_count = 0
        
        # Running Mean/Std for Rewards and Costs (Normalization)
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4
        self.cost_mean = 0.0
        self.cost_var = 1.0
        self.cost_count = 1e-4

    def _make_grad_scaler(self, enabled: bool):
        if not enabled:
            return torch.cuda.amp.GradScaler(enabled=False)
        try:
            from torch.amp import GradScaler as AmpGradScaler  # type: ignore
            return AmpGradScaler("cuda", enabled=True)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=True)

    def _autocast(self):
        if not self.use_amp:
            return contextlib.nullcontext()
        try:
            from torch.amp import autocast as amp_autocast  # type: ignore
            return amp_autocast("cuda")
        except Exception:
            return torch.cuda.amp.autocast()

    def step(self):
        self.actor_critic.eval()
        last_actions = None
        rollout_start = time.perf_counter()
        reward_term_sums = {}
        reward_term_counts = {}
        cost_term_sums = {}
        cost_term_counts = {}

        def _sanitize_term_key(key: str) -> str:
            name = str(key)
            if "/" in name:
                name = name.split("/")[-1]
            name = name.replace(" ", "_")
            return name
        for _ in range(self.num_steps_per_env):
            with torch.no_grad():
                actions = self.actor_critic.act(self.current_obs)
                actions_log_prob, values, cost_values, _ = self.actor_critic.evaluate(self.current_obs, actions)
                mu = self.actor_critic.actor_mean(self.current_obs)
                sigma = self.actor_critic.std.expand_as(mu)

            next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
            dones = torch.logical_or(
                to_device_tensor(terminated, self.device, dtype=torch.bool),
                to_device_tensor(truncated, self.device, dtype=torch.bool),
            )

            rewards = to_device_tensor(rewards, device=self.device).view(-1)
            costs = extract_costs(infos, self.cost_keys, self.device, self.num_envs)
            next_obs = extract_obs(next_obs, device=self.device)

            # Update episode stats
            self.ep_rew_buf += rewards
            self.ep_cost_buf += costs
            self.ep_len_buf += 1

            # Normalize rewards for training stability (simple running mean/std)
            # We do this manually here or rely on env wrapper? 
            # Usually RSL-RL does not normalize rewards by default, but scales them.
            # Given the user's issue with small/negative rewards, let's check if we need scaling.
            # For now, we keep raw rewards for storage to compute returns correctly.

            env_done_indices = dones.nonzero(as_tuple=False).flatten()
            if len(env_done_indices) > 0:
                 self.cur_ep_rew_total += self.ep_rew_buf[env_done_indices].sum().item()
                 self.cur_ep_cost_total += self.ep_cost_buf[env_done_indices].sum().item()
                 self.cur_ep_len_total += self.ep_len_buf[env_done_indices].sum().item()
                 self.cur_ep_count += len(env_done_indices)

                 self.ep_rew_buf[env_done_indices] = 0
                 self.ep_cost_buf[env_done_indices] = 0
                 self.ep_len_buf[env_done_indices] = 0

            self.storage.add_transitions(self.current_obs, actions, rewards, costs, dones, values, cost_values, actions_log_prob, mu, sigma)

            self.current_obs = next_obs
            last_actions = actions

            # Collect per-step reward/cost terms from env logs
            log_src = None
            if isinstance(infos, dict):
                if "log" in infos:
                    log_src = infos["log"]
                elif "extras" in infos and isinstance(infos["extras"], dict) and "log" in infos["extras"]:
                    log_src = infos["extras"]["log"]
            if log_src is not None:
                for key, val in log_src.items():
                    key_lower = str(key).lower()
                    if not any(token in key_lower for token in ("reward", "cost", "penalty")):
                        continue
                    try:
                        tensor_val = to_device_tensor(val, self.device)
                    except Exception:
                        continue
                    if tensor_val.numel() == 0:
                        continue
                    total = tensor_val.sum().item()
                    count = tensor_val.numel()
                    name = _sanitize_term_key(key)
                    if "reward" in key_lower:
                        reward_term_sums[name] = reward_term_sums.get(name, 0.0) + total
                        reward_term_counts[name] = reward_term_counts.get(name, 0) + count
                    else:
                        cost_term_sums[name] = cost_term_sums.get(name, 0.0) + total
                        cost_term_counts[name] = cost_term_counts.get(name, 0) + count

        collection_time = time.perf_counter() - rollout_start
        update_start = time.perf_counter()
        with torch.no_grad():
            if last_actions is None:
                last_actions = self.actor_critic.act(self.current_obs)
            _, last_values, last_cost_values, _ = self.actor_critic.evaluate(self.current_obs, last_actions)

        self.storage.compute_returns(last_values, last_cost_values, self.gamma, self.lam, self.cost_gamma, self.cost_lam)

        avg_cost = self.storage.costs.mean().item()
        ep_len = self.env.max_episode_length if hasattr(self.env, "max_episode_length") else 1000
        estimated_ep_cost = avg_cost * ep_len

        delta = estimated_ep_cost - self.cost_limit
        self.pid_i = max(0.0, self.pid_i + delta * self.pid_ki)
        self.cost_penalty = max(0.0, delta * self.pid_kp + self.pid_i)

        self.actor_critic.train()
        rollout_reward = self.storage.rewards.mean().item()
        rollout_cost = self.storage.costs.mean().item()
        loss_info = self.update(estimated_ep_cost)
        learning_time = time.perf_counter() - update_start

        self.storage.clear()

        loss_info.update({
            "mean_reward": rollout_reward,
            "mean_cost": rollout_cost,
            "penalty": self.cost_penalty,
            "collection_time": collection_time,
            "learning_time": learning_time,
            "mean_action_std": float(self.storage.sigma.mean().item()),
            "action_std_param": float(self.actor_critic.std.detach().mean().item()),
        })

        # Add averaged reward/cost term breakdowns
        for key, total in reward_term_sums.items():
            count = max(1, reward_term_counts.get(key, 1))
            # Always append _term to identify these as breakdown metrics
            loss_info[f"{key}_term"] = float(total / count)
        for key, total in cost_term_sums.items():
            count = max(1, cost_term_counts.get(key, 1))
            loss_info[f"{key}_term"] = float(total / count)

        if self.cur_ep_count > 0:
             loss_info["episode/reward"] = self.cur_ep_rew_total / self.cur_ep_count
             loss_info["episode/cost"] = self.cur_ep_cost_total / self.cur_ep_count
             loss_info["episode/length"] = self.cur_ep_len_total / self.cur_ep_count
             # Reset accumulators
             self.cur_ep_rew_total = 0
             self.cur_ep_cost_total = 0
             self.cur_ep_len_total = 0
             self.cur_ep_count = 0
        
        return loss_info

    def update(self, estimated_ep_cost):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_cost_loss = 0
        mean_entropy = 0
        mean_kl = 0
        mean_clip_frac = 0
        total_samples = 0
        
        # Mini-batch update
        num_updates = 0
        total_samples_size = self.num_envs * self.num_steps_per_env
        # Calculate number of minibatches based on configured batch_size
        if self.batch_size > 0:
            num_mini_batches = max(1, int(total_samples_size // self.batch_size))
        else:
            num_mini_batches = 4

        for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
            old_actions_log_prob_batch, _, _, \
            cost_returns_batch, cost_advantages_batch, cost_values_batch in self.storage.mini_batch_generator(num_mini_batches=num_mini_batches, num_epochs=self.num_epochs):

            obs_batch = obs_batch.to(self.device, non_blocking=True)
            actions_batch = actions_batch.to(self.device, non_blocking=True)
            advantages_batch = advantages_batch.to(self.device, non_blocking=True)
            returns_batch = returns_batch.to(self.device, non_blocking=True)
            old_actions_log_prob_batch = old_actions_log_prob_batch.to(self.device, non_blocking=True)
            cost_returns_batch = cost_returns_batch.to(self.device, non_blocking=True)
            cost_advantages_batch = cost_advantages_batch.to(self.device, non_blocking=True)
            cost_values_batch = cost_values_batch.to(self.device, non_blocking=True)
            
            with self._autocast():
                action_log_probs, values, cost_values, dist_entropy = self.actor_critic.evaluate(obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_actions_log_prob_batch)

                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                # Do not normalize cost advantages! This destroys the physical meaning of the Lagrange multiplier (penalty).
                # cost_advantages_batch = (cost_advantages_batch - cost_advantages_batch.mean()) / (cost_advantages_batch.std() + 1e-8)

                combined_advantage = advantages_batch - self.cost_penalty * cost_advantages_batch
                
                # Fix broadcasting error: ratio is (N,), combined_advantage is (N, 1)
                if combined_advantage.dim() > 1:
                    combined_advantage = combined_advantage.squeeze(-1)

                # Clip advantages to avoid explosion
                combined_advantage = torch.clamp(combined_advantage, -10.0, 10.0)

                surr1_c = ratio * combined_advantage
                surr2_c = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * combined_advantage
                policy_loss_combined = -torch.min(surr1_c, surr2_c).mean()

                value_loss = (returns_batch - values).pow(2).mean()
                cost_value_loss = (cost_returns_batch - cost_values).pow(2).mean()

                loss = policy_loss_combined + \
                       self.value_loss_coef * value_loss + \
                       self.cost_value_loss_coef * cost_value_loss - \
                       self.entropy_coef * dist_entropy.mean()
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += policy_loss_combined.item()
            mean_cost_loss += cost_value_loss.item()
            mean_entropy += dist_entropy.mean().item()
            mean_kl += (old_actions_log_prob_batch - action_log_probs.detach()).mean().item()
            mean_clip_frac += (torch.abs(ratio - 1.0) > self.clip_param).float().mean().item()
            total_samples += advantages_batch.shape[0]
            num_updates += 1
        
        return {
            "loss": mean_surrogate_loss / num_updates if num_updates else 0.0,
            "value_loss": mean_value_loss / num_updates if num_updates else 0.0,
            "cost_value_loss": mean_cost_loss / num_updates if num_updates else 0.0,
            "penalty": self.cost_penalty,
            "estimated_ep_cost": estimated_ep_cost,
            "entropy": mean_entropy / num_updates if num_updates else 0.0,
            "approx_kl": mean_kl / num_updates if num_updates else 0.0,
            "clip_frac": mean_clip_frac / num_updates if num_updates else 0.0,
            "batch_samples": total_samples
        }
