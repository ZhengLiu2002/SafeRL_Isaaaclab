import time
import torch
import torch.optim as optim

# Allow imports to work whether the package is imported or the script is run directly.
try:
    from .modules import ActorCritic
    from .storage import RolloutStorage
    from .utils.tensor_utils import extract_costs, extract_obs, to_device_tensor
except ImportError:
    from modules import ActorCritic
    from storage import RolloutStorage
    from utils.tensor_utils import extract_costs, extract_obs, to_device_tensor

class CPO:
    """
    Constrained Policy Optimization (CPO)
    Uses Conjugate Gradient for Trust Region and Line Search for Safety Constraint.
    """
    def __init__(self,
                 env,
                 actor_critic: ActorCritic,
                 device='cpu',
                 num_envs=1,
                 num_steps_per_env=24,
                 batch_size=None, # CPO uses full batch
                 gamma=0.99,
                 lam=0.95,
                 cost_gamma=0.99,
                 cost_lam=0.95,
                 cost_limit=25.0,
                 target_kl=0.01,
                 damping_coeff=0.1,
                 backtrack_coeff=0.8,
                 max_backtracks=10,
                 entropy_coef=0.0,
                 cost_keys=[],
                 storage_device=None,
                 fvp_sample_size=None
                 ):
        
        self.env = env
        self.actor_critic = actor_critic
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device) if storage_device else self.device
        self.num_envs = num_envs
        self.num_steps_per_env = num_steps_per_env
        self.gamma = gamma
        self.lam = lam
        self.cost_gamma = cost_gamma
        self.cost_lam = cost_lam
        self.cost_limit = cost_limit
        self.target_kl = target_kl
        self.damping_coeff = damping_coeff
        self.backtrack_coeff = backtrack_coeff
        self.max_backtracks = max_backtracks
        self.entropy_coef = entropy_coef
        self.cost_keys = cost_keys or []
        self.fvp_sample_size = fvp_sample_size

        self.device = torch.device(device)
        init_obs, _ = self.env.reset()
        self.current_obs = extract_obs(init_obs, device=self.device)
        self.num_envs = getattr(env, "num_envs", self.current_obs.shape[0] if self.current_obs.dim() > 1 else 1)

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
        
        # Optimizers for Value functions (Actor is updated manually via CPO step)
        self.critic_optimizer = optim.Adam(self.actor_critic.critic.parameters(), lr=1e-3)
        self.cost_critic_optimizer = optim.Adam(self.actor_critic.cost_critic.parameters(), lr=1e-3)
        
        torch.backends.cudnn.benchmark = True

        # Episode stats buffers
        self.ep_rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.ep_cost_buf = torch.zeros(self.num_envs, device=self.device)
        self.ep_len_buf = torch.zeros(self.num_envs, device=self.device)
        self.cur_ep_rew_total = 0
        self.cur_ep_cost_total = 0
        self.cur_ep_len_total = 0
        self.cur_ep_count = 0

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
        
        ep_len = self.env.max_episode_length if hasattr(self.env, 'max_episode_length') else 1000
        estimated_ep_cost = self.storage.costs.mean().item() * ep_len
        mean_reward = self.storage.rewards.mean().item()
        mean_cost = self.storage.costs.mean().item()

        loss_info = self.update(estimated_ep_cost)
        learning_time = time.perf_counter() - update_start
        self.storage.clear()

        loss_info["estimated_ep_cost"] = estimated_ep_cost
        loss_info["mean_reward"] = mean_reward
        loss_info["mean_cost"] = mean_cost
        loss_info["collection_time"] = collection_time
        loss_info["learning_time"] = learning_time
        loss_info["mean_action_std"] = float(self.storage.sigma.mean().item())
        loss_info["action_std_param"] = float(self.actor_critic.std.detach().mean().item())

        # Add averaged reward/cost term breakdowns
        for key, total in reward_term_sums.items():
            count = max(1, reward_term_counts.get(key, 1))
            name = key if key not in loss_info else f"{key}_term"
            loss_info[name] = float(total / count)
        for key, total in cost_term_sums.items():
            count = max(1, cost_term_counts.get(key, 1))
            name = key if key not in loss_info else f"{key}_term"
            loss_info[name] = float(total / count)

        if self.cur_ep_count > 0:
            loss_info["episode/reward"] = self.cur_ep_rew_total / self.cur_ep_count
            loss_info["episode/cost"] = self.cur_ep_cost_total / self.cur_ep_count
            loss_info["episode/length"] = self.cur_ep_len_total / self.cur_ep_count
            self.cur_ep_rew_total = 0
            self.cur_ep_cost_total = 0
            self.cur_ep_len_total = 0
            self.cur_ep_count = 0

        return loss_info

    def update(self, estimated_ep_cost):
        obs, actions, returns, advantages, values, log_probs, cost_returns, cost_advantages, cost_values, mu, sigma = self.storage.full_batch()

        obs = obs.to(self.device)
        actions = actions.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        values = values.to(self.device)
        log_probs = log_probs.to(self.device)
        cost_returns = cost_returns.to(self.device)
        cost_advantages = cost_advantages.to(self.device)
        cost_values = cost_values.to(self.device)
        mu = mu.to(self.device)
        sigma = sigma.to(self.device)

        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        for _ in range(10):
            curr_values, curr_cost_values = self.actor_critic.get_values(obs)
            value_loss = (returns - curr_values.view(-1)).pow(2).mean()
            cost_value_loss = (cost_returns - curr_cost_values.view(-1)).pow(2).mean()

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()
            
            self.cost_critic_optimizer.zero_grad()
            cost_value_loss.backward()
            self.cost_critic_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)

        old_dist = torch.distributions.Normal(mu, sigma)
        new_mean = self.actor_critic.actor_mean(obs)
        new_dist = torch.distributions.Normal(new_mean, self.actor_critic.std)
        curr_log_probs = new_dist.log_prob(actions).sum(dim=-1)

        ratio = torch.exp(curr_log_probs - log_probs)
        loss_pi = (ratio * advantages).mean()
        loss_cost = (ratio * cost_advantages).mean()

        policy_params = self._policy_params()
        grads_g = torch.autograd.grad(loss_pi, policy_params, retain_graph=True)
        grads_b = torch.autograd.grad(loss_cost, policy_params, retain_graph=True)
        flat_g = torch.cat([g.view(-1) for g in grads_g])
        flat_b = torch.cat([g.view(-1) for g in grads_b])

        def Fvp(v):
            if self.fvp_sample_size and self.fvp_sample_size < obs.shape[0]:
                indices = torch.randperm(obs.shape[0], device=self.device)[:self.fvp_sample_size]
                sample_obs = obs[indices]
                sample_mu = mu[indices]
                sample_sigma = sigma[indices]
            else:
                sample_obs = obs
                sample_mu = mu
                sample_sigma = sigma

            # Re-compute distribution on sampled data
            curr_mean = self.actor_critic.actor_mean(sample_obs)
            curr_dist = torch.distributions.Normal(curr_mean, self.actor_critic.std)
            old_dist_sample = torch.distributions.Normal(sample_mu, sample_sigma)

            kl = torch.distributions.kl_divergence(old_dist_sample, curr_dist).mean()
            
            grads = torch.autograd.grad(kl, policy_params, create_graph=True)
            flat_grad_kl = torch.cat([g.view(-1) for g in grads])
            kl_v = (flat_grad_kl * v).sum()
            grads_v = torch.autograd.grad(kl_v, policy_params, retain_graph=True)
            flat_grads_v = torch.cat([g.contiguous().view(-1) for g in grads_v])
            return flat_grads_v + v * self.damping_coeff

        q = self.conjugate_gradient(Fvp, flat_g)
        r = self.conjugate_gradient(Fvp, flat_b)

        gHg = torch.dot(flat_g, q)
        bHr = torch.dot(flat_b, r)
        bHq = torch.dot(flat_b, q)
        c = estimated_ep_cost - self.cost_limit

        if c < 0 and bHq <= 0:
            lam = torch.sqrt(2 * self.target_kl / (gHg + 1e-8))
            nu = 0.0
        else:
            trust_denom = gHg - (bHq * bHq) / (bHr + 1e-8)
            trust_denom = torch.clamp(trust_denom, min=1e-8)
            lam = torch.sqrt(2 * self.target_kl / trust_denom)
            nu = torch.clamp((lam * bHq + c) / (bHr + 1e-8), min=0.0)

        lam = torch.clamp(lam, min=1e-6, max=1e6)
        step_dir = (1.0 / lam) * (q - nu * r)

        ls_info = self.line_search(step_dir, obs, actions, log_probs, advantages, cost_advantages, mu, sigma, c)

        return {
            "value_loss": mean_value_loss / 10.0,
            "cost_value_loss": mean_cost_value_loss / 10.0,
            "kl": ls_info.get("kl", 0.0),
            "cost_surrogate": ls_info.get("cost_surrogate", 0.0),
            "reward_surrogate": loss_pi.item(),
            "entropy": new_dist.entropy().sum(dim=-1).mean().item(),
            "approx_kl": torch.distributions.kl_divergence(old_dist, new_dist).mean().item(),
        }

    def conjugate_gradient(self, fvp_func, b, nsteps=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)
        for _ in range(nsteps):
            Ap = fvp_func(p)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def _policy_params(self):
        return list(self.actor_critic.actor_mean.parameters()) + [self.actor_critic.std]

    def _flatten_policy_params(self):
        return torch.cat([p.data.view(-1) for p in self._policy_params()])

    def _set_policy_params(self, flat_params):
        pointer = 0
        for param in self._policy_params():
            num_param = param.numel()
            param.data.copy_(flat_params[pointer:pointer + num_param].view(param.shape))
            pointer += num_param

    def line_search(self, step, obs, actions, old_log_probs, advantages, cost_advantages, mu, sigma, c):
        step_frac = 1.0
        old_params = self._flatten_policy_params()
        old_loss = -(advantages).mean()
        old_cost = cost_advantages.mean()
        old_dist = torch.distributions.Normal(mu, sigma)

        for _ in range(self.max_backtracks):
            new_params = old_params + step_frac * step
            self._set_policy_params(new_params)

            with torch.no_grad():
                new_log_probs, _, _, _ = self.actor_critic.evaluate(obs, actions)
                ratio = torch.exp(new_log_probs - old_log_probs)
                reward_loss = -(ratio * advantages).mean()
                cost_surrogate = (ratio * cost_advantages).mean()

                new_mean = self.actor_critic.actor_mean(obs)
                new_dist = torch.distributions.Normal(new_mean, self.actor_critic.std)
                kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()

            target_cost_change = -c if c > 0 else 0.0
            cost_ok = cost_surrogate <= target_cost_change + 1e-6
            if reward_loss <= old_loss and kl <= self.target_kl * 1.5 and cost_ok:
                return {"kl": kl.item(), "cost_surrogate": cost_surrogate.item(), "reward_loss": reward_loss.item()}

            step_frac *= self.backtrack_coeff

        self._set_policy_params(old_params)
        return {"kl": 0.0, "cost_surrogate": old_cost.item(), "reward_loss": old_loss.item()}
