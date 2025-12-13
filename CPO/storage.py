import torch

class RolloutStorage:
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, actions_shape, device='cpu', pin_memory=False):
        self.device = torch.device(device)
        self.pin_memory = pin_memory and self.device.type == "cpu"
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env

        zeros_kwargs = {"device": self.device}
        if self.pin_memory:
            zeros_kwargs["pin_memory"] = True

        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, **zeros_kwargs)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, **zeros_kwargs)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, **zeros_kwargs)
        self.costs = torch.zeros(num_transitions_per_env, num_envs, 1, **zeros_kwargs)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, **zeros_kwargs)
        
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, **zeros_kwargs)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, **zeros_kwargs)
        self.cost_values = torch.zeros(num_transitions_per_env, num_envs, 1, **zeros_kwargs)
        
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, **zeros_kwargs)
        self.cost_returns = torch.zeros(num_transitions_per_env, num_envs, 1, **zeros_kwargs)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, **zeros_kwargs)
        self.cost_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, **zeros_kwargs)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, **zeros_kwargs)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, **zeros_kwargs)

        self.step = 0

    def add_transitions(self, observations, actions, rewards, costs, dones, values, cost_values, actions_log_prob, mu, sigma):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        observations = observations.to(self.device, non_blocking=self.pin_memory)
        actions = actions.to(self.device, non_blocking=self.pin_memory)
        rewards = rewards.to(self.device, non_blocking=self.pin_memory)
        costs = costs.to(self.device, non_blocking=self.pin_memory)
        dones = dones.to(self.device, non_blocking=self.pin_memory)
        values = values.to(self.device, non_blocking=self.pin_memory)
        cost_values = cost_values.to(self.device, non_blocking=self.pin_memory)
        actions_log_prob = actions_log_prob.to(self.device, non_blocking=self.pin_memory)
        mu = mu.to(self.device, non_blocking=self.pin_memory)
        sigma = sigma.to(self.device, non_blocking=self.pin_memory)

        self.observations[self.step].copy_(observations)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.costs[self.step].copy_(costs.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.cost_values[self.step].copy_(cost_values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, last_cost_values, gamma, lam, cost_gamma, cost_lam):
        last_values = last_values.to(self.device)
        last_cost_values = last_cost_values.to(self.device)

        values_ext = torch.cat([self.values, last_values.unsqueeze(0)], dim=0)
        cost_values_ext = torch.cat([self.cost_values, last_cost_values.unsqueeze(0)], dim=0)

        gae = torch.zeros_like(last_values)
        cost_gae = torch.zeros_like(last_cost_values)

        for t in reversed(range(self.num_transitions_per_env)):
            # IMPORTANT:
            # `self.dones[t]` indicates whether the transition at time t ended an episode.
            # Since IsaacLab auto-resets terminated envs inside `env.step`, `obs[t+1]` can be
            # the first state of the *next* episode. Therefore we must mask bootstrapping with
            # `done_t` (not `done_{t+1}`), otherwise returns/advantages leak across episodes.
            next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * values_ext[t + 1] * next_non_terminal - values_ext[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[t] = gae

            delta_cost = self.costs[t] + cost_gamma * cost_values_ext[t + 1] * next_non_terminal - cost_values_ext[t]
            cost_gae = delta_cost + cost_gamma * cost_lam * next_non_terminal * cost_gae
            self.cost_advantages[t] = cost_gae

        self.returns = self.advantages + self.values
        self.cost_returns = self.cost_advantages + self.cost_values

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = max(1, batch_size // num_mini_batches)
        
        flat_obs = self.observations.flatten(0, 1)
        flat_actions = self.actions.flatten(0, 1)
        flat_values = self.values.flatten(0, 1)
        flat_returns = self.returns.flatten(0, 1)
        flat_log_probs = self.actions_log_prob.flatten(0, 1)
        flat_advantages = self.advantages.flatten(0, 1)
        flat_mu = self.mu.flatten(0, 1)
        flat_sigma = self.sigma.flatten(0, 1)
        flat_cost_returns = self.cost_returns.flatten(0, 1)
        flat_cost_advantages = self.cost_advantages.flatten(0, 1)
        flat_cost_values = self.cost_values.flatten(0, 1)

        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                batch_idx = indices[start:end]

                yield (
                    flat_obs[batch_idx],
                    flat_actions[batch_idx],
                    flat_values[batch_idx],
                    flat_advantages[batch_idx],
                    flat_returns[batch_idx],
                    flat_log_probs[batch_idx],
                    flat_mu[batch_idx],
                    flat_sigma[batch_idx],
                    flat_cost_returns[batch_idx],
                    flat_cost_advantages[batch_idx],
                    flat_cost_values[batch_idx],
                )

    def full_batch(self):
        obs = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        values = self.values.flatten(0, 1)
        log_probs = self.actions_log_prob.flatten(0, 1)
        cost_returns = self.cost_returns.flatten(0, 1)
        cost_advantages = self.cost_advantages.flatten(0, 1)
        cost_values = self.cost_values.flatten(0, 1)
        mu = self.mu.flatten(0, 1)
        sigma = self.sigma.flatten(0, 1)
        
        return obs, actions, returns, advantages, values, log_probs, cost_returns, cost_advantages, cost_values, mu, sigma
