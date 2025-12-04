import torch
from CPO.algorithm import CPO

class PCPO(CPO):
    """
    Projection-Based CPO (PCPO)
    First updates the policy to maximize reward (using TRPO step), 
    then projects it back to the safe set if constraints are violated.
    """
    def __init__(self, env, actor_critic, **kwargs):
        super().__init__(env, actor_critic, **kwargs)
        
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
        flat_g = torch.cat([g.view(-1) for g in grads_g])

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

        # Reward step (TRPO-style)
        q = self.conjugate_gradient(Fvp, flat_g)
        gHg = torch.dot(flat_g, q)
        trpo_step = torch.sqrt(2 * self.target_kl / (gHg + 1e-8)) * q

        # Cost projection
        grads_b = torch.autograd.grad(loss_cost, policy_params, retain_graph=True)
        flat_b = torch.cat([g.view(-1) for g in grads_b])
        r = self.conjugate_gradient(Fvp, flat_b)

        c = estimated_ep_cost - self.cost_limit
        trpo_constraint_val = c + torch.dot(flat_b, trpo_step)

        if trpo_constraint_val > 0:
            lam_proj = trpo_constraint_val / (torch.dot(flat_b, r) + 1e-8)
            step_dir = trpo_step - lam_proj * r
        else:
            step_dir = trpo_step

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
