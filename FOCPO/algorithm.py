import torch
from CPO.algorithm import CPO

class FOCPO(CPO):
    """
    First-Order Constrained Policy Optimization (FOCPO)
    Essentially CPO but replaces the second-order KL constraint (Fisher Information Matrix) 
    with a first-order approximation (trust region via clipping or penalty), 
    or simply using First Order approximation for the projection step.
    
    However, the standard name "FOCPO" often implies a two-step update similar to PCPO 
    but using first-order methods (like Adam/SGD) for the update direction instead of conjugate gradient.
    
    Implementation here follows a simplified approach often cited as FOCOPS/FOCPO:
    Solve the constrained optimization problem in the tangent space (linear approximations)
    and convert the result into a policy gradient update.
    
    Maximize: g^T (theta - theta_old)
    Subject to: KL(theta || theta_old) <= delta
                c + b^T (theta - theta_old) <= 0
                
    Solution is found via Lagrange duality, similar to CPO but the update is applied 
    directly without CG if we assume identity Hessian (First Order), or we use CG but 
    simplified. 
    
    Given user request for FOCPO alongside PCPO/CPO, I will implement it as:
    Two-phase update: 
    1. Find optimal step direction `d` using Lagrange multipliers on the linear approximations.
    2. Update policy.
    """
    def __init__(self, env, actor_critic, **kwargs):
        super().__init__(env, actor_critic, **kwargs)
        
    def update(self, estimated_ep_cost):
        obs, actions, returns, advantages, values, log_probs, cost_returns, cost_advantages, cost_values, mu, sigma = self.storage.full_batch()

        obs = obs.to(self.device)
        actions = actions.to(self.device)
        returns = returns.to(self.device).squeeze(-1)
        advantages = advantages.to(self.device).squeeze(-1)
        values = values.to(self.device).squeeze(-1)
        log_probs = log_probs.to(self.device)
        cost_returns = cost_returns.to(self.device).squeeze(-1)
        cost_advantages = cost_advantages.to(self.device).squeeze(-1)
        cost_values = cost_values.to(self.device).squeeze(-1)
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

        new_mean = self.actor_critic.actor_mean(obs)
        new_std = torch.nn.functional.softplus(self.actor_critic.std) + 1e-5
        new_dist = torch.distributions.Normal(new_mean, new_std)
        curr_log_probs = new_dist.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(curr_log_probs - log_probs)

        loss_pi = (ratio * advantages).mean()
        loss_cost = (ratio * cost_advantages).mean()

        policy_params = self._policy_params()
        grads_g = torch.autograd.grad(loss_pi, policy_params, retain_graph=True)
        flat_g = torch.cat([g.view(-1) for g in grads_g])

        grads_b = torch.autograd.grad(loss_cost, policy_params, retain_graph=True)
        flat_b = torch.cat([g.view(-1) for g in grads_b])

        c = estimated_ep_cost - self.cost_limit
        tr_radius = torch.sqrt(torch.tensor(2 * self.target_kl, device=self.device))

        g_norm = torch.norm(flat_g)
        step_reward = tr_radius * flat_g / (g_norm + 1e-8)

        cost_change = torch.dot(flat_b, step_reward)
        b_norm_sq = torch.dot(flat_b, flat_b)

        if c + cost_change > 0 and b_norm_sq > 0:
            lam = (c + cost_change) / (b_norm_sq + 1e-8)
            step_dir = step_reward - lam * flat_b
        else:
            step_dir = step_reward

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
