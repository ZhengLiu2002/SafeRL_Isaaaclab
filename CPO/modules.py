import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_actions, actor_hidden_dims=[256, 256, 256], critic_hidden_dims=[256, 256, 256], activation="elu", init_noise_std=0.5):
        super(ActorCritic, self).__init__()
        
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ELU()

        actor_layers = []
        actor_layers.append(nn.Linear(num_obs, actor_hidden_dims[0]))
        nn.init.orthogonal_(actor_layers[-1].weight, gain=np.sqrt(2))
        nn.init.constant_(actor_layers[-1].bias, 0.0)
        actor_layers.append(self.activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                layer = nn.Linear(actor_hidden_dims[l], num_actions)
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0.0)
                actor_layers.append(layer)
            else:
                layer = nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
                actor_layers.append(layer)
                actor_layers.append(self.activation)
        self.actor_mean = nn.Sequential(*actor_layers)
        
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        critic_layers = []
        critic_layers.append(nn.Linear(num_obs, critic_hidden_dims[0]))
        nn.init.orthogonal_(critic_layers[-1].weight, gain=np.sqrt(2))
        nn.init.constant_(critic_layers[-1].bias, 0.0)
        critic_layers.append(self.activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                layer = nn.Linear(critic_hidden_dims[l], 1)
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
                critic_layers.append(layer)
            else:
                layer = nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1])
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
                critic_layers.append(layer)
                critic_layers.append(self.activation)
        self.critic = nn.Sequential(*critic_layers)

        cost_critic_layers = []
        cost_critic_layers.append(nn.Linear(num_obs, critic_hidden_dims[0]))
        nn.init.orthogonal_(cost_critic_layers[-1].weight, gain=np.sqrt(2))
        nn.init.constant_(cost_critic_layers[-1].bias, 0.0)
        cost_critic_layers.append(self.activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                layer = nn.Linear(critic_hidden_dims[l], 1)
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
                cost_critic_layers.append(layer)
            else:
                layer = nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1])
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
                cost_critic_layers.append(layer)
                cost_critic_layers.append(self.activation)
        self.cost_critic = nn.Sequential(*cost_critic_layers)

    def forward(self):
        raise NotImplementedError

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        # Numerically stable inverse tanh.
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _action_dist(self, observations: torch.Tensor) -> Normal:
        mean = self.safe_actor_mean(observations, context="dist")
        std = nn.functional.softplus(self.std) + 1e-5
        return Normal(mean, std)

    def _squashed_log_prob(self, dist: Normal, actions: torch.Tensor) -> torch.Tensor:
        # actions are assumed to be in (-1, 1).
        eps = 1e-6
        clipped = torch.clamp(actions, -1.0 + eps, 1.0 - eps)
        pre_tanh = self._atanh(clipped)
        log_prob = dist.log_prob(pre_tanh).sum(dim=-1)
        log_prob -= torch.log(1.0 - clipped.pow(2) + eps).sum(dim=-1)
        return log_prob

    def _sanitize_mean(self, mean: torch.Tensor, context: str = "act") -> torch.Tensor:
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.nan_to_num(mean, nan=0.0, posinf=100.0, neginf=-100.0)
            if torch.isnan(mean).any() or torch.isinf(mean).any():
                raise ValueError(f"Actor mean contains NaN/Inf during {context} even after sanitization.")
            print(f"[Actor][warn] Actor mean contained NaN/Inf during {context}; sanitized before sampling.")
        return torch.clamp(mean, -100.0, 100.0)

    def safe_actor_mean(self, observations, context: str = "act"):
        raw_mean = self.actor_mean(observations)
        return self._sanitize_mean(raw_mean, context=context)
    
    def act(self, observations):
        dist = self._action_dist(observations)
        pre_tanh = dist.sample()
        actions = torch.tanh(pre_tanh)
        return actions

    def get_actions_log_prob(self, actions):
        raise NotImplementedError("Use evaluate() instead")

    def act_inference(self, observations):
        mean = self.safe_actor_mean(observations, context="inference")
        return torch.tanh(mean)

    def evaluate(self, observations, actions):
        dist = self._action_dist(observations)
        action_log_probs = self._squashed_log_prob(dist, actions)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        values = self.critic(observations)
        cost_values = self.cost_critic(observations)
        
        return action_log_probs, values, cost_values, dist_entropy

    def get_values(self, observations):
        values = self.critic(observations)
        cost_values = self.cost_critic(observations)
        return values, cost_values
