import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_actions, actor_hidden_dims=[256, 256, 256], critic_hidden_dims=[256, 256, 256], activation="elu", init_noise_std=1.0):
        super(ActorCritic, self).__init__()
        
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ELU() # Default

        # Actor
        actor_layers = []
        actor_layers.append(nn.Linear(num_obs, actor_hidden_dims[0]))
        actor_layers.append(self.activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(self.activation)
        self.actor_mean = nn.Sequential(*actor_layers)
        
        # Standard deviation for the policy (learnable parameter)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        # Critic (Reward Value Function)
        critic_layers = []
        critic_layers.append(nn.Linear(num_obs, critic_hidden_dims[0]))
        critic_layers.append(self.activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(self.activation)
        self.critic = nn.Sequential(*critic_layers)

        # Cost Critic (Cost Value Function) - for Safe RL
        # Uses same architecture as reward critic usually
        cost_critic_layers = []
        cost_critic_layers.append(nn.Linear(num_obs, critic_hidden_dims[0]))
        cost_critic_layers.append(self.activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                cost_critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                cost_critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                cost_critic_layers.append(self.activation)
        self.cost_critic = nn.Sequential(*cost_critic_layers)

    def forward(self):
        raise NotImplementedError
    
    def act(self, observations):
        mean = self.actor_mean(observations)
        dist = Normal(mean, self.std)
        actions = dist.sample()
        
        # In Isaac Lab, actions are often clipped in the env wrapper, but we return raw actions here
        # and let the algo handle log_prob
        
        return actions

    def get_actions_log_prob(self, actions):
        # This assumes we already have the distribution properties or re-compute?
        # Usually we recompute distribution from current observations
        raise NotImplementedError("Use evaluate() instead")

    def act_inference(self, observations):
        return self.actor_mean(observations)

    def evaluate(self, observations, actions):
        """
        Evaluate actions for a given observation: return action_log_prob, value, cost_value, entropy
        """
        mean = self.actor_mean(observations)
        dist = Normal(mean, self.std)
        
        action_log_probs = dist.log_prob(actions).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        values = self.critic(observations)
        cost_values = self.cost_critic(observations)
        
        return action_log_probs, values, cost_values, dist_entropy

    def get_values(self, observations):
        values = self.critic(observations)
        cost_values = self.cost_critic(observations)
        return values, cost_values
