import torch
import torch.nn as nn

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
    
    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


obs_dim=3
observation=torch.as_tensor([0.5, 0.1, 0],dtype=torch.float32)
hidden_sizes=[64,64]
activation=nn.Tanh

critic=MLPCritic(obs_dim,hidden_sizes,activation)
print('v_net={}'.format(critic.v_net))
print('v_net(obs)={}'.format(critic.v_net(observation)))
print('v_net forward={}'.format(critic.forward(observation)))
