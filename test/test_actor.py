import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPActor(nn.Module):
    def __init__(self,obs,obs_dim,hidden_sizes,act_dim,activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.obs=obs

    def _distribution(self):
        mu = self.mu_net(self.obs)
        std = torch.exp(self.log_std)
        self.pi=Normal(mu,std)
        return self.pi
    
    def _get_action(self):
        self.act=self.pi.sample()
        return self.act
    
    def _log_prob_of_act_from_distribution(self):
        logp_a=pi.log_prob(self.act).sum(axis=-1)
        return logp_a

obs_dim=3
act_dim=3
observation=torch.as_tensor([0.5, 0.1, 0],dtype=torch.float32)
hidden_sizes=[64,64]
activation=nn.Tanh

actor=MLPActor(observation,obs_dim,hidden_sizes,act_dim,activation)
pi=actor._distribution()
act=actor._get_action()
logp_a=actor._log_prob_of_act_from_distribution()

print('actor={},\npi={},\nact={},\nlogp_a={}'.format(actor,pi,act,logp_a))

