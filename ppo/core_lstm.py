import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def net(observation_shape, action_shape, activation=nn.ReLU, output_activation=nn.Identity):
    model = nn.Sequential(*[
        nn.Linear(np.prod(observation_shape), 128), activation(),
        nn.Linear(128, 128), activation(),
        nn.Linear(128, 128), activation(),
        nn.Linear(128, np.prod(action_shape))
    ])
    return model

class cnn_model(nn.Module):
    def __init__(self, num_inputs, num_out, activation=nn.ReLU):
        super(cnn_model, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32*6*6, 512)#32*6*6, 256)#512, 256)
        # self.critic_linear = nn.Linear(512, 1)
        # self.actor_linear = nn.Linear(512, num_actions)
        self.fc_out = nn.Linear(512, num_out)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hidden):
        #x = x / 255.0  # scale to 0-1
        #print(x.shape)
        #x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x ,hidden = self.lstm(x, hidden)#, (torch.zeros((1, 512)), torch.zeros((1, 512)))
        out = self.fc_out(x)
        return out, (x,hidden)

def cnn_net(observation_shape, action_shape, activation=nn.ReLU, output_activation=nn.Identity):
    model = nn.Sequential(*[
        nn.Conv2d(observation_shape, 32, 3, stride=2, padding=1),activation(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1), activation(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1), activation(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1), activation(),
        nn.Linear(32 * 6 * 6, 512), activation(),
        nn.Linear(512, np.prod(action_shape))
    ])

    return model

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class userActor(nn.Module):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, pretrain=None):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = cnn_model(obs_dim, act_dim, activation=activation)
        if pretrain != None:
            print('\n\nLoading pretrained from %s.\n\n' % pretrain)
        #    prams = torch.load(pretrain)
         #   import copy
          #  self.logits_net.load_state_dict(prams.state_dict())#copy.deepcopy(prams))
        print(self.mu_net)

    def forward(self, obs, act=None, hidden=None):
        mu, hidden=self.mu_net(obs, hidden)
        std = torch.exp(self.log_std)
        pi = Normal(mu,std)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act).sum(axis=-1)
        return pi, logp_a, hidden


class userCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = cnn_model(obs_dim, 1, activation=activation)#cnn_net([obs_dim] + list(hidden_sizes) + [1], activation)
        print(self.v_net)

    def forward(self, obs, hidden):
        v, _ = self.v_net(obs, hidden)
        return torch.squeeze(v, -1) # Critical to ensure v has right shape.


