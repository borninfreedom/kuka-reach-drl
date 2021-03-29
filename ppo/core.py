#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   core.py
@Time    :   2021/03/20 14:32:33
@Author  :   Yan Wen 
@Version :   1.0
@Contact :   z19040042@s.upc.edu.cn
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib


import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
from ppo.logx import Logger
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


IS_DEBUG=False
core_logger=Logger(output_dir="../logs/",is_debug=IS_DEBUG)

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


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        core_logger.log("layers={}".format(layers),'green')
        core_logger.log("nn.Sequential={}".format(nn.Sequential(*layers)))

    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class userActor(nn.Module):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, pretrain=None):
        super().__init__()
        self.logits_net = cnn_model(obs_dim, act_dim, activation=activation)
        if pretrain != None:
            print('\n\nLoading pretrained from %s.\n\n' % pretrain)
        #    prams = torch.load(pretrain)
         #   import copy
          #  self.logits_net.load_state_dict(prams.state_dict())#copy.deepcopy(prams))
        print(self.logits_net)

    def forward(self, obs, act=None, hidden=None):
        logits, hidden=self.logits_net(obs, hidden)
        pi = Categorical(logits=logits)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a, hidden


class userCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = cnn_model(obs_dim, 1, activation=activation)#cnn_net([obs_dim] + list(hidden_sizes) + [1], activation)
        print(self.v_net)

    def forward(self, obs, hidden):
        v, _ = self.v_net(obs, hidden)
        return torch.squeeze(v, -1) # Critical to ensure v has right shape.


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        core_logger.log("mu_net={}".format(self.mu_net))

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        core_logger.log("mu={},std={},Normal(mu,std)={}".format(mu,std,Normal(mu,std)))
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        core_logger.log("pi.log_prob(act)={}".format(pi.log_prob(act)),'red')
        core_logger.log("pi.log_prob(act).sum(axis=-1)={}".format(pi.log_prob(act).sum(axis=-1)))

        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        core_logger.log("v_net={}".format(self.v_net))


    def forward(self, obs):
        core_logger.log("torch.squeeze(self.v_net(obs), -1)".format(torch.squeeze(self.v_net(obs), -1)),'green')
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            core_logger.log("pi={}".format(pi))
            a = pi.sample()
            core_logger.log("a={}".format(a))
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            core_logger.log("logp_a={}".format(logp_a))

            v = self.v(obs)
            core_logger.log("v={}".format(v),'blue')
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]



class CNNSharedNet(nn.Module):
    def __init__(self, observation_space, hidden_sizes):
        super(CNNSharedNet, self).__init__()
        pretrained_CNN = 'resnet'+str(hidden_sizes[0])
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', pretrained_CNN, pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        if observation_space.shape[0] == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_sizes[1])

    def forward(self, x):
        return self.resnet(x)

class CNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=[18,64,64], activation=nn.Tanh):
        super().__init__()
        # shared network
        self.shared = CNNSharedNet(observation_space, hidden_sizes)
        hidden_sizes.pop(0)
        dummy_obs_dim_to_be_replaced = 1
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(dummy_obs_dim_to_be_replaced, action_space.shape[0], hidden_sizes, activation)
            self.pi.mu_net[0] = self.shared
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(dummy_obs_dim_to_be_replaced, action_space.n, hidden_sizes, activation)
            self.pi.logits_net[0] = self.shared
        # build value function
        self.v  = MLPCritic(dummy_obs_dim_to_be_replaced, hidden_sizes, activation)
        self.v.v_net[0] = self.shared

    def step(self, obs):
        obs = obs
        with torch.no_grad():
            pi = self.pi._distribution(obs.unsqueeze(0))
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            a = a[0]
            v = self.v(obs.unsqueeze(0))
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

