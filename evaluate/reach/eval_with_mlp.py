#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   eval.py
@Time    :   2021/03/20 14:32:06
@Author  :   Yan Wen 
@Version :   1.0
@Contact :   z19040042@s.upc.edu.cn
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib


from env.kuka_reach_env import KukaReachEnv
from ppo.ppo import ppo
from spinup.utils.mpi_tools import mpi_fork
import ppo.core as core
import torch
# import os
# print(os.getcwd())

env=KukaReachEnv(is_good_view=True,is_render=True)
obs=env.reset()

ac=torch.load("pretrained/reach-mlp/model.pt")
print('ac={}'.format(ac))

actions=ac.act(torch.as_tensor(obs,dtype=torch.float32))

print('actions={}'.format(actions))

sum_reward=0
for i in range(50):
    obs=env.reset()
    for step in range(1000):
        actions=ac.act(torch.as_tensor(obs,dtype=torch.float32))
        obs,reward,done,info=env.step(actions)
        sum_reward+=reward
        if done:
            break

print('sum reward={}'.format(sum_reward))

