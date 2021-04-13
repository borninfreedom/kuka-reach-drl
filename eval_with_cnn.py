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

from env.kuka_cam_reach import KukaCamReachEnv, CustomSkipFrame
from ppo.ppo_cnn import ppo
from spinup.utils.mpi_tools import mpi_fork
import ppo.core_cnn as core
import torch
# import os
# print(os.getcwd())

env = KukaCamReachEnv(is_good_view=True, is_render=True)
env = CustomSkipFrame(env)
obs = env.reset()

ac = torch.load("pretrained/reach-cnn/model.pt")
print('ac={}'.format(ac))

actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))

print('actions={}'.format(actions))

sum_reward = 0
success_times=0
for i in range(50):
    obs = env.reset()
    for step in range(env.max_steps_one_episode):
        actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))
        obs, reward, done, info = env.step(actions)
        if reward==1:
            success_times+=1
        if done:
            break
    

print('sum reward={}'.format(sum_reward))
print('success rate={}'.format(success_times/50))



