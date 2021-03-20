from env.kuka_reach_env import KukaReachEnv
from ppo.ppo import ppo
from spinup.utils.mpi_tools import mpi_fork
import ppo.core as core
import torch
# import os
# print(os.getcwd())

env=KukaReachEnv(is_good_view=True,is_render=True)
obs=env.reset()

ac=torch.load("logs/ppo-kuka-reach/ppo-kuka-reach_s0/pyt_save/model.pt")
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

