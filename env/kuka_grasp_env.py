#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   kuka_reach_env.py
@Time    :   2021/03/20 14:33:24
@Author  :   Yan Wen 
@Version :   1.0
@Contact :   z19040042@s.upc.edu.cn
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib

import pybullet as p
import pybullet_data
import os
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from math import sqrt
import random
import time
import logging
import math
from kuka_base_env import KukaBaseEnv
from colorama import Fore,Back,init
init(autoreset=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(threadName)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    filename='./logs/kuka_grasp_env.log',
    filemode='w')

logger = logging.getLogger(__name__)

formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
stream_handler = logging.StreamHandler()

stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


# logging模块的使用
# 级别                何时使用
# DEBUG       细节信息，仅当诊断问题时适用。
# INFO        确认程序按预期运行
# WARNING     表明有已经或即将发生的意外（例如：磁盘空间不足）。程序仍按预期进行
# ERROR       由于严重的问题，程序的某些功能已经不能正常执行
# CRITICAL    严重的错误，表明程序已不能继续执行


class KukaGraspEnv(KukaBaseEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    max_steps_one_episode = 100

    def __init__(self, is_render=False, is_good_view=False):
        super(KukaGraspEnv,self).__init__(is_render,is_good_view)


    def resolve_reset_return(self):
        self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]
        return np.array(self.object_pos).astype(np.float32)
    


if __name__ == '__main__':
    # 这一部分是做baseline，即让机械臂随机选择动作，看看能够得到的分数
    env = KukaGraspEnv(is_render=True,is_good_view=True)
    
    print(Fore.GREEN+'env={}'.format(env))
    print(env.observation_space.shape)
    # print(env.observation_space.sample())
    # print(env.action_space.sample())
    print(env.action_space.shape)
    obs = env.reset()
    print(Fore.BLUE+'obs={}'.format(obs))
    # sum_reward=0
    # for i in range(10):
    #     env.reset()
    #     for i in range(2000):
    #         action=env.action_space.sample()
    #         #action=np.array([0,0,0.47-i/1000])
    #         obs,reward,done,info=env.step(action)
    #       #  print("i={},\naction={},\nobs={},\ndone={},\n".format(i,action,obs,done,))
    #         print(colored("reward={},info={}".format(reward,info),"cyan"))
    #        # print(colored("info={}".format(info),"cyan"))
    #         sum_reward+=reward
    #         if done:
    #             break
    #        # time.sleep(0.1)
    # print()
    # print(sum_reward)

