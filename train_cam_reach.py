#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   train_cam_reach.py
@Time    :   2021/03/25 20:45:28
@Author  :   Yan Wen 
@Version :   1.0
@Contact :   z19040042@s.upc.edu.cn
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib

from env.kuka_cam_reach import KukaCamReachEnv
from ppo import core
from ppo.ppo import ppo
from spinup.utils.mpi_tools import mpi_fork
import logging
from ppo.logx import Logger
from colorama import Fore,init
init(autoreset=True)

if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')

    #modified this to satisfy the custom env
    #parser.add_argument('--env', type=str, default=env)
    parser.add_argument('--is_render',type=bool,default=False)
    parser.add_argument('--is_good_view',type=bool,default=False)

    #parser.add_argument('--hid', type=int, default=64)
    #parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=6)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='ppo-kuka-cam-reach')
    parser.add_argument('--log_dir', type=str, default="./logs")
    args = parser.parse_args()


    env=KukaCamReachEnv(is_good_view=args.is_good_view,is_render=args.is_render)

    mpi_fork(args.cpu)  # run parallel code with mpi

    #obs=env.reset()

    #print('obs={},\nobs.shape={}'.format(obs,obs.shape))
    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed,data_dir=args.log_dir)

    # cnn_policy=core.CNNSharedNet(observation_space=env.observation_space,hidden_sizes=(18,64,64))
    # print('cnn_policy=\n{}'.format(cnn_policy))

    # ac=core.CNNActorCritic(observation_space=env.observation_space,action_space=env.action_space)
    # print('ac={}'.format(ac))
    print(Fore.GREEN+'cal ppo.')
    ppo(env,
        actor_critic=core.CNNActorCritic,
        ac_kwargs=dict(hidden_sizes=[18,64,64]),
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=env.max_steps_one_episode*args.cpu,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs)