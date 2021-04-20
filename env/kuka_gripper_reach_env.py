#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   kuka_base_env.py
@Time    :   2021/04/19 21:42:15
@Author  :   Yan Wen 
@Version :   1.0
@Contact :   z19040042@s.upc.edu.cn
@Desc    :   None
'''

# here put the import lib
import logging
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
import math
from colorama import Fore, Back, init

init(autoreset=True)
import sys

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(threadName)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
#     #filename='./logs/client1-{}.log'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())),
#     #filemode='w')
# )

logger = logging.getLogger(__name__)

formatter = logging.Formatter(
    '%(asctime)s - %(threadName)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
)

stream_handler = logging.StreamHandler()

stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class KukaGripperReachEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    max_steps_one_episode = 1000

    def __init__(self, is_render=False, is_good_view=False):

        self.is_render = is_render
        self.is_good_view = is_good_view

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.gripper_length = 0.257

        self.x_low_obs = 0.2
        self.x_high_obs = 0.7
        self.y_low_obs = -0.3
        self.y_high_obs = 0.3
        self.z_low_obs = 0
        self.z_high_obs = 0.55

        self.x_low_action = -0.4
        self.x_high_action = 0.4
        self.y_low_action = -0.4
        self.y_high_action = 0.4
        self.z_low_action = -0.6
        self.z_high_action = 0.3

        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        self.action_space = spaces.Box(low=np.array(
            [self.x_low_action, self.y_low_action, self.z_low_action]),
                                       high=np.array([
                                           self.x_high_action,
                                           self.y_high_action,
                                           self.z_high_action
                                       ]),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([
                self.x_low_obs, self.y_low_obs, self.z_low_obs, self.x_low_obs,
                self.y_low_obs, self.z_low_obs + self.gripper_length
            ]),
            high=np.array([
                self.x_high_obs, self.y_high_obs, self.z_high_obs,
                self.x_high_obs, self.y_high_obs,
                self.z_high_obs + self.gripper_length
            ]),
            dtype=np.float32)

        self.step_counter = 0

        self.end_effector_index = 6
        self.gripper_index = 7

        self.urdf_root_path = pybullet_data.getDataPath()
        # lower limits for null space
        self.lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.upper_limits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.joint_damping = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]

        self.init_joint_positions = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
            -0.006539, 0.000048, -0.299912, 0.000000, -0.000043, 0.299960,
            0.000000, -0.000200
        ]

        self.orientation = p.getQuaternionFromEuler([math.pi, 0, math.pi])

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        #p.connect(p.GUI)
        self.step_counter = 0

        p.resetSimulation()
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated = False
        p.setGravity(0, 0, -10)

        #这些是周围那些白线，用来观察是否超过了obs的边界
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"),
                   basePosition=[0, 0, -0.65])

        self.kuka_id = p.loadSDF(
            os.path.join(self.urdf_root_path,
                         "kuka_iiwa/kuka_with_gripper2.sdf"))[0]
        # The return of loadSDF is a tuple (0,)

        table_uid = p.loadURDF(os.path.join(self.urdf_root_path,
                                            "table/table.urdf"),
                               basePosition=[0.5, 0, -0.65])
        p.changeVisualShape(table_uid, -1, rgbaColor=[1, 1, 1, 1])

        self.object_id = p.loadURDF(
            os.path.join(self.urdf_root_path, "random_urdfs/000/000.urdf"),
            basePosition=[
                random.uniform(self.x_low_obs, self.x_high_obs),
                random.uniform(self.y_low_obs, self.y_high_obs), 0.01
            ],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))

        for i in range(p.getNumJoints(self.kuka_id)):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.init_joint_positions[i],
            )

        p.resetJointState(self.kuka_id, 8, -0.3)
        p.resetJointState(self.kuka_id, 10, 0)
        p.resetJointState(self.kuka_id, 11, 0.3)
        p.resetJointState(self.kuka_id, 13, 0)

        p.stepSimulation()
        return self._resolve_obs_return()

    def _resolve_obs_return(self):
        object_position = list(
            p.getBasePositionAndOrientation(self.object_id)[0])
        robot_end_effector_position = list(
            p.getLinkState(self.kuka_id, self.end_effector_index)[4])

        logger.debug(Fore.BLUE + 'object_position={}'.format(object_position))
        logger.debug(Fore.YELLOW + 'robot end effector position={}'.format(
            robot_end_effector_position))

        return np.array(object_position + robot_end_effector_position).astype(
            np.float32)

    def step(self, action):
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv

        current_pos = np.array(
            p.getLinkState(self.kuka_id,
                           self.end_effector_index)[4]).astype(np.float32)

        new_robot_pos = [
            current_pos[0] + dx, current_pos[1] + dy, current_pos[2] + dz
        ]

        robot_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_id,
            endEffectorLinkIndex=self.end_effector_index,
            targetPosition=[
                new_robot_pos[0], new_robot_pos[1], new_robot_pos[2]
            ],
            targetOrientation=self.orientation,
            jointDamping=self.joint_damping,
        )

        for i in range(self.end_effector_index):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=robot_joint_positions[i],
            )
        p.stepSimulation()

        if self.is_good_view:
            time.sleep(0.05)

        self.step_counter += 1
        return self._reward()

    def _reward(self):

        robot_end_effector_position = np.array(
            p.getLinkState(self.kuka_id,
                           self.end_effector_index)[4]).astype(np.float32)

        # We got the position of the robot end effector not gripper end effector,
        # so we need to compasent the gripper length
        # What we want is the gripper end effector position
        robot_end_effector_position[2] -= self.gripper_length
        object_position = np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(
                np.float32)

        square_dx = np.float32(
            (robot_end_effector_position[0] - object_position[0])**2)
        square_dy = np.float32(
            (robot_end_effector_position[1] - object_position[1])**2)
        square_dz = np.float32(
            (robot_end_effector_position[2] - object_position[2])**2)

        distance = np.float32(sqrt(square_dx + square_dy + square_dz))

        x = np.float32(robot_end_effector_position[0])
        y = np.float32(robot_end_effector_position[1])
        z = np.float32(robot_end_effector_position[2])
        logger.debug(Fore.RED + 'x,y,z={},{},{}'.format(x, y, z))

        terminated = bool(x < self.x_low_obs or x > self.x_high_obs
                          or y < self.y_low_obs or y > self.y_high_obs
                          or z < self.z_low_obs or z > self.z_high_obs)

        if terminated:
            reward = -0.1
            self.terminated = True

        elif self.step_counter > self.max_steps_one_episode:
            reward = -0.1
            self.terminated = True

        elif distance < 0.1:
            reward = 1
            self.terminated = True
        else:
            reward = 0
            self.terminated = False

        info = {'distance:', distance}

        return self._resolve_obs_return(), reward, self.terminated, info

    def close(self):
        p.disconnect()


if __name__ == '__main__':
    # 这一部分是做baseline，即让机械臂随机选择动作，看看能够得到的分数
    env = KukaGripperReachEnv(is_good_view=True,is_render=True)
    print('env={}'.format(env))
    print(env.observation_space.shape)
    # print(env.observation_space.sample())
    # print(env.action_space.sample())
    print(env.action_space.shape)
    obs = env.reset()
    print(Fore.RED + 'obs={}'.format(obs))
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print('obs={},reward={},done={},info={}'.format(obs, reward, done, info))

    sum_reward = 0
    success_times = 0
    for i in range(100):
        env.reset()
        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print('obs={},reward={},done={},info={}'.format(
                obs, reward, done, info))
            sum_reward += reward
            if reward == 1:
                success_times += 1
            if done:
                break
        # time.sleep(0.1)
    print()
    print('sum_reward={}'.format(sum_reward))
    print('success rate={}'.format(success_times / 50))
