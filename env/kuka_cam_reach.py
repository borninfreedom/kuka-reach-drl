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
import sys
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from math import sqrt
import random
import time
from numpy import arange
import logging
import math
import cv2
from termcolor import colored
from colorama import Fore,init
init(autoreset=True)    # this lets colorama takes effect only in current line.
                        # Otherwise, colorama will let the sentences below 'print(Fore.GREEN+'xx')'
                        # all become green color.

#### 一些变量 ######
LOGGING_LEVEL=logging.INFO
# is_render=False
# is_good_view=False   #这个的作用是在step时加上time.sleep()，把机械比的动作放慢，看的更清，但是会降低训练速度
#########################

# logging.basicConfig(
#     level=LOGGING_LEVEL,
#     format='%(asctime)s - %(threadName)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
#     filename='../logs/reach_env.log'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())),
#     filemode='w')
# logger = logging.getLogger(__name__)
# env_logger=logging.getLogger('env.py')

# logging模块的使用
# 级别                何时使用
# DEBUG       细节信息，仅当诊断问题时适用。
# INFO        确认程序按预期运行
# WARNING     表明有已经或即将发生的意外（例如：磁盘空间不足）。程序仍按预期进行
# ERROR       由于严重的问题，程序的某些功能已经不能正常执行
# CRITICAL    严重的错误，表明程序已不能继续执行

class KukaReachEnv(gym.Env):
    metadata = {'render.modes':['human','rgb_array'],'video.frames_per_second':50}
    max_steps_one_episode = 1000

    def __init__(self,is_render=False,is_good_view=False):


        # some camera parameters
        self.camera_parameters={
            'width':720,
            'height':720,
            'fov':60,
            'near':0.02,
            'far':1,
            'eye_position':[0.6, 0, 0.6],
            'target_position':[0.6, 0, 0],
            'camera_up_vector':[1, 0, 0], # I really do not know the parameter's effect.
            'light_direction':[0.5,0,1],  #the direction is from the light source position to the origin of the world frame.
        }
   

        self.view_matrix=p.computeViewMatrix(
            cameraEyePosition=self.camera_parameters['eye_position'],
            cameraTargetPosition=self.camera_parameters['target_position'],
            cameraUpVector=self.camera_parameters['camera_up_vector']
        )

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
            aspect=self.camera_parameters['width']/self.camera_parameters['height'],
            nearVal=self.camera_parameters['near'],
            farVal=self.camera_parameters['far']
        )


        self.is_render=is_render
        self.is_good_view=is_good_view

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.x_low_obs=0.2
        self.x_high_obs=0.7
        self.y_low_obs=-0.3
        self.y_high_obs=0.3
        self.z_low_obs=0
        self.z_high_obs=0.55

        self.x_low_action=-0.4
        self.x_high_action=0.4
        self.y_low_action=-0.4
        self.y_high_action=0.4
        self.z_low_action=-0.6
        self.z_high_action=0.3

        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        self.action_space=spaces.Box(low=np.array([self.x_low_action,self.y_low_action,self.z_low_action]),
                                     high=np.array([self.x_high_action,self.y_high_action,self.z_high_action]),
                                     dtype=np.float32)
        self.observation_space=spaces.Box(low=np.array([self.x_low_obs,self.y_low_obs,self.z_low_obs]),
                                     high=np.array([self.x_high_obs,self.y_high_obs,self.z_high_obs]),
                                     dtype=np.float32)
        self.step_counter=0

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
        self.joint_damping = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]

        self.init_joint_positions = [0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539]

        self.orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])

        self.seed()
        self.reset()

    def seed(self,seed=None):
        self.np_random,seed=seeding.np_random(seed)
        return [seed]

    def reset(self):
        #p.connect(p.GUI)
        self.step_counter=0

        p.resetSimulation()
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated=False
        p.setGravity(0, 0, -10)

        #这些是周围那些白线，用来观察是否超过了obs的边界
        p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_low_obs,0],
                           lineToXYZ=[self.x_low_obs,self.y_low_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_high_obs,0],
                           lineToXYZ=[self.x_low_obs,self.y_high_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_high_obs,self.y_low_obs,0],
                           lineToXYZ=[self.x_high_obs,self.y_low_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_high_obs,self.y_high_obs,0],
                           lineToXYZ=[self.x_high_obs,self.y_high_obs,self.z_high_obs])

        p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_low_obs,self.z_high_obs],
                           lineToXYZ=[self.x_high_obs,self.y_low_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_high_obs,self.z_high_obs],
                           lineToXYZ=[self.x_high_obs,self.y_high_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_low_obs,self.z_high_obs],
                           lineToXYZ=[self.x_low_obs,self.y_high_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_high_obs,self.y_low_obs,self.z_high_obs],
                           lineToXYZ=[self.x_high_obs,self.y_high_obs,self.z_high_obs])

        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"), basePosition=[0, 0, -0.65])
        self.kuka_id = p.loadURDF(os.path.join(self.urdf_root_path, "kuka_iiwa/model.urdf"), useFixedBase=True)
        p.loadURDF(os.path.join(self.urdf_root_path, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
       # p.loadURDF(os.path.join(self.urdf_root_path, "tray/traybox.urdf"),basePosition=[0.55,0,0])
        #object_id=p.loadURDF(os.path.join(self.urdf_root_path, "random_urdfs/000/000.urdf"), basePosition=[0.53,0,0.02])
        self.object_id=p.loadURDF(os.path.join(self.urdf_root_path,"random_urdfs/000/000.urdf"),
                   basePosition=[random.uniform(self.x_low_obs,self.x_high_obs),
                                 random.uniform(self.y_low_obs,self.y_high_obs),
                                 0.01])

        self.num_joints = p.getNumJoints(self.kuka_id)

        for i in range(self.num_joints):
            p.resetJointState(bodyUniqueId=self.kuka_id,
                              jointIndex=i,
                              targetValue=self.init_joint_positions[i],
                              )

        self.robot_pos_obs=p.getLinkState(self.kuka_id,self.num_joints-1)[4]
        #logging.debug("init_pos={}\n".format(p.getLinkState(self.kuka_id,self.num_joints-1)))
        p.stepSimulation()

        self.images = p.getCameraImage(
            width=self.camera_parameters['width'],
            height=self.camera_parameters['height'],
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            shadow=True,
            lightDirection=self.camera_parameters['light_direction'],
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )[2]       # 2 stands for rgbPixels return, it contains [r,g,b,alpha],we dropout alpha.

        p.enableJointForceTorqueSensor(
            bodyUniqueId=self.kuka_id,

            jointIndex=self.num_joints-1,
            enableSensor=True
        )

        # print(Fore.GREEN+'force_sensor={}'.format(self._get_force_sensor_value()))

        self.object_pos=p.getBasePositionAndOrientation(self.object_id)[0]
        #return np.array(self.object_pos).astype(np.float32)
        #return np.array(self.robot_pos_obs).astype(np.float32)
        return self.images

    def step(self,action):
        dv=0.005
        dx=action[0]*dv
        dy=action[1]*dv
        dz=action[2]*dv

        self.current_pos=p.getLinkState(self.kuka_id,self.num_joints-1)[4]
       # logging.debug("self.current_pos={}\n".format(self.current_pos))
        self.new_robot_pos=[self.current_pos[0]+dx,
                            self.current_pos[1]+dy,
                            self.current_pos[2]+dz]
        #logging.debug("self.new_robot_pos={}\n".format(self.new_robot_pos))
        self.robot_joint_positions = p.calculateInverseKinematics(
                                                       bodyUniqueId=self.kuka_id,
                                                       endEffectorLinkIndex=self.num_joints - 1,
                                                       targetPosition=[self.new_robot_pos[0],
                                                                       self.new_robot_pos[1],
                                                                       self.new_robot_pos[2]],
                                                       targetOrientation=self.orientation,
                                                       jointDamping=self.joint_damping,
                                                       )
        for i in range(self.num_joints):
            p.resetJointState(bodyUniqueId=self.kuka_id,
                              jointIndex=i,
                              targetValue=self.robot_joint_positions[i],
                              )
        p.stepSimulation()

        #在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        if self.is_good_view:
            time.sleep(0.05)

        self.step_counter+=1
        

        return self._reward()


    def _reward(self):

        #一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        self.robot_state=p.getLinkState(self.kuka_id,self.num_joints-1)[4]
        # self.object_state=p.getBasePositionAndOrientation(self.object_id)
        # self.object_state=np.array(self.object_state).astype(np.float32)
        #
        self.object_state=np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]
        ).astype(np.float32)

        square_dx=(self.robot_state[0]-self.object_state[0])**2
        square_dy=(self.robot_state[1]-self.object_state[1])**2
        square_dz = (self.robot_state[2] - self.object_state[2]) ** 2

        #用机械臂末端和物体的距离作为奖励函数的依据
        self.distance=sqrt(square_dx+square_dy+square_dz)
        #print(self.distance)

        x=self.robot_state[0]
        y=self.robot_state[1]
        z=self.robot_state[2]

        #如果机械比末端超过了obs的空间，也视为done，而且会给予一定的惩罚
        terminated=bool(
            x<self.x_low_obs
            or x>self.x_high_obs
            or y<self.y_low_obs
            or y>self.y_high_obs
            or z<self.z_low_obs
            or z>self.z_high_obs
        )

        if terminated:
            reward=-0.1
            self.terminated=True

        #如果机械臂一直无所事事，在最大步数还不能接触到物体，也需要给一定的惩罚
        elif self.step_counter>self.max_steps_one_episode:
            reward=-0.1
            self.terminated=True

        elif self.distance<0.1:
            reward=1
            self.terminated=True
        else:
            reward=0
            self.terminated=False

        info={'distance:',self.distance}
        #self.observation=self.robot_state
        self.observation=self.object_state
        return np.array(self.observation).astype(np.float32),reward,self.terminated,info

    def close(self):
        p.disconnect()

    def run_for_debug(self,target_position):
        temp_robot_joint_positions = p.calculateInverseKinematics(
                                                       bodyUniqueId=self.kuka_id,
                                                       endEffectorLinkIndex=self.num_joints - 1,
                                                       targetPosition=target_position,
                                                       targetOrientation=self.orientation,
                                                       jointDamping=self.joint_damping,
                                                       )
        for i in range(self.num_joints):
            p.resetJointState(bodyUniqueId=self.kuka_id,
                              jointIndex=i,
                              targetValue=temp_robot_joint_positions[i],
                              )
        p.stepSimulation()

        #在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        if self.is_good_view:
            time.sleep(0.05)

        return self._get_force_sensor_value()

    def _get_force_sensor_value(self):
        force_sensor_value=p.getJointState(
            bodyUniqueId=self.kuka_id,
            jointIndex=self.num_joints-1
        )[2][2]     
        # the first 2 stands for jointReactionForces, the second 2 stands for Fz, 
        # the pybullet methods' return is a tuple,so can not 
        # index it with str like dict. I think it can be improved 
        # that return value is a dict rather than tuple.
        return force_sensor_value

if __name__ == '__main__':
    # 这一部分是做baseline，即让机械臂随机选择动作，看看能够得到的分数
    env=KukaReachEnv(is_good_view=True,is_render=True)
   # print(env)
   # print(env.observation_space.shape)
   # print(env.observation_space.sample())
    a=env.reset()
    b=cv2.cvtColor(a,cv2.COLOR_RGBA2RGB)
    # for i in range(720):
    #     for j in range(720):
    #         for k in range(3):
    #             if not a[i][j][k]==b[i][j][k]:
    #                 print(Fore.RED+'there is unequal')
    #                 raise ValueError('there is unequal.')

              

    #print(a)
    #force_sensor=env.run_for_debug([0.6,0.0,0.03])
   # print(Fore.RED+'after force sensor={}'.format(force_sensor))
    #print(env.action_space.sample())

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