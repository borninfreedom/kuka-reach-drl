#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   slide_bar_for_camera.py
@Time    :   2021/03/20 19:45:08
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

import os,inspect
current_dir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(current_dir)
import sys
sys.path.append('../')

from utils.SlideBars import SlideBars

p.connect(p.GUI)
p.setGravity(0,0,-10)

# # some camera parameters
# camera_parameters={
#     'width':720,
#     'height':720,
#     'fov':60,
#     'near':0.02,
#     'far':1,
#     'eye_position':[0.6, 0, 0.6],
#     'target_position':[0.6, 0, 0],
#     'camera_up_vector':[1, 0, 0], # I really do not know the parameter's effect.
#     'light_direction':[0.5,0,1],  #the direction is from the light source position to the origin of the world frame.
# }

# view_matrix=p.computeViewMatrix(
#     cameraEyePosition=camera_parameters['eye_position'],
#     cameraTargetPosition=camera_parameters['target_position'],
#     cameraUpVector=camera_parameters['camera_up_vector']
# )

# projection_matrix = p.computeProjectionMatrixFOV(
#     fov=camera_parameters['fov'],
#     aspect=camera_parameters['width']/camera_parameters['height'],
#     nearVal=camera_parameters['near'],
#     farVal=camera_parameters['far']
# )

x_low_obs=0.2
x_high_obs=0.7
y_low_obs=-0.3
y_high_obs=0.3
z_low_obs=0
z_high_obs=0.55

p.configureDebugVisualizer(lightPosition=[5,0,5])
  
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
                                cameraTargetPosition=[0.55, -0.35, 0.2])
#这些是周围那些白线，用来观察是否超过了obs的边界
p.addUserDebugLine(lineFromXYZ=[x_low_obs,y_low_obs,0],
                    lineToXYZ=[x_low_obs,y_low_obs,z_high_obs])
p.addUserDebugLine(lineFromXYZ=[x_low_obs,y_high_obs,0],
                    lineToXYZ=[x_low_obs,y_high_obs,z_high_obs])
p.addUserDebugLine(lineFromXYZ=[x_high_obs,y_low_obs,0],
                    lineToXYZ=[x_high_obs,y_low_obs,z_high_obs])
p.addUserDebugLine(lineFromXYZ=[x_high_obs,y_high_obs,0],
                    lineToXYZ=[x_high_obs,y_high_obs,z_high_obs])

p.addUserDebugLine(lineFromXYZ=[x_low_obs,y_low_obs,z_high_obs],
                    lineToXYZ=[x_high_obs,y_low_obs,z_high_obs])
p.addUserDebugLine(lineFromXYZ=[x_low_obs,y_high_obs,z_high_obs],
                    lineToXYZ=[x_high_obs,y_high_obs,z_high_obs])
p.addUserDebugLine(lineFromXYZ=[x_low_obs,y_low_obs,z_high_obs],
                    lineToXYZ=[x_low_obs,y_high_obs,z_high_obs])
p.addUserDebugLine(lineFromXYZ=[x_high_obs,y_low_obs,z_high_obs],
                    lineToXYZ=[x_high_obs,y_high_obs,z_high_obs])

urdf_root_path = pybullet_data.getDataPath()
p.loadURDF(os.path.join(urdf_root_path, "plane.urdf"), basePosition=[0, 0, -0.65])
kuka_id = p.loadURDF(os.path.join(urdf_root_path, "kuka_iiwa/model.urdf"), useFixedBase=True)
p.loadURDF(os.path.join(urdf_root_path, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
p.loadURDF(os.path.join(urdf_root_path,"cube_small.urdf"),basePosition=[x_low_obs,y_low_obs,0])
p.loadURDF(os.path.join(urdf_root_path,"cube_small.urdf"),basePosition=[x_low_obs,y_high_obs,0])
p.loadURDF(os.path.join(urdf_root_path,"cube_small.urdf"),basePosition=[x_high_obs,y_low_obs,0])
p.loadURDF(os.path.join(urdf_root_path,"cube_small.urdf"),basePosition=[x_high_obs,y_high_obs,0])
# p.loadURDF(os.path.join(urdf_root_path, "tray/traybox.urdf"),basePosition=[0.55,0,0])
#object_id=p.loadURDF(os.path.join(urdf_root_path, "random_urdfs/000/000.urdf"), basePosition=[0.53,0,0.02])
object_id=p.loadURDF(os.path.join(urdf_root_path,"random_urdfs/000/000.urdf"),
            basePosition=[random.uniform(x_low_obs,x_high_obs),
                            random.uniform(y_low_obs,y_high_obs),
                            0.01])

num_joints = p.getNumJoints(kuka_id)
init_joint_positions = [0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539]

orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])

for i in range(num_joints):
    p.resetJointState(bodyUniqueId=kuka_id,
                        jointIndex=i,
                        targetValue=init_joint_positions[i],
                        )
# images = p.getCameraImage(
#     width=camera_parameters['width'],
#     height=camera_parameters['height'],
#     viewMatrix=view_matrix,
#     projectionMatrix=projection_matrix,
#     shadow=True,
#     lightDirection=camera_parameters['light_direction'],
#     renderer=p.ER_BULLET_HARDWARE_OPENGL
# )[2]       # 2 stands for rgbPixels return, it contains [r,g,b,alpha],we dropout alpha.

width_id=p.addUserDebugParameter(
    paramName='width',
    rangeMin=0,
    rangeMax=760,
    startValue=320
)

height_id=p.addUserDebugParameter(
    paramName='height',
    rangeMin=0,
    rangeMax=760,
    startValue=320
)

fov_id=p.addUserDebugParameter(
    paramName='fov',
    rangeMin=0,
    rangeMax=100,
    startValue=48
)

near_id=p.addUserDebugParameter(
    paramName='near',
    rangeMin=0,
    rangeMax=2,
    startValue=0.5
)

far_id=p.addUserDebugParameter(
    paramName='far',
    rangeMin=0,
    rangeMax=10,
    startValue=1
)

# height_id=p.addUserDebugParameter(
#     paramName='width',
#     rangeMin=0,
#     rangeMax=1024,
#     startValue=720
# )

eye_position_x_id=p.addUserDebugParameter(
    paramName='eye_position_x',
    rangeMin=-5,
    rangeMax=5,
    startValue=0.59
)

eye_position_y_id=p.addUserDebugParameter(
    paramName='eye_position_y',
    rangeMin=-5,
    rangeMax=5,
    startValue=0
)

eye_position_z_id=p.addUserDebugParameter(
    paramName='eye_position_z',
    rangeMin=0,
    rangeMax=5,
    startValue=0.8
)

target_position_x_id=p.addUserDebugParameter(
    paramName='target_position_x',
    rangeMin=-5,
    rangeMax=5,
    startValue=0.55
)   

target_position_y_id=p.addUserDebugParameter(
    paramName='target_position_y',
    rangeMin=-5,
    rangeMax=5,
    startValue=0
)   

target_position_z_id=p.addUserDebugParameter(
    paramName='target_position_z',
    rangeMin=0,
    rangeMax=5,
    startValue=0.463
)   

camera_up_vector_x_id=p.addUserDebugParameter(
    paramName='camera_up_vector_x',
    rangeMin=-5,
    rangeMax=5,
    startValue=1
)   

camera_up_vector_y_id=p.addUserDebugParameter(
    paramName='camera_up_vector_y',
    rangeMin=-5,
    rangeMax=5,
    startValue=0
)   

camera_up_vector_z_id=p.addUserDebugParameter(
    paramName='camera_up_vector_z',
    rangeMin=0,
    rangeMax=5,
    startValue=1
)   

light_direction_x_id=p.addUserDebugParameter(
    paramName='light_direction_x',
    rangeMin=-5,
    rangeMax=5,
    startValue=0.6
)   

light_direction_y_id=p.addUserDebugParameter(
    paramName='light_direction_y',
    rangeMin=-5,
    rangeMax=5,
    startValue=0
)   

light_direction_z_id=p.addUserDebugParameter(
    paramName='light_direction_z',
    rangeMin=0,
    rangeMax=5,
    startValue=1
)   
    
slide_bars=SlideBars(kuka_id)
motorIndices=slide_bars.add_slidebars()

while True:


    slide_values=slide_bars.get_slidebars_values()
    p.setJointMotorControlArray(kuka_id,
                            motorIndices,
                            p.POSITION_CONTROL,
                            targetPositions=slide_values,
                            )


    width_value=p.readUserDebugParameter(
        itemUniqueId=width_id
    )
    height_value=p.readUserDebugParameter(
        itemUniqueId=height_id
    )
    fov_value=p.readUserDebugParameter(
        itemUniqueId=fov_id
    )
    near_value=p.readUserDebugParameter(
        itemUniqueId=near_id
    )
    far_value=p.readUserDebugParameter(
        itemUniqueId=far_id
    )
    eye_position_x_value=p.readUserDebugParameter(
        itemUniqueId=eye_position_x_id
    )
    eye_position_y_value=p.readUserDebugParameter(
        itemUniqueId=eye_position_y_id
    )
    eye_position_z_value=p.readUserDebugParameter(
        itemUniqueId=eye_position_z_id
    )
    target_position_x_value=p.readUserDebugParameter(
        itemUniqueId=target_position_x_id
    )
    target_position_y_value=p.readUserDebugParameter(
        itemUniqueId=target_position_y_id
    )
    target_position_z_value=p.readUserDebugParameter(
        itemUniqueId=target_position_z_id
    )
    camera_up_vector_x_value=p.readUserDebugParameter(
        itemUniqueId=camera_up_vector_x_id
    )
    camera_up_vector_y_value=p.readUserDebugParameter(
        itemUniqueId=camera_up_vector_y_id
    )
    camera_up_vector_z_value=p.readUserDebugParameter(
        itemUniqueId=camera_up_vector_z_id
    )
    light_direction_x_value=p.readUserDebugParameter(
        itemUniqueId=light_direction_x_id
    )
    light_direction_y_value=p.readUserDebugParameter(
        itemUniqueId=light_direction_y_id
    )
    light_direction_z_value=p.readUserDebugParameter(
        itemUniqueId=light_direction_z_id
    )


    view_matrix=p.computeViewMatrix(
        cameraEyePosition=[eye_position_x_value,
                            eye_position_y_value,
                            eye_position_z_value],
        cameraTargetPosition=[target_position_x_value,
                target_position_y_value,
                target_position_z_value],
        cameraUpVector=[camera_up_vector_x_value,
                camera_up_vector_y_value,
                camera_up_vector_z_value]
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov_value,
        aspect=width_value/height_value,
        nearVal=near_value,
        farVal=far_value
    )

    images = p.getCameraImage(
        width=int(width_value),
        height=int(height_value),
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        lightDirection=[light_direction_x_value,light_direction_y_value,light_direction_z_value],
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )      # 2 stands for rgbPixels return, it contains [r,g,b,alpha],we dropout alpha.
    
    # images=p.getCameraImage(
    #     width=int(width_value),
    #     height=int(height_value),
    # )
    p.stepSimulation()