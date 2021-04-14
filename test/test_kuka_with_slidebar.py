#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   test_kuka_with_slidebar.py
@Time    :   2021/04/14 21:35:14
@Author  :   Yan Wen 
@Version :   1.0
@Contact :   z19040042@s.upc.edu.cn
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib

import os
import inspect
import sys
import pybullet as p
import time
import pybullet_data
import random
import math
# This code snap can resolve the problem with python import in different folders
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(current_dir)
sys.path.append('../')

from utils.SlideBars import SlideBars

x_low_obs = 0.2
x_high_obs = 0.7
y_low_obs = -0.3
y_high_obs = 0.3
z_low_obs = 0
z_high_obs = 0.55

x_low_action = -0.4
x_high_action = 0.4
y_low_action = -0.4
y_high_action = 0.4
z_low_action = -0.6
z_high_action = 0.3

p.connect(p.GUI)
p.setGravity(0, 0, -10)

p.configureDebugVisualizer(lightPosition=[5, 0, 5])
p.resetDebugVisualizerCamera(cameraDistance=1.5,
                             cameraYaw=0,
                             cameraPitch=-40,
                             cameraTargetPosition=[0.55, -0.35, 0.2])

urdf_root_path = pybullet_data.getDataPath()
# lower limits for null space
lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
# upper limits for null space
upper_limits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
# joint ranges for null space
joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
# restposes for null space
rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
# joint damping coefficents
joint_damping = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]

init_joint_positions = [
    0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539
]

orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])

p.setAdditionalSearchPath(pybullet_data.getDataPath())
kuka_uid = p.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf")[0]
num_joints = p.getNumJoints(kuka_uid)
print('num_joints={}'.format(num_joints))

p.loadURDF(os.path.join(urdf_root_path, "plane.urdf"),
           basePosition=[0, 0, -0.65])
#kuka_uid = p.loadURDF(os.path.join(urdf_root_path, "kuka_iiwa/model.urdf"), useFixedBase=True)
table_uid = p.loadURDF(os.path.join(urdf_root_path, "table/table.urdf"),
                       basePosition=[0.5, 0, -0.65])
p.changeVisualShape(table_uid, -1, rgbaColor=[1, 1, 1, 1])
# p.loadURDF(os.path.join(urdf_root_path, "tray/traybox.urdf"),basePosition=[0.55,0,0])
#object_id=p.loadURDF(os.path.join(urdf_root_path, "random_urdfs/000/000.urdf"), basePosition=[0.53,0,0.02])
object_id = p.loadURDF(os.path.join(urdf_root_path,
                                    "random_urdfs/000/000.urdf"),
                       basePosition=[
                           random.uniform(x_low_obs, x_high_obs),
                           random.uniform(y_low_obs, y_high_obs), 0.01
                       ])

for i in range(num_joints):
    num_info = p.getJointInfo(kuka_uid, i)
    print(num_info)

slide_bars = SlideBars(kuka_uid)
motorIndices = slide_bars.add_slidebars()
while True:
    p.stepSimulation()
    slide_values = slide_bars.get_slidebars_values()
    p.setJointMotorControlArray(
        kuka_uid,
        motorIndices,
        p.POSITION_CONTROL,
        targetPositions=slide_values,
    )

# while True:
#     p.stepSimulation()
