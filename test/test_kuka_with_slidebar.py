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
    joint_info = p.getJointInfo(kuka_uid, i)
    print(joint_info)

"""
(0, b'J0', 0, 7, 6, 1, 0.5, 0.0, -2.96706, 2.96706, 300.0, 10.0, b'lbr_iiwa_link_1', (0.0, 0.0, 1.0), (0.1, 0.0, 0.0875), (0.0, 0.0, 0.0, 1.0), -1)
(1, b'J1', 0, 8, 7, 1, 0.5, 0.0, -2.0944, 2.0944, 300.0, 10.0, b'lbr_iiwa_link_2', (0.0, 0.0, 1.0), (0.0, 0.03, 0.08249999999999999), (-9.381873917569987e-07, 0.7071080798588513, 0.707105482510614, 9.381839456086129e-07), 0)
(2, b'J2', 0, 9, 8, 1, 0.5, 0.0, -2.96706, 2.96706, 300.0, 10.0, b'lbr_iiwa_link_3', (0.0, 0.0, 1.0), (-0.0003, 0.14549999999862046, -0.04200075117044365), (-9.381873917569987e-07, 0.7071080798588513, 0.707105482510614, -9.381839456086129e-07), 1)
(3, b'J3', 0, 10, 9, 1, 0.5, 0.0, -2.0944, 2.0944, 300.0, 10.0, b'lbr_iiwa_link_4', (0.0, 0.0, 1.0), (0.0, -0.03, 0.08550000000000002), (-0.7071080798594737, 0.0, 0.0, 0.7071054825112363), 2)
(4, b'J4', 0, 11, 10, 1, 0.5, 0.0, -2.96706, 2.96706, 300.0, 10.0, b'lbr_iiwa_link_5', (0.0, 0.0, 1.0), (0.0, 0.11749999999875532, -0.03400067770634158), (9.381873917569989e-07, 0.7071080798588513, 0.707105482510614, 9.381839456086127e-07), 3)
(5, b'J5', 0, 12, 11, 1, 0.5, 0.0, -2.0944, 2.0944, 300.0, 10.0, b'lbr_iiwa_link_6', (0.0, 0.0, 1.0), (-0.0001, -0.021, 0.1394999999999999), (-0.7071080798594737, 0.0, 1.0080490974868132e-23, 0.7071054825112363), 4)
(6, b'J6', 0, 13, 12, 1, 0.5, 0.0, -3.05433, 3.05433, 300.0, 10.0, b'lbr_iiwa_link_7', (0.0, 0.0, 1.0), (0.0, 0.0803999999994535, -0.00040029752961337673), (-9.381873917569987e-07, 0.7071080798588513, 0.707105482510614, -9.381839456086129e-07), 5)
(7, b'gripper_to_arm', 0, 14, 13, 1, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'base_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.02400000000000004), (0.0, 0.0, 0.0, 1.0), 6)
(8, b'base_left_finger_joint', 0, 15, 14, 1, 0.0, 0.0, -10.4, 10.01, 100.0, 0.0, b'left_finger', (0.0, 1.0, 0.0), (0.0, 0.024, 0.04500000000000015), (0.0, 0.024997395914712325, 0.0, 0.9996875162757026), 7)
(9, b'left_finger_base_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'left_finger_base', (0.0, 0.0, 0.0), (-0.0009954177603205688, 0.0, 0.04014991667795068), (0.0, 0.12467473338522773, 0.0, 0.992197667229329), 8)
(10, b'left_base_tip_joint', 0, 16, 15, 1, 0.0, 0.0, -10.1, 10.3, 0.0, 0.0, b'left_finger_tip', (0.0, 1.0, 0.0), (0.0064011650627963075, 0.0, 0.021752992447456466), (0.0, -0.24740395925452294, 0.0, 0.9689124217106448), 9)
(11, b'base_right_finger_joint', 0, 17, 16, 1, 0.0, 0.0, -10.01, 10.4, 100.0, 0.0, b'right_finger', (0.0, 1.0, 0.0), (0.0, 0.024, 0.04500000000000015), (0.0, -0.024997395914712325, 0.0, 0.9996875162757026), 7)
(12, b'right_finger_base_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'right_finger_base', (0.0, 0.0, 0.0), (0.0009954177603205688, 0.0, 0.04014991667795068), (0.0, -0.12467473338522773, 0.0, 0.992197667229329), 11)
(13, b'right_base_tip_joint', 0, 18, 17, 1, 0.0, 0.0, -10.3, 10.1, 0.0, 0.0, b'right_finger_tip', (0.0, 1.0, 0.0), (-0.0064011650627963075, 0.0, 0.021752992447456466), (0.0, 0.24740395925452294, 0.0, 0.9689124217106448), 12)
"""
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
