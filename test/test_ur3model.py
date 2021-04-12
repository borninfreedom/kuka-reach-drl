import os
import inspect
import sys
import pybullet as p
import time

# This code snap can resolve the problem with python import in different folders
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(current_dir)
sys.path.append('../')

from utils import SlideBars

p.connect(p.GUI)
p.setGravity(0, 0, -10)
ur3_uid = p.loadURDF("../models/ur3_with_gripper.urdf")
num_joints = p.getNumJoints(ur3_uid)
print('num_joints={}'.format(num_joints))

for i in range(num_joints):
    num_info = p.getJointInfo(ur3_uid, i)
    print(num_info)

slide_bars = SlideBars(ur3_uid)
motorIndices = slide_bars.add_slidebars()
while True:
    p.stepSimulation()
    slide_values = slide_bars.get_slidebars_values()
    p.setJointMotorControlArray(
        ur3_uid,
        motorIndices,
        p.POSITION_CONTROL,
        targetPositions=slide_values,
    )

# while True:
#     p.stepSimulation()