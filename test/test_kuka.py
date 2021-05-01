import pybullet as p
import pybullet_data
import random
import math
import time
from colorama import Fore,Back,init
init(autoreset=True)
p.connect(p.GUI)
p.setGravity(0, 0, -10)
p.setAdditionalSearchPath('../models/')

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

# lower limits for null space
lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
# upper limits for null space
upper_limits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
# joint ranges for null space
joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
# restposes for null space
rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
# joint damping coefficents
joint_damping = [
    0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
]
p.loadURDF("plane.urdf", basePosition=[0, 0, -0.65])
#kuka_id = p.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf")[0]
kuka_id = p.loadURDF("diana/,useFixedBase=True)
object_orn=p.getQuaternionFromEuler([0,0,math.pi/2])
p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])
# p.loadURDF(os.path.join(urdf_root_path, "tray/traybox.urdf"),basePosition=[0.55,0,0])
#object_id=p.loadURDF(os.path.join(urdf_root_path, "random_urdfs/000/000.urdf"), basePosition=[0.53,0,0.02])
object_id = p.loadURDF("random_urdfs/000/000.urdf",
                       basePosition=[
                           random.uniform(x_low_obs, x_high_obs),
                           random.uniform(y_low_obs, y_high_obs),
                           0.01
                       ],baseOrientation=object_orn)

object_pos=p.getBasePositionAndOrientation(object_id)[0]
object_pos=list(object_pos)
object_pos[2]+=0.257
#object_pos=[0.5,0,0.257]
print(object_pos)
num_joints=p.getNumJoints(kuka_id)
for i in range(num_joints):
    print(p.getJointInfo(kuka_id,i))
orientation = p.getQuaternionFromEuler(
    [-math.pi, 0, -math.pi/2])
init_pos=[0.5,0,0.6]
init_joint_angles=p.calculateInverseKinematics(kuka_id,6,init_pos,orientation,lower_limits,upper_limits,joint_ranges,rest_poses)
# jointPositions = [
# 0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
# -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
# ]
for i in range(7):
    p.resetJointState(kuka_id,i,init_joint_angles[i])
    
# for jointIndex in range(num_joints):
#     p.resetJointState(kuka_id, jointIndex, jointPositions[jointIndex])
'''
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
'''
motor_indices=[]
for i in range(num_joints):
    jointInfo = p.getJointInfo(kuka_id, i)
    qIndex = jointInfo[3]
    if qIndex > -1:
        motor_indices.append(i)

orientation = p.getQuaternionFromEuler(
    [-math.pi, 0, -math.pi/2])

joint_angle=p.calculateInverseKinematics(kuka_id,6,object_pos,orientation,lower_limits,upper_limits,joint_ranges,rest_poses)
print(joint_angle)
'''
(0.13114866527745114, 
1.0629605718491106, 
0.05068869856160148, 
-1.6176172235290052, 
-0.01650185506252476, 
0.9877395036237796, 
-0.04838155362856612, 
-0.04838155362856612, 
0.0, 
0.0, 
0.5503808245213768, 
0.0)
'''

object_pos_for_calc=object_pos
object_pos_for_calc[2]=0.5

while True:
   
    object_id = p.loadURDF("random_urdfs/000/000.urdf",
                        basePosition=[
                            random.uniform(x_low_obs, x_high_obs),
                            random.uniform(y_low_obs, y_high_obs),
                            0.01
                        ],baseOrientation=object_orn)

    object_pos=p.getBasePositionAndOrientation(object_id)[0]
    object_pos=list(object_pos)
    object_pos[2]=0.6
    print(Fore.RED+'object_pos={}'.format(object_pos))
    object_pos_for_calc=object_pos.copy()
    while True:
        # object_pos[0]+=0.01
        # object_pos[2]+=0.257
        
        object_pos_for_calc[2]-=0.05
        print(object_pos_for_calc)
        if object_pos_for_calc[2]<0.257:
            object_pos_for_calc[2]=0.257
            
        joint_angle=p.calculateInverseKinematics(kuka_id,6,object_pos_for_calc,orientation,lower_limits,upper_limits,joint_ranges,rest_poses)

        for i in range(len(joint_angle)):
            p.resetJointState(kuka_id,i,joint_angle[i])
            end_effortor_pos=p.getLinkState(kuka_id,6)[4]
            if end_effortor_pos[2]<0.4 and end_effortor_pos[2]>0.26:
                p.resetJointState(kuka_id,8,-0.3)
                p.resetJointState(kuka_id,10,0)
                p.resetJointState(kuka_id,11,0.3)
                p.resetJointState(kuka_id,13,0)
                break
            elif end_effortor_pos[2]<=0.26 and end_effortor_pos[2]>=0:
                p.resetJointState(kuka_id,8,0)
                p.resetJointState(kuka_id,10,0)
                p.resetJointState(kuka_id,11,0)
                p.resetJointState(kuka_id,13,0)
                break
            else:
                break
        p.stepSimulation()
        time.sleep(0.1)
    p.removeBody(object_id)
    time.sleep(0.1)
        
    


    
    




