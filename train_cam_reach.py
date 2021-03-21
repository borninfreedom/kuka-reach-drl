from env.kuka_cam_reach import KukaCamReachEnv
from ppo import core

env=KukaCamReachEnv()

obs=env.reset()
print('obs={},\nobs.shape={}'.format(obs,obs.shape))

cnn_policy=core.CNNSharedNet(observation=obs,hidden_sizes=(18,64,64))
print('cnn_policy=\n{}'.format(cnn_policy))
