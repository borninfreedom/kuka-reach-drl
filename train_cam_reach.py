from env.kuka_cam_reach import KukaCamReachEnv
from ppo import core

env=KukaCamReachEnv()

obs=env.reset()
print('obs={},\nobs.shape={}'.format(obs,obs.shape))

cnn_policy=core.CNNSharedNet(observation_space=env.observation_space,hidden_sizes=(18,64,64))
print('cnn_policy=\n{}'.format(cnn_policy))

ac=core.CNNActorCritic(observation_space=env.observation_space,action_space=env.action_space)
print('ac={}'.format(ac))
