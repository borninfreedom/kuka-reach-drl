from env.kuka_reach_env import KukaReachEnv
from ppo.ppo import ppo
from spinup.utils.mpi_tools import mpi_fork
import ppo.core as core


cnn_policy=core.CNNSharedNet(observation_space=[1,3,720,720],hidden_sizes=(18,64,64))
print(cnn_policy)