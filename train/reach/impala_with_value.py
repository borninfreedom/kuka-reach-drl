import os,inspect
current_dir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(current_dir)
import sys
sys.path.append('../../')

import time
import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from custom_envs.kuka_reach_env import KukaReachEnv

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

ray.shutdown()
ray.init(ignore_reinit_error=True)

config = {
    'env': KukaReachEnv,
    'env_config':{
        "is_render":False,
        "is_good_view":False,
        "max_steps_one_episode":1000
    },
    'num_workers':1,
    'num_gpus':1,
    'framework':'torch',
    'render_env':False,
    'num_gpus_per_worker':0,
    'num_envs_per_worker':1,
}

stop = {
    'episode_reward_mean': 200
}
st=time.time()
results = tune.run(
    'IMPALA', # Specify the algorithm to train
    config=config,
    stop=stop
)
print('elapsed time=',time.time()-st)
 
ray.shutdown()