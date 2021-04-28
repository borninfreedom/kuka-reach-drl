import ray
from ray import tune
import time
ray.shutdown()
ray.init(ignore_reinit_error=True)

config = {
    'env': 'Breakout-v0',
    'num_workers':10,
    'num_gpus':1,
    'framework':'torch',
    'render_env':False,
    'num_gpus_per_worker':0,
    'num_envs_per_worker':2,
}
stop = {
    'episode_reward_mean': 500
}
st=time.time()
results = tune.run(
    'PPO', # Specify the algorithm to train
    config=config,
    stop=stop
)
print('elapsed time=',time.time()-st)
 
ray.shutdown()