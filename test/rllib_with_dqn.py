import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer

ray.shutdown()

# twice the conservative memory settings from https://docs.ray.io/en/master/memory-management.html
ray.init(
    ignore_reinit_error=True,
)

ENV = "BreakoutNoFrameskip-v4"
TARGET_REWARD = 200
TRAINER = DQNTrainer

tune.run(
    TRAINER,
    stop={"episode_reward_mean": TARGET_REWARD},
    config={
      "env": ENV,
      "num_gpus": 0,  #1,
      "monitor": True,
      "evaluation_num_episodes": 25,
      "num_workers": 0,
      # based on https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/dqn/atari-duel-ddqn.yaml
      "double_q": True,
      "dueling": True,
      "num_atoms": 1,
      "noisy": False,
      "framework":"torch",
      "prioritized_replay": False,
      "n_step": 1,
      "target_network_update_freq": 8000,
      "lr": .0000625,
      "adam_epsilon": .00015,
      "hiddens": [512],
      "learning_starts": 20000,
      "buffer_size": 400_000,
      "rollout_fragment_length": 4,
      "train_batch_size": 32,
      "exploration_config": {
        "epsilon_timesteps": 500_000,
        "final_epsilon": 0.01,
      },
      "prioritized_replay_alpha": 0.5,
      "final_prioritized_replay_beta": 1.0,
      "prioritized_replay_beta_annealing_timesteps": 2_000_000,
      "timesteps_per_iteration": 10_000,
    }
)