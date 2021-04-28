import gym

env=gym.make('Breakout-v0')

print(env)
print(env.observation_space)
print(env.observation_space.sample())
print(env.action_space)
print(env.action_space.sample())