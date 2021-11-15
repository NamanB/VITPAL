import gym
import gym_minigrid

# from gym_minigrid.wrappers import *


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    if env_key == 'MiniGrid-FrozenLakeS7-v0':
        env = gym_minigrid.wrappers.RGBImgObsWrapper(env)
    env.seed(seed)
    return env
