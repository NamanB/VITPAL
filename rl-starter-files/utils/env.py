import gym

from gym_minigrid.wrappers import *


def make_env(env_key, seed=None):
    # TODO: add a parameter to specify which agent/algorithm, then make corresponding wrappers automatically (normal vs expert)
    env = gym.make(env_key)
    if env_key == 'MiniGrid-FrozenLakeS7-v0':
        # always use ImgObsWrapper to ignore mission text input for obs
        # env = ImgObsWrapper(RGBImgPartialObsWrapper(env))
        # env = VitpalRGBImgObsWrapper(env, tile_size=2)
        # env = VitpalExpertImgObsWrapper(env)
        env = VitpalTrainWrapper(env,tile_size=2)
    env.seed(seed)
    return env
