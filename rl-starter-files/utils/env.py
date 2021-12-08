import gym

from gym_minigrid.wrappers import *


def make_env(env_key, seed=None, lava_render_dist=-1):
    # TODO: add a parameter to specify which agent/algorithm, then make corresponding wrappers automatically (normal vs expert)
    env = gym.make(env_key)
    if env_key == 'MiniGrid-FrozenLakeS7-v0':
        # always use ImgObsWrapper to ignore mission text input for obs
        # env = ImgObsWrapper(RGBImgPartialObsWrapper(env))
        # env = VitpalRGBImgObsWrapper(env, tile_size=2)
        # env = VitpalExpertImgObsWrapper(env)
        print(lava_render_dist)
        # env = VitpalExpertImgObsWrapper(env, lava_render_dist=lava_render_dist)
        # env = VitpalExpertImgObsWrapper(env, lava_render_dist=1)
        env = VitpalTrainWrapper(env,tile_size=2)
    env.seed(seed)
    return env
