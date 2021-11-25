#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
from gym_minigrid.minigrid import Grid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

def redraw(img):
    if args.agent_privileged:
        grid, vis_mask = Grid.decode(img)
        vis_mask=np.zeros(vis_mask.shape)

        xdist = np.abs(np.arange(vis_mask.shape[0]) - env.agent_pos[1])
        ydist = np.abs(np.arange(vis_mask.shape[1]) - env.agent_pos[0])
        xdist, ydist = np.meshgrid(xdist, ydist)
        vis_mask = (xdist + ydist) <= args.lava_render_dist
        vis_mask[0] = False
        vis_mask[:, 0] = False
        vis_mask[-1] = False
        vis_mask[:, -1] = False

        img = grid.render(args.tile_size,
            env.agent_pos,
            env.agent_dir,
            highlight_mask=vis_mask,
            lava_render_dist=args.lava_render_dist)
    elif args.agent_normal:
        # to see agent view, comment out this line and write pass
        img = env.render('rgb_array', tile_size=args.tile_size, highlight=False, lava_render_dist=0) 
    elif not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-FrozenLakeS7-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)
parser.add_argument(
    '--agent_normal',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)
parser.add_argument(
    '--agent_privileged',
    default=False,
    help="draw the agent sees",
    action='store_true'
)
parser.add_argument(
    "--lava_render_dist",
    type=int,
    help="manhattan distance within which to render lava (-1 for all)",
    default=-1
)



args = parser.parse_args()

env = gym.make(args.env)

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

if args.agent_normal:
    env = VitpalRGBImgObsWrapper(env)
    print('loaded wrapper')

if args.agent_privileged:
    env = VitpalExpertImgObsWrapper(env, lava_render_dist=args.lava_render_dist)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
