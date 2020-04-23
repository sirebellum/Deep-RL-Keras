""" Deep RL Algorithms for OpenAI Gym environments
"""

import os
import sys
import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from A2C.a2c import A2C
from A3C.a3c import A3C
from DDQN.ddqn import DDQN
from DDPG.ddpg import DDPG
from PPO.ppo import PPO

from tensorflow.compat.v1.keras.backend import set_session
from keras.utils import to_categorical

from utils.atari_environment import AtariEnvironment
from utils.continuous_environments import Environment
from utils.networks import get_session

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=1.0)
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 16}, gpu_options=gpu_options) 

# import wandb0
import wandb
from wandb.keras import WandbCallback

# initialize a new wandb run
wandb.init(project="qualcomm")
# define hyperparameters
wandb.config.episodes = 500000
wandb.config.batch_size = 128
export_path = os.path.join(wandb.run.dir, "model.h5")

gym.logger.set_level(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Trazining parameters')
    #
    parser.add_argument('--type', type=str, default='PPO',help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")
    parser.add_argument('--is_atari', dest='is_atari', action='store_false', help="Atari Environment")
    parser.add_argument('--with_PER', dest='with_per', action='store_true', help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    #
    parser.add_argument('--nb_episodes', type=int, default=wandb.config.episodes, help="Number of training episodes")
    parser.add_argument('--epochs', type=int, default=1, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=wandb.config.batch_size, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=4, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--training_interval', type=int, default=wandb.config.batch_size, help="Network training frequency")
    parser.add_argument('--n_threads', type=int, default=12, help="Number of threads (A3C)")
    #
    parser.add_argument('--gather_stats', dest='gather_stats', action='store_true',help="Compute Average reward per episode (slower)")
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
    parser.add_argument('--env', type=str, default='SpaceInvaders-v0',help="OpenAI Gym Environment")
    #
    parser.add_argument('--load', type=str, default=None,help="OpenAI Gym Environment")
    parser.add_argument('--buffer_size', type=int, default=10000,help="Number of samples to store")
    #
    #
    parser.set_defaults(render=False)
    return parser.parse_args(args)

def main(args=None):
    try:
        # Parse arguments
        if args is None:
            args = sys.argv[1:]
        args = parse_args(args)

        set_session(get_session(config=config))

        # Atari Environment Wrapper
        env = AtariEnvironment(args)
        state_dim = env.get_state_size()
        action_dim = env.get_action_size()

        # Pick algorithm to train
        if(args.type=="DDQN"):
            algo = DDQN(action_dim, state_dim, args)
        elif(args.type=="A2C"):
            algo = A2C(action_dim, state_dim, args.consecutive_frames)
        elif(args.type=="A3C"):
            algo = A3C(action_dim, state_dim, args.consecutive_frames, is_atari=args.is_atari)
        elif(args.type=="DDPG"):
            algo = DDPG(action_dim, state_dim, act_range, args.consecutive_frames)
        elif(args.type=="PPO"):
            algo = PPO(action_dim, state_dim, args.consecutive_frames, lr=1e-5)

        # Train
        stats = algo.train(env, args)

    except KeyboardInterrupt:
        algo.save(export_path)
        env.env.close()
    algo.save(export_path)
    env.env.close()


if __name__ == "__main__":
    main()
