""" Deep RL Algorithms for OpenAI Gym environments
"""

import os
import sys
import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from DDQN.ddqn import DDQN
from A3C.a3c import A3C

from tensorflow.compat.v1.keras.backend import set_session
from keras.utils import to_categorical

from utils.atari_environment import AtariEnvironment
from utils.continuous_environments import Environment
from utils.networks import get_session

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=1.0)
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 16}, gpu_options=gpu_options)
tf.compat.v1.disable_eager_execution()

# import wandb
import wandb
from wandb.keras import WandbCallback

# initialize a new wandb run
wandb.init(project="qualcomm")
# define hyperparameters
wandb.config.episodes = 50000
wandb.config.batch_size = 32
wandb.config.learning_rate = 1e-7
input_size = (32,32)
export_path = os.path.join(wandb.run.dir, "model.h5")

hp = {"lr": wandb.config.learning_rate}

gym.logger.set_level(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import string, random
def randomString(stringLength=20):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Trazining parameters')
    #
    parser.add_argument('--type', type=str, default='A3C',help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")
    parser.add_argument('--is_atari', dest='is_atari', action='store_false', help="Atari Environment")
    parser.add_argument('--with_PER', dest='with_per', action='store_true', help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    #
    parser.add_argument('--nb_episodes', type=int, default=wandb.config.episodes, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=wandb.config.batch_size, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=4, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--training_interval', type=int, default=wandb.config.batch_size, help="Network training frequency")
    parser.add_argument('--n_threads', type=int, default=16, help="Number of threads (A3C)")
    #
    parser.add_argument('--gather_stats', dest='gather_stats', action='store_true',help="Compute Average reward per episode (slower)")
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
    parser.add_argument('--env', type=str, default='SpaceInvaders-v0',help="OpenAI Gym Environment")
    #
    parser.add_argument('--load', type=str, default=None,help="OpenAI Gym Environment")
    #
    parser.add_argument('--video_dir', type=str, default=randomString())
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
    #summary_writer = tf.summary.FileWriter(args.type + "/tensorboard_" + args.env)

    # Atari Environments
    env = AtariEnvironment(args, input_size)
    state_dim = env.get_state_size()
    action_dim = env.get_action_size()

    # Pick algorithm to train
    if(args.type=="DDQN"):
        algo = DDQN(action_dim, state_dim, args, input_size, hp, export_path, env)
    elif(args.type=="A3C"):
        algo = A3C(action_dim, state_dim, args.consecutive_frames, args, is_atari=args.is_atari, lr=wandb.config.learning_rate)

    # Train
    stats = algo.train(env, args)

  except KeyboardInterrupt:
    algo.save(export_path)
    env.env.close()

  algo.save(export_path)
  env.env.close()

if __name__ == "__main__":
    main()
