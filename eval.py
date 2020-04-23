fname = "model.h5"
run_path="joshherr/qualcomm/3nq7zi1m"

# import wandb
import wandb

import numpy as np
import random
import math
import glob
import io
import os
import cv2
import base64
import tensorflow as tf

from collections import deque
from datetime import datetime
import keras

import sys
from utils.atari_environment import AtariEnvironment
import argparse

gpu_options = tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.2)
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 12}, gpu_options=gpu_options) 
sess = tf.Session(config=config) 
tf.keras.backend.set_session(sess)

def custom_loss(y_actual, y_predicted):
  entropy_loss = keras.backend.mean(keras.backend.sum(y_predicted*keras.backend.log(y_predicted), axis=1))
  mse_loss = keras.backend.mean(keras.backend.square(y_predicted - y_actual))
  return mse_loss - entropy_loss

from keras.models import load_model
import keras.losses
keras.losses.custom_loss = custom_loss

# restore model
api = wandb.Api()
run = api.run(run_path)
local_path = None
with run.file(fname).download(replace=True) as f:
  local_path = f.name
agent = load_model(local_path, compile=False)
#agent = load_model(fname, compile=False)
agent.summary()

import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(30)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from gym import spaces
def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Trazining parameters')
    #
    parser.add_argument('--is_atari', dest='is_atari', action='store_false', help="Atari Environment")
    #
    parser.add_argument('--consecutive_frames', type=int, default=4, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--env', type=str, default='SpaceInvaders-v0',help="OpenAI Gym Environment")
    #
    parser.set_defaults(render=False)
    return parser.parse_args(args)

# **** Caution: Do not modify this cell ****
# initialize total reward across episodes
cumulative_reward = 0
episode = 0

def evaluate(episodic_reward, reset=False):
  '''
  Takes in the reward for an episode, calculates the cumulative_avg_reward
    and logs it in wandb. If episode > 100, stops logging scores to wandb.
    Called after playing each episode. See example below.

  Arguments:
    episodic_reward - reward received after playing current episode
  '''
  global episode
  global cumulative_reward
  if reset:
    cumulative_reward = 0
    episode = 0
    
  episode += 1
  print("Episode: %d"%(episode))

  # your models will be evaluated on 100-episode average reward
  # therefore, we stop logging after 100 episodes
  if (episode > 100):
    print("Scores from episodes > 100 won't be logged in wandb.")
    return

  # log total reward received in this episode to wandb
  wandb.log({'episodic_reward': episodic_reward})

  # add reward from this episode to cumulative_reward
  cumulative_reward += episodic_reward

  # calculate the cumulative_avg_reward
  # this is the metric your models will be evaluated on
  cumulative_avg_reward = cumulative_reward/episode

  # log cumulative_avg_reward over all episodes played so far
  wandb.log({'cumulative_avg_reward': cumulative_avg_reward})

  return cumulative_avg_reward

from numpy.random import seed
from tensorflow import set_random_seed

args = sys.argv[1:]
args = parse_args(args)

env = AtariEnvironment(args, test=True)
state_dim = env.get_state_size()
action_dim = env.get_action_size()

action_size = env.action_space.n
print("Actions available(%d): %r"%(env.action_space.n, env.env.get_action_meanings()))

# initialize a new wandb run
wandb.init(project="qualcomm-evaluation")

# define hyperparameters
wandb.config.episodes = 100
wandb.config.runpath = run_path

for seed_ in [10]:#, 50, 100, 200, 500]:
  seed(seed_)
  set_random_seed(seed_)
  print("Seed: ",seed_)
  episode = 0

  # run for 100 episodes
  # Note: Please adjust this as needed to work with your model architecture.
  # Make sure you still call evaluate() with the reward received in each episode
  for i in range(wandb.config.episodes):
    # Set reward received in this episode = 0 at the start of the episode
    episodic_reward = 0
    reset = False

    # play a random game
    state = env.reset()

    done = False
    while not done:
      env.render()

      sreward = 0
      reward = 0

      action = agent.predict(np.expand_dims(state, axis=0))

      action = np.argmax(action)
      #action = np.random.choice(np.arange(action_dim), p=action[0])

      # perform the action and fetch next state, reward
      state, reward, done, _ = env.step(action)

      episodic_reward += reward
    
    # call evaluation function - takes in reward received after playing an episode
    # calculates the cumulative_avg_reward over 100 episodes & logs it in wandb
    if i == 0:
      reset = True

    cumulative_avg_reward = evaluate(episodic_reward, reset)

    if i >= 99:
      break

    '''
    # render gameplay video
    if (i %10 == 0):
      mp4list = glob.glob('video/*.mp4')
      if len(mp4list) > 0:
        mp4 = mp4list[-1]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)

        # log gameplay video in wandb
        wandb.log({"gameplays": wandb.Video(mp4, fps=4, format="gif")})
    '''

print("Final score: ", np.mean(cumulative_avg_reward))
