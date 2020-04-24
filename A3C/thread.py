""" Training thread for A3C
"""

import numpy as np
from threading import Thread, Lock
from keras.utils import to_categorical
from utils.networks import tfSummary

import keras.backend as K 
import wandb
import cv2
from tensorflow.compat.v1.keras.backend import set_session

from collections import deque

episode = 0
lock = Lock()

# **** Caution: Do not modify this cell ****
# initialize total reward across episodes
cumulative_reward = 0
episode = 0

def evaluate(episodic_reward):
  '''
  Takes in the reward for an episode, calculates the cumulative_avg_reward
    and logs it in wandb. If episode > 100, stops logging scores to wandb.
    Called after playing each episode. See example below.

  Arguments:
    episodic_reward - reward received after playing current episode
  '''
  global episode
  global cumulative_reward
  episode += 1

  # log total reward received in this episode to wandb
  wandb.log({'episodic_reward': episodic_reward})

  # add reward from this episode to cumulative_reward
  cumulative_reward += episodic_reward

  # calculate the cumulative_avg_reward
  # this is the metric your models will be evaluated on
  cumulative_avg_reward = cumulative_reward/episode

  # log cumulative_avg_reward over all episodes played so far
  wandb.log({'cumulative_avg_reward': cumulative_avg_reward})


def training_thread(agent, Nmax, env, action_dim, f, tqdm, render, num):
    """ Build threads to run shared computation across
    """

    global episode
    position = deque(maxlen=50); position.append(0)
    set_session(agent.actor.sess)
    with agent.actor.sess.as_default():
        with agent.actor.graph.as_default():
            while episode < Nmax:

                # Reset episode
                time, cumul_reward, done = 0, 0, False
                old_state = env.reset()
                actions, states, rewards = [], [], []
                while not done and episode < Nmax:
                    # Actor picks an action (following the policy)
                    a, a_vect = agent.policy_action(np.expand_dims(old_state, axis=0))
                    # Retrieve new state, reward, and whether the state is terminal
                    new_state, r, done, _ = env.step(a)
                    cumul_reward += r

                    # Reward for not staying in place
                    if a == 2: position.append(position[-1]+1)
                    if a == 3: position.append(position[-1]-1)
                    r_w = abs(max(position) - min(position))/100
                    r += r_w

                    # Memorize (s, a, r) for training
                    actions.append(to_categorical(a, action_dim))
                    rewards.append(r)
                    states.append(old_state)
                    # Update current state
                    old_state = new_state
                    time += 1

                    # Asynchronous training
                    if(time%f==0 or done):
                        lock.acquire()
                        agent.train_models(states, actions, rewards, done, a_vect)
                        lock.release()
                        actions, states, rewards = [], [], []

                # Update episode count
                with lock:
                    tqdm.set_description("Score: " + str(cumul_reward))
                    tqdm.update(1)
                    if(episode < Nmax):
                        episode += 1

                    if num == 0:
                        evaluate(cumul_reward)
                        wandb.log({'confidence': np.amax(a_vect)})
