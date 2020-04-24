import sys
import random
import numpy as np

from tqdm import tqdm
from .agent import Agent
from random import random, randrange

from utils.memory_buffer import MemoryBuffer
from utils.networks import tfSummary
from utils.stats import gather_stats
from collections import deque

import cv2
import wandb
import base64
import glob
import io

from math import log
from scipy.stats import entropy

from multiprocessing import Pool

# **** Caution: Do not modify this cell ****
# initialize total reward across episodes
cumulative_reward = 0
episode = 0

def evaluate(episodic_reward, epsilon):
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

  wandb.log({'epsilon': epsilon})

def init_buffer(env):
    data = []
    # Init buffer
    while len(data) < 1000:
        done = False
        position = deque(maxlen=50); position.append(0)
        old_state = env.reset()

        # Initialize with a right bias
        for s in range(np.random.randint(0,50)):
            old_state, r, done, _ = env.step(4)
            if done:
                done = False
                old_state = env.reset()

        while not done:
            # Actor picks a random
            a = np.random.randint(6)

            # Retrieve new state, reward, and whether the state is terminal
            new_state, r, done, _ = env.step(a)

            # Memorize for experience replay
            if r == 0: r_r = 0
            elif r > 0: r_r = 1
            else: r_r = -1

            # Reward for not staying in place
            if a == 2: position.append(position[-1]+1)
            if a == 3: position.append(position[-1]-1)
            r_w = abs(max(position) - min(position))/10000
            r_r += r_w

            data.append([old_state, a, r_r, done, new_state])

            old_state = new_state

    return data

class DDQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, action_dim, state_dim, args, input_size, hp, export_path, env):
        """ Initialization
        """

        self.export_path = export_path

        # Environment and DDQN parameters
        self.with_per = args.with_per
        self.action_dim = action_dim
        self.state_dim = (args.consecutive_frames,) + state_dim
        #
        self.lr = hp["lr"]
        self.gamma = 0.99
        # Exploration parameters for epsilon greedy strategy
        self.explore_start = self.epsilon = 1.0 # exploration probability at start
        self.explore_stop = 0.1                 # minimum exploration probability
        self.decay_rate = 0.000001             # exponential decay rate for exploration prob

        self.buffer_size = 20000
        self.input_size = input_size

        self.video_dir = args.video_dir

        # Create actor and critic networks
        self.agent = Agent(self.state_dim, action_dim, self.lr, args.dueling, input_size, args.load)
        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(self.buffer_size, args.with_per)

        try:
            # Init buffer
            threads = 16
            p = Pool(processes=threads)
            while self.buffer.size() < self.buffer_size:

                # Set up threaded frame accumulation
                buffers = p.map_async(init_buffer, [env]*threads)
                datas = buffers.get()

                # Record in global memory
                for data in datas:
                    for entry in data:
                        self.memorize(*entry)

                # Mitigate memory leak
                del buffers
                del datas

                print("Buffer size: {}".format(self.buffer.size()))

        except KeyboardInterrupt:
            p.close()
            p.join()
        p.close()
        p.join()

        # Train on pure randomness for a while
        tqdm_e = tqdm(range(2000), desc='Score', leave=True, unit=" episodes")
        for e in tqdm_e:
            record = False
            if e % 100 == 0: record = True
            self.train_agent(args.batch_size, record)

            if e%1000 == 0:
                self.agent.transfer_weights()

            # Display score
            tqdm_e.refresh()


    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            a_vect = self.agent.predict(s)[0]
            return np.argmax(a_vect)

    def train_agent(self, batch_size, record=False):
        """ Train Q-network on batch sampled from the buffer
        """
        # Sample experience from memory buffer (optionally with PER)
        s, a, r, d, new_s, idx = self.buffer.sample_batch(batch_size)

        # Apply Bellman Equation on batch samples to train our DDQN
        q = self.agent.predict(s)
        next_q = self.agent.predict(new_s)
        q_targ = self.agent.target_predict(new_s)

        for i in range(s.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = np.argmax(next_q[i,:])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]

        # Train on batch
        self.agent.fit(s, q, record=record)

    def train(self, env, args):
        """ Main DDQN Training Algorithm
        """

        results = []
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        decay_step = 0
        self.t = 0
        for e in tqdm_e:
            # Reset episode
            time, cumul_reward, cumul_r_r, done  = 0, 0, 0, False
            position = deque(maxlen=50); position.append(0)
            old_state = env.reset()

            while not done:
                decay_step += 1
                env.render()
                # Actor picks an action (following the policy)
                a = self.policy_action(old_state)

                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a)

                # Memorize for experience replay
                if r == 0: r_r = 0
                elif r > 0: r_r = 1
                else: r_r = -1

                # Reward for not staying in place
                if a == 2: position.append(position[-1]+1)
                if a == 3: position.append(position[-1]-1)
                r_w = abs(max(position) - min(position))/10000
                r_r += r_w

                self.memorize(old_state, a, r_r, done, new_state)

                # Update current state
                old_state = new_state
                cumul_reward += r
                cumul_r_r += r_r
                time += 1

                self.epsilon = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(-self.decay_rate * decay_step)

                # Train DDQN
                if(self.buffer.size() > args.batch_size) and self.t%2000 == 0:
                    self.train_agent(args.batch_size)
                self.t+=1

                if self.t%10000 == 0:
                    self.agent.transfer_weights()

            if e % 50 == 0:
                self.agent.save("./model.h5")
                wandb.save("./model.h5")

            if e%100 == 0:
                # wandb logging
                evaluate(cumul_reward, self.epsilon)
                self.train_agent(args.batch_size, record=True)

            # Display score
            text = "Score: {}, Fake Score: {:.2f}".format(str(cumul_reward), cumul_r_r)
            tqdm_e.set_description(text)
            tqdm_e.refresh()

            # render gameplay video
            if (e %50 == 0):
              mp4list = glob.glob('video/'+self.video_dir+'/*.mp4')
              if len(mp4list) > 0:
                mp4 = mp4list[-1]
                video = io.open(mp4, 'r+b').read()
                encoded = base64.b64encode(video)
                # log gameplay video in wandb
                wandb.log({"gameplays": wandb.Video(mp4, fps=4, format="gif")})

        return results

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """

        if(self.with_per):
            q_val = self.agent.predict(state)
            q_val_t = self.agent.target_predict(new_state)
            next_best_action = np.argmax(q_val)
            new_val = reward + self.gamma * q_val_t[0, next_best_action]
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0
        self.buffer.memorize(state, action, reward, done, new_state, td_error)

    def save(self, path):
        self.agent.save(path)

    def load_weights(self, path):
        self.agent.load_weights(path)
