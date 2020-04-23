import sys
import gym
import time
import threading
import numpy as np

from tqdm import tqdm
from keras.models import Model
from keras import regularizers
from keras.layers import Input, Dense, Flatten, Reshape, Lambda, Concatenate, Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from keras.models import load_model
import keras

from .critic import Critic
from .actor import Actor
from .thread import training_thread
from utils.atari_environment import AtariEnvironment
from utils.continuous_environments import Environment
from utils.networks import conv_block
from utils.stats import gather_stats

import tensorflow as tf
import wandb

class A3C:
    """ Asynchronous Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, k, args, gamma = 0.99, lr = 0.0001, is_atari=False):
        """ Initialization
        """
        # Environment and A3C parameters
        self.act_dim = act_dim
        if(is_atari):
            self.env_dim = env_dim
        else:
            self.env_dim = (k,) + env_dim
        self.gamma = gamma
        self.lr = lr
        self.is_eval = False
        # Create actor and critic networks
        self.shared = self.buildNetwork()
        self.actor = Actor(self.env_dim, act_dim, self.shared, lr)
        self.critic = Critic(self.env_dim, act_dim, self.shared, lr)
        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()

        self.episode = 0

        if args.load is not None:
            # restore model
            fname = "model.h5"
            f_cname = "model.h5_critic.h5"
            run_path = "joshherr/qualcomm/"+args.load

            api = wandb.Api()
            run = api.run(run_path)
            local_path = None
            with run.file(fname).download(replace=True) as f:
              local_path = f.name
            self.actor.load_weights(local_path)
            with run.file(f_cname).download(replace=True) as f:
              local_path = f.name
            self.critic.load_weights(local_path)

    def buildNetwork(self):
        """ Assemble shared layers
        """
        inp = Input((self.env_dim))
        x = Reshape(self.env_dim)(inp)

        # Break input into s images for each history window
        inputs = []
        for s in range(self.env_dim[-1]):
          inputs.append(Reshape((self.env_dim[0], self.env_dim[1], 1)) \
                          (Lambda(lambda x: x[:,:,:,s])(inp)))
        
        ### Create feature extraction network for each history image
        outs = []
        for i in inputs:
            x = Conv2D(kernel_size=(8,8),
                        strides=(4,4),
                        filters=8,
                        activation = 'relu',
                        padding = 'same',
                        kernel_regularizer=keras.regularizers.l2(0.01))(i)
            x = Conv2D(kernel_size=(4,4),
                       strides=(2,2),
                       filters=16,
                       activation = 'relu',
                       padding = 'same',
                       kernel_regularizer=keras.regularizers.l2(0.01))(x)
            x = Conv2D(kernel_size=(3,3),
                       strides=(1,1),
                       filters=16,
                       activation = 'relu',
                       padding = 'same',
                       kernel_regularizer=keras.regularizers.l2(0.01))(x)
            x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

            outs.append(Reshape((self.env_dim[0]//2//2//2,
                                 self.env_dim[1]//2//2//2,-1))(x))

        ### Concatenate history network outputs
        x = Concatenate(axis=-1)(outs)
        x = Conv2D(kernel_size=(3,3),
                   strides=(1,1),
                   filters=32,
                   activation = 'relu',
                   padding = 'same',
                   kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = Conv2D(kernel_size=(3,3),
                   strides=(1,1),
                   filters=64,
                   activation = 'relu',
                   padding = 'same',
                   kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = Conv2D(kernel_size=(3,3),
                   strides=(1,1),
                   filters=64,
                   activation = 'relu',
                   padding = 'same',
                   kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)      

        x = Flatten()(x)
        return Model(inp, x)

    def policy_action(self, s):
        """ Use the actor's network to predict the next action to take, using the policy
        """
        a_vect = self.actor.predict(s).ravel()
        if self.is_eval:
            return np.argmax(a_vect)
        return np.random.choice(np.arange(self.act_dim), 1, p=a_vect)[0], a_vect

    def discount(self, r, done, s):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, states, actions, rewards, done, a_vect):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards, done, states[-1])
        state_values = self.critic.predict(np.array(states))
        advantages = discounted_rewards - np.reshape(state_values, len(state_values))

        # Networks optimization
        self.a_opt([np.array(states), np.array(actions), advantages])
        self.episode += 1

        if self.episode % 10 == 0:
            self.c_opt([np.array(states), discounted_rewards])

    def train(self, env, args):

        # Instantiate one environment per thread
        envs = [AtariEnvironment(args, self.env_dim) for i in range(args.n_threads)]
        state_dim = envs[0].get_state_size()
        action_dim = envs[0].get_action_size()

        # Create threads
        tqdm_e = tqdm(range(int(args.nb_episodes)), desc='Score', leave=True, unit=" episodes")

        threads = [threading.Thread(
                target=training_thread,
                daemon=True,
                args=(self,
                    args.nb_episodes,
                    envs[i],
                    action_dim,
                    args.training_interval,
                    tqdm_e,
                    args.render,
                    i)) for i in range(args.n_threads)]

        for t in threads:
            t.start()
            time.sleep(0.5)
        try:
            [t.join() for t in threads]
        except KeyboardInterrupt:
            print("Exiting all threads...")
        return None

    def save(self, path):
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)
