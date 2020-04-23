import numpy as np
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Dropout, Reshape, CuDNNGRU
from keras.optimizers import Adam
from .agent import Agent

import tensorflow as tf
import keras.backend as K
from tensorflow.compat.v1.keras.backend import set_session

class Actor(Agent):
    """ Actor for the A2C Algorithm
    """

    def __init__(self, inp_dim, out_dim, network, lr):
        Agent.__init__(self, inp_dim, out_dim, lr)
        self.model = self.addHead(network)
        self.action_pl = K.placeholder(shape=(None, self.out_dim))
        self.advantages_pl = K.placeholder(shape=(None,))
        # Pre-compile for threading# Load model
        self.sess = tf.Session()
        set_session(self.sess)
        self.graph = tf.get_default_graph()
        self.model._make_predict_function()

    def addHead(self, network):
        """ Assemble Actor network to predict probability of each action
        """
        x = Dense(512)(network.output)
        x = Dense(512)(x)
        self.out = Dense(self.out_dim, activation='softmax')(x)
        return Model(network.input, self.out)

    def optimizer(self):
        """ Actor Optimization: Advantages + Entropy term to encourage exploration
        (Cf. https://arxiv.org/abs/1602.01783)
        """
        weighted_actions = K.sum(self.action_pl * self.model.output, axis=1)
        eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(self.advantages_pl)
        entropy = K.sum(self.model.output * K.log(self.model.output + 1e-10), axis=1)
        loss = 0.1 * entropy - K.sum(eligibility)

        updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], loss)
        return K.function([self.model.input, self.action_pl, self.advantages_pl], [self.out], updates=updates)

    def save(self, path):
        self.model.save(path)

    def load_weights(self, path):
        self.model.load_weights(path)
