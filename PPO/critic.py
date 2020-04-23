import numpy as np
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from keras.optimizers import Adam
from .agent import Agent
import keras

class Critic(Agent):
    """ Critic for the A3C Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr):
        Agent.__init__(self, inp_dim, out_dim, lr)
        # self.model = self.addHead(network)
        self.model = self.build_critic(inp_dim, lr)
        # self.discounted_r = K.placeholder(shape=(None,))
        # Pre-compile for threading
        # self.model._make_predict_function()

    def build_critic(self, env_dim, lr):
        HIDDEN_SIZE = 128
        NUM_LAYERS = 2
        state_input = Input(shape=(*env_dim,))

        x = Conv2D(kernel_size=(8,8),
                   strides=(4,4),
                   filters=32,
                   activation = 'relu',
                   padding = 'same',
                   kernel_regularizer=keras.regularizers.l2(0.01))(state_input)
        x = Conv2D(kernel_size=(4,4),
                   strides=(2,2),
                   filters=64,
                   activation = 'relu',
                   padding = 'same',
                   kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Conv2D(kernel_size=(3,3),
                   strides=(1,1),
                   filters=64,
                   activation = 'relu',
                   padding = 'same',
                   kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x) 

        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=lr), loss='mse')
        model.summary()

        return model

    # def addHead(self, network):
    #     """ Assemble Critic network to predict value of each state
    #     """
    #     x = Dense(128, activation='relu')(network.output)
    #     out = Dense(1, activation='linear')(x)
    #     return Model(network.input, out)

    # def optimizer(self):
    #     """ Critic Optimization: Mean Squared Error over discounted rewards
    #     """
    #     critic_loss = K.mean(K.square(self.discounted_r - self.model.output))
    #     updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], critic_loss)
    #     return K.function([self.model.input, self.discounted_r], [], updates=updates)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
