import numpy as np
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Reshape, Concatenate, Lambda
from keras.optimizers import Adam
from .agent import Agent
from .ppo_loss import proximal_policy_optimization_loss
import keras

class Actor(Agent):
    """ Actor for the A2C Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr, loss_clipping, entropy_loss):
        Agent.__init__(self, inp_dim, out_dim, lr)
        self.model = self.build_actor(inp_dim, out_dim, lr, loss_clipping, entropy_loss)
        # self.model = self.addHead(network)
        # self.action_pl = K.placeholder(shape=(None, self.out_dim))
        # self.advantages_pl = K.placeholder(shape=(None,))
        # # Pre-compile for threading
        # self.model._make_predict_function()


    def build_actor(self, env_dim, act_dim, lr, loss_clipping, entropy_loss):
        HIDDEN_SIZE = 128
        NUM_LAYERS = 2
        state_input = Input(shape=(*env_dim,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(act_dim,))

        inputs = []
        for s in range(env_dim[-1]):
          inputs.append(Reshape((env_dim[0], env_dim[1], 1)) \
                          (Lambda(lambda x: x[:,:,:,s])(state_input)))
        
        ### Create feature extraction network for each history image
        outs = []
        for inp in inputs:
          x = Conv2D(kernel_size=(3,3),
                     strides=(1,1),
                     filters=8,
                     activation = 'relu',
                     padding = 'same',
                     kernel_regularizer=keras.regularizers.l2(0.01))(inp)
          x = BatchNormalization()(x)
          x = Conv2D(kernel_size=(3,3),
                     strides=(1,1),
                     filters=16,
                     activation = 'relu',
                     padding = 'same',
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
          x = BatchNormalization()(x)
          x = Conv2D(kernel_size=(3,3),
                     strides=(1,1),
                     filters=16,
                     activation = 'relu',
                     padding = 'same',
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
          x = BatchNormalization()(x)
          x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

          x = Conv2D(kernel_size=(3,3),
                     strides=(1,1),
                     filters=16,
                     activation = 'relu',
                     padding = 'same',
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
          x = BatchNormalization()(x)
          x = Conv2D(kernel_size=(3,3),
                     strides=(1,1),
                     filters=32,
                     activation = 'relu',
                     padding = 'same',
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
          x = BatchNormalization()(x)
          x = Conv2D(kernel_size=(3,3),
                     strides=(1,1),
                     filters=32,
                     activation = 'relu',
                     padding = 'same',
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
          x = BatchNormalization()(x)
          x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

          x = Conv2D(kernel_size=(3,3),
                     strides=(1,1),
                     filters=32,
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
          x = Conv2D(kernel_size=(3,3),
                     strides=(1,1),
                     filters=64,
                     activation = 'relu',
                     padding = 'same',
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
          x = BatchNormalization()(x)
          x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

          outs.append(Reshape((env_dim[0]//2//2//2,
                               env_dim[1]//2//2//2,-1))(x))

        ### Concatenate history network outputs
        x = Concatenate(axis=-1)(outs)
        x = Conv2D(kernel_size=(3,3),
                   strides=(1,1),
                   filters=64,
                   activation = 'relu',
                   padding = 'same',
                   kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Conv2D(kernel_size=(3,3),
                   strides=(1,1),
                   filters=128,
                   activation = 'relu',
                   padding = 'same',
                   kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Conv2D(kernel_size=(3,3),
                   strides=(1,1),
                   filters=128,
                   activation = 'relu',
                   padding = 'same',
                   kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)        

        x = Flatten()(x)
        x = Dense(1024, activation="relu",
                  kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = Dense(1024, activation="relu",
                  kernel_regularizer=keras.regularizers.l2(0.01))(x)

        out_actions = Dense(act_dim, activation='softmax', name='output')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=lr),
                      loss=[
                          proximal_policy_optimization_loss(
                            advantage=advantage,
                            old_prediction=old_prediction,
                            loss_clipping=loss_clipping,
                            entropy_loss=entropy_loss
                          )
                        ]
                    )
        model.summary()

        return model

    # def addHead(self, network):
    #     """ Assemble Actor network to predict probability of each action
    #     """
    #     x = Dense(128, activation='relu')(network.output)
    #     out = Dense(self.out_dim, activation='softmax')(x)
    #     return Model(network.input, out)

    # def optimizer(self):
    #     """ Actor Optimization: Advantages + Entropy term to encourage exploration
    #     (Cf. https://arxiv.org/abs/1602.01783)
    #     """
    #     weighted_actions = K.sum(self.action_pl * self.model.output, axis=1)
    #     eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(self.advantages_pl)
    #     entropy = K.sum(self.model.output * K.log(self.model.output + 1e-10), axis=1)
    #     loss = 0.001 * entropy - K.sum(eligibility)

    #     updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], loss)
    #     return K.function([self.model.input, self.action_pl, self.advantages_pl], [], updates=updates)

    def save(self, path):
        self.model.save(path)

    def load_weights(self, path):
        self.model.load_weights(path)
