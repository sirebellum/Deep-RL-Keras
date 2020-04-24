
import sys
import numpy as np
import keras.backend as K
import keras

from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, LSTM, Lambda, CuDNNGRU, Dropout, Conv2D, MaxPooling2D, Activation
from keras.layers import BatchNormalization, Concatenate, ConvLSTM2D, MaxPooling3D
from keras.regularizers import l2
from utils.networks import conv_block

# import wandb
import wandb
from wandb.keras import WandbCallback

class Agent:
    """ Agent Class (Network) for DDQN
    """

    def __init__(self, state_dim, action_dim, lr, dueling, input_size, load):
        self.state_dim = state_dim
        self.input_size = input_size
        self.action_dim = action_dim
        self.tau = 0.001
        self.dueling = dueling
        # Initialize Deep Q-Network
        self.model = self.network(dueling)
        self.model.compile(RMSprop(lr, clipvalue=1.0), loss="mse")

        self.model.summary()

        if load is not None:
            # restore model
            fname = "model.h5"
            run_path = "joshherr/qualcomm/"+load

            api = wandb.Api()
            run = api.run(run_path)
            local_path = None
            with run.file(fname).download(replace=True) as f:
              local_path = f.name

            self.model.load_weights(local_path)

        # Build target Q-Network
        self.target_model = self.network(dueling)
        self.target_model.compile(RMSprop(lr), loss="mse")
        self.target_model.set_weights(self.model.get_weights())

    def network(self, dueling):
        """ Build Deep Q-Network
        """
        # Break input into s images for each history window
        image = Input((*self.input_size, self.state_dim[0]))
        inputs = []
        for s in range(self.state_dim[0]):
          inputs.append(Reshape((*self.input_size, 1)) \
                          (Lambda(lambda x: x[:,:,:,s])(image)))
        
        ### Create feature extraction network for each history image
        outs = []
        for inp in inputs:
          x = Conv2D(kernel_size=(3,3),
                     strides=(1,1),
                     filters=8,
                     activation = 'relu',
                     padding = 'same',
                     kernel_regularizer=keras.regularizers.l2(0.01))(inp)
          x = Conv2D(kernel_size=(3,3),
                     strides=(1,1),
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

          x = Conv2D(kernel_size=(3,3),
                     strides=(1,1),
                     filters=16,
                     activation = 'relu',
                     padding = 'same',
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
          x = Conv2D(kernel_size=(3,3),
                     strides=(1,1),
                     filters=32,
                     activation = 'relu',
                     padding = 'same',
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
          x = Conv2D(kernel_size=(3,3),
                     strides=(1,1),
                     filters=32,
                     activation = 'relu',
                     padding = 'same',
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
          x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

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

          outs.append(Reshape((self.input_size[0]//2//2//2,
                               self.input_size[0]//2//2//2,-1))(x))

        ### Concatenate history network outputs
        x = Concatenate(axis=-1)(outs)
        x = Conv2D(kernel_size=(3,3),
                   strides=(1,1),
                   filters=64,
                   activation = 'relu',
                   padding = 'same',
                   kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = Conv2D(kernel_size=(3,3),
                   strides=(1,1),
                   filters=128,
                   activation = 'relu',
                   padding = 'same',
                   kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = Conv2D(kernel_size=(3,3),
                   strides=(1,1),
                   filters=128,
                   activation = 'relu',
                   padding = 'same',
                   kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)        

        x = Flatten()(x)
        x = Dense(1024, activation="relu",
                  kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = Dense(1024, activation="relu",
                  kernel_regularizer=keras.regularizers.l2(0.01))(x)
        ### Action decision
        if(dueling):
            # Have the network estimate the Advantage function as an intermediate layer
            x = Dense(self.action_dim + 1, activation='linear')(x)
            x = Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True), output_shape=(self.action_dim,))(x)
        else:
            x = Dense(self.action_dim, activation='linear')(x)

        return Model(image, x)

    def transfer_weights(self):
        """ Transfer Weights from Model to Target at rate Tau
        """
        W = self.model.get_weights()
        tgt_W = self.target_model.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.target_model.set_weights(tgt_W)

    def fit(self, inp, targ, record=False):
        """ Perform one epoch of training
        """
        # Add noise for training
        inp = inp.astype(np.float32)
        inp += np.random.normal(0,0.01,inp.shape)

        if record:
          self.model.fit(self.reshape(inp), targ, epochs=1, verbose=0,
                         callbacks=[WandbCallback(save_model=False)])
        else:
          self.model.fit(self.reshape(inp), targ, epochs=1, verbose=0)

    def predict(self, inp):
        """ Q-Value Prediction
        """
        inp = inp.astype(np.float32)
        return self.model.predict(self.reshape(inp))

    def target_predict(self, inp):
        """ Q-Value Prediction (using target network)
        """
        inp = inp.astype(np.float32)
        return self.target_model.predict(self.reshape(inp))

    def reshape(self, x):
        if len(x.shape) < 4 and len(self.state_dim) > 2: return np.expand_dims(x, axis=0)
        elif len(x.shape) < 3: return np.expand_dims(x, axis=0)
        else: return x

    def save(self, path):
        self.model.save(path)

    def load_weights(self, path):
        self.model.load_weights(path)
