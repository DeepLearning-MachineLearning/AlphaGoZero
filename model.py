import config
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras.optimizers import SGD
from keras import regularizers
from keras.optimizers import SGD
# from loss import softmax_cross_entropy_with_logits

import keras.backend as K

class Gen_Model:
    def __init__(self, reg_const, lr, input_dim, output_dim):
        self.reg_const = reg_const
        self.lr = lr
        self.input_dim = input_dim
        self.output_dim = output_dim

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split=validation_split,
                              batch_size=batch_size)



class Residual_CNN(Gen_Model):

    def __init__(self,lr,reg_const,hidden_layer,input_dim,output_dim):
        Gen_Model.__init__(reg_const,lr,input_dim,output_dim)
        self.hidden_layer = hidden_layer
        self.num_layers = len(hidden_layer)
        self.model = self.build_model()

    def residual_layer(self,input_block,filters,kernel_size):
        x = self.conv_layer(input_block,filters,kernel_size)

        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            data_format="channels_first",
            padding="same",
            use_bias=False,
            activation="relu",
            kernel_regularizer=regularizers.l2(self.reg_const),
        )(x)
        x = BatchNormalization(axis=1)(x)

        x = add([input_block,x])(x)

        x = LeakyReLU()(x)

        return (x)

    def conv_layer(self,x,filters, kernel_size):
        x = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            data_format = "channels_first",
            padding = "same",
            activation="relu",
            kernel_regularizer= regularizers.l2(self.reg_const),
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        return (x)

    def _build_model(self):

        model_input = Input(shape = self.input_dim,name = "main_input")
        # first build a conv layer
        x = self.conv_layer(model_input,self.hidden_layer[0]["filters"],self.hidden_layer[0]["kernel_size"])
        # then build some residual layers
        if len(self.hidden_layer)>1:
            for h in self.hidden_layer[1:]:
                x = self.residual_layer(model_input,self.hidden_layer[0]["filters"],self.hidden_layer[0]["kernel_size"])


        vh = self.value_head(x)
        ph = self.policy_head(x)
        model = Model(inputs=model_input,outputs=[vh,ph] )
        model.compile(
                      loss={"value_head":'mean_squared_error','policy_head':tf.nn.softmax_cross_entropy_with_logits},
                      optimizer=SGD(lr=self.lr,momentum=config.MOMENTUM),
                      loss_weights={"overhead":0.5,"policy_head":0.5}
                      )

        return model

    def value_head(self,x):  # input:(batch,75,6,7)

        x = Conv2D( # make input to be (batch, 1, 6, 7)
            filters = 1,
            kernel_size = (1,1),
            data_format = "channels_first",
            padding = "same",
            use_bias= False,
            activation="relu",
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)
        #norm and activation
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        # flat to (batch, 42)
        x = Flatten()(x)

        # conver to (batch, 20)
        x = Dense(
            20,   #output size
            use_bias=False,
            activation="linear",
            kernel_regularizer= regularizers.l2(self.reg_const),
        )(x)

        x = LeakyReLU()(x) #activation

        x = Dense(  # convert to (batch, 1)
            self.output_dim,
            use_bias=False,
            activation="relu",
            kernel_regularizer= regularizers.l2(self.reg_const),
            name = "value_head"
        )(x)


        return (x)

    def policy_head(self,x):  # input:(batch, 75, 6 ,7)
        x = Conv2D( #convert to (batch, 2, 6, 7) because there are two filters
            filters=2,
            kernel_size=(1, 1),
            data_format="channels_first",
            padding="same",
            use_bias=False,
            activation="linear",
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x) # flat to (batch, 84)

        x = Dense( # fully connect layer make it (batch, 42), which is 42 possible action logits
            self.output_dim,  # output size
            use_bias=False,
            activation="linear",
            kernel_regularizer=regularizers.l2(self.reg_const),
            name="policy_head"
        )(x)

        return (x)