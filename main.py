import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib
from game import *
import config
import initial
from memory import Memory
from model import Residual_CNN,Gen_Model
import pickle

def main():
    env = Game()
    if initial.INITIAL_MEMORY_VERSION == None:
        memory =  Memory(config.MEMORY_SIZE)

    # load neural network
    current_NN = Residual_CNN(config.REG_CONST,config.LEARNING_RATE,(2,) + env.grid_shape, env.action_size, config.HIDDEN_CNN_LAYERS) #????
    best_NN = Residual_CNN(config.REG_CONST,config.LEARNING_RATE,(2,) + env.grid_shape, env.action_size, config.HIDDEN_CNN_LAYERS)

    best_player_version = 0
    best_NN.model.set_weights(current_NN.model.get_weights())

    #create players
    current_player = 
    best_player =
    iter = 0

    while 1:
        pass

if __name__ == '__main__':
    main()
