import numpy as np
import random

import MTTS as mc
import time

from game import GameState
import tensorflow as tf
import keras

import config

class User:
    def __init__(self,name,state_size,action_size):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size

    def act(self):
        action = input('Enter your chosen action: ')
        pi = np.zeros(self.action_size)
        pi[action] = 1
        value = None
        NN_value = None
        return (action, pi, value, NN_value)




class Agent:
    def __init__(self,name,state_size,action_size,model):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.model = model

    def simulate(self):
        




