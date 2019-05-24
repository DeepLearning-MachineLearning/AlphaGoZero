
import numpy as np
import copy

def softmax(x):
    p = np.exp(x - np.max(x))
    p = p/np.sum(p)

    return p

class Node():
    def __init__(self, state):
        # self._parent = parent
        # self._children = {}
        # self._Q = 0
        # self._u = 0
        # self._n_visitors = 0
        # self._P = prior
        self.state = state
        self.playerTurn = state.playerTurn
        self.id = state.id
        self.edges = []

    def is_leaf(self):
        if len(self.edges) > 0:
            return False
        else:
            return True

class Edges():
    def __init__(self,innode,outnode,prior,action):
        self.innode = innode
        self.outnode = outnode
        self.prior = prior
        self.action = action
        self.playerTurn = innode.state.playerTurn
        self.id = innode.id + "|" + outnode.id  # change a little bit
        self.stats = {
            "_Q" : 0,
            "_U" : 0,
            "_n_visitors" : 0,
            "_P" : prior,
        }


class MTCT():
    def __init__(self,root,cpcut):
        self.root = root





