# this is the rule of the game
# game: connect4

import numpy as np


class Game:

    def __init__(self):
        self.name = "connect"
        self.inputshape = (2,6,7) # two player
        self.gridshape = (6,7)
        self.actionSpace = np.array(np.zeros(42),dtype=np.int)
        self.gameState = GameState(np.array(np.zeros(42),dtype=np.int),1)
        self.pieces = {'1':'X', '0': '-', '-1':'O'}
        self.state_size = len(self.gameState.binary)
        self.action_size = len(self.actionSpace)
        self.currentPlayer = 1

    def initialize(self):
        self.gameState = GameState(np.array(np.zeros(42),dtype=np.int),1)

        return self.gameState

    def step(self,action):
        next_state, value, done = self.gameState.takeAction(action)
        self.gameState = next_state
        self.currentPlayer = -self.currentPlayer
        info = None
        return ((next_state, value, done, info))

    def identities(self,state,actionValues): # solve identity state
        identities = [(state, actionValues)]
        currentboard = state.board
        currentAV = actionValues
        newboard = []
        newAV = []
        for i in range(6):
            tmp = []
            tmp2 = []
            for j in range(7):
                tmp.append(currentboard[j+i*7])
                tmp2.append(currentAV[j+i*7])
            tmp.reverse()
            tmp2.reverse()
            for j in range(7):
                newboard.append(tmp[j])
                newAV.append(tmp2[j])
        identities.append((GameState(newboard,state.playerTurn),newAV))
        return identities


class GameState:
    def __init__(self, board, playerTurn):
        self.board = board
        self.pieces = {'1':'X', '0': '-', '-1':'O'}
        self.playerTurn = playerTurn
        self.binary = self._binary()
        self.id = self._convertToId()
        self.allowedActions = self._allowAction()
        self.isEndGame = self.checkEnd()
        self.value = self._getvalue()
        self.score = self._getscore()
        self.winners = [  # to judge if someone wins
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [7, 8, 9, 10],
            [8, 9, 10, 11],
            [9, 10, 11, 12],
            [10, 11, 12, 13],
            [14, 15, 16, 17],
            [15, 16, 17, 18],
            [16, 17, 18, 19],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
            [22, 23, 24, 25],
            [23, 24, 25, 26],
            [24, 25, 26, 27],
            [28, 29, 30, 31],
            [29, 30, 31, 32],
            [30, 31, 32, 33],
            [31, 32, 33, 34],
            [35, 36, 37, 38],
            [36, 37, 38, 39],
            [37, 38, 39, 40],
            [38, 39, 40, 41],

            [0, 7, 14, 21],
            [7, 14, 21, 28],
            [14, 21, 28, 35],
            [1, 8, 15, 22],
            [8, 15, 22, 29],
            [15, 22, 29, 36],
            [2, 9, 16, 23],
            [9, 16, 23, 30],
            [16, 23, 30, 37],
            [3, 10, 17, 24],
            [10, 17, 24, 31],
            [17, 24, 31, 38],
            [4, 11, 18, 25],
            [11, 18, 25, 32],
            [18, 25, 32, 39],
            [5, 12, 19, 26],
            [12, 19, 26, 33],
            [19, 26, 33, 40],
            [6, 13, 20, 27],
            [13, 20, 27, 34],
            [20, 27, 34, 41],

            [3, 9, 15, 21],
            [4, 10, 16, 22],
            [10, 16, 22, 28],
            [5, 11, 17, 23],
            [11, 17, 23, 29],
            [17, 23, 29, 35],
            [6, 12, 18, 24],
            [12, 18, 24, 30],
            [18, 24, 30, 36],
            [13, 19, 25, 31],
            [19, 25, 31, 37],
            [20, 26, 32, 38],

            [3, 11, 19, 27],
            [2, 10, 18, 26],
            [10, 18, 26, 34],
            [1, 9, 17, 25],
            [9, 17, 25, 33],
            [17, 25, 33, 41],
            [0, 8, 16, 24],
            [8, 16, 24, 32],
            [16, 24, 32, 40],
            [7, 15, 23, 31],
            [15, 23, 31, 39],
            [14, 22, 30, 38],
        ]

    def takeAction(self,action):  # take an action
        newboard = np.array(self.board)
        newboard[action] = self.playerTurn # build a new board and add new action
        # then generate the new state
        newState = GameState(newboard,-self.playerTurn) # convert to another player

        value = 0
        done = False

        if newState.isEndGame: # check if it's the end state
            done = True
            value = newState.value[0]

        return (newState,value,done)



    def checkEnd(self):
        if (self.board.sum() == 42): # it's full
            return True

        for x,y,z,a in self.winners: # the other player wins
            if self.board[x]+self.board[y]+self.board[z]+self.board[a] == -4*self.playerTurn:
                return True
        return False


    def _binary(self): # convert a state to a binary list
        first_posision = np.ndarray(np.zeros(len(self.board)),dtype=np.int)
        first_posision[self.board == self.playerTurn] = 1

        second_posision = np.ndarray(np.zeros(len(self.board)), dtype=np.int)
        second_posision[self.board == -self.playerTurn] = 1 # other player

        total_posision = np.append(first_posision,second_posision)

        return total_posision

    def _convertToId(self):  # Id each unique state
        player1_position = np.zeros(len(self.board), dtype=np.int)
        player1_position[self.board == 1] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -1] = 1

        position = np.append(player1_position, other_position)

        id = ''.join(map(str, position))

        return id

    def _allowAction(self):
        allow = []
        for i in range(7):
            for j in range(6):
                if self.board[j+6*i] != 0:
                    break
                if self.board[j+6*i] == 0:
                    allow.append(j+6*i)

        return allow
        # for i in range(len(self.board)):
        #     if i >= len(self.board) -7:
        #         if self.board[i] == 0: # no one is here
        #             allow.append(i)
        #     else:
        #         if self.board[i] == 0 && self.board[i+7] != 0:
        #             allow.append(i)
        #
        # return allow

    def _getvalue(self):
        for x, y, z, a in self.winners:
            if (self.board[x] + self.board[y] + self.board[z] + self.board[a]
                    == 4 * -self.playerTurn):
                return (-1, -1, 1) # second is current player, third is the other one
        return (0, 0, 0)

    def _getscore(self):
        tmp = self._getvalue()
        return (tmp[1],tmp[2])

