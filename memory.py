import config
import numpy as np
from collections import deque


class Memory:
    def __init__(self):
        self.MEMORY_SIZE = config.MEMORY_SIZE
        self.ltmemory = deque(maxlen=config.MEMORY_SIZE)  # long term memory
        self.stmemory = deque(maxlen=config.MEMORY_SIZE)  # short term memory, to store data when a game is not ending


    def commit_stmemory(self,identities,state,actionValues):
        for r in identities(state,actionValues):
            self.stmemory.append({
                "board":r[0].board,
                "state": r[0],
                "id": r[0].id,
                "AV": r[1],
                "playerTurn":r[0].playerTurn
            })

    def commit_ltmemory(self):
        for i in self.stmemory:
            self.ltmemory.append(i)
        self.clear_stmemory()

    def clear_stmemory(self):
        self.stmemory = deque(maxlen=config.MEMORY_SIZE)


