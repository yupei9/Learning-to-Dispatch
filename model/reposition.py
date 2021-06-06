import math
from abc import abstractmethod
from typing import Dict, List
import time
from dispatch import Dispatcher
from parse import HEX_GRID, RepositionData
import numpy as np

SPEED = 6 # 3 m/s @ 2 second interval

def timestamp2timegapindex(timestamp: int) -> int:
    '''
    时间戳转成time index
    @param timestamp:
    @return:
    '''
    time_local = time.localtime(timestamp)
    t_hour, t_min = time_local.tm_hour, time_local.tm_min
    return int(t_hour * 6 + t_min // 10)
    
class Repositioner:
    def __init__(self, dispatcher: Dispatcher, gamma: float):
        self.dispatcher = dispatcher
        self.gamma = gamma

    @abstractmethod
    def reposition(self, data: RepositionData) -> List[Dict[str, str]]:
        ...


class ScoredCandidate:
    def __init__(self, grid_id: str, score: float):
        self.grid_id = grid_id
        self.score = score

    def __repr__(self):
        return f'{self.grid_id}|{self.score}'


class StateValueGreedy(Repositioner):
    def reposition(self, data: RepositionData) -> List[Dict[str, str]]:
        # Rank candidates using Dispatcher state values
        candidate_grid_ids = []  # type: List[ScoredCandidate]
        for grid_id in self.dispatcher.get_grid_ids():
            value = self.dispatcher.student.predict([np.array([grid_id]),np.array([timestamp2timegapindex(data.timestamp)])])[0][0]
            candidate_grid_ids.append(ScoredCandidate(grid_id, value))

        # Rank discounted incremental gain
        reposition = []  # type: List[Dict[str, str]]
        for driver_id, current_grid_id in data.drivers:
            current_value = self.dispatcher.student.predict([np.array([grid_id]),np.array([timestamp2timegapindex(data.timestamp)])])[0][0]
            best_grid_id, best_value = current_grid_id, 0  # don't move for lower gain
            for grid_candidate in candidate_grid_ids:
                time = HEX_GRID.distance(current_grid_id, grid_candidate.grid_id) / SPEED
                distance_discount = math.pow(self.gamma, time)  
                proposed_value = self.dispatcher.student.predict([np.array([grid_id]),np.array([timestamp2timegapindex(data.timestamp)])])[0][0]
                incremental_value = distance_discount * proposed_value - current_value
                if incremental_value > best_value:
                    best_grid_id, best_value = grid_candidate.grid_id, incremental_value
            reposition.append(dict(driver_id=driver_id, destination=best_grid_id))
        return reposition