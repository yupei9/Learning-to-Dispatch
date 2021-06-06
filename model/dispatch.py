import collections
import csv
import math
import os
import random
from abc import abstractmethod
from typing import Dict, List, Set, Tuple
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from parse import DispatchCandidate, Driver, HEX_GRID, Request
from keras.models import load_model
import time
import numpy as np

CANCEL_DISTANCE_FIT = lambda x: 0.02880619 * math.exp(0.00075371 * x)
STEP_SECONDS = 2

def timestamp2timegapindex(timestamp: int) -> int:
    '''
    时间戳转成time index
    @param timestamp:
    @return:
    '''
    time_local = time.localtime(timestamp)
    t_hour, t_min = time_local.tm_hour, time_local.tm_min
    return int(t_hour * 6 + t_min // 10)

class Dispatcher:
    def __init__(self, alpha, gamma, idle_reward):
        self.alpha = alpha
        self.gamma = gamma
        self.idle_reward = idle_reward

    @staticmethod 
    def _init_state_values() -> Dict[str, float]:
        grid_dict = collections.defaultdict(float)
        grid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dict.pkl')
        cur_dict = pickle.load(open(grid_path, "rb"))
        for grid_str in cur_dict.keys():
            grid_dict[grid_str] = cur_dict[grid_str]
        return grid_dict

    @abstractmethod
    def dispatch(self, drivers: Dict[str, Driver], requests: Dict[str, Request],
                 candidates: Dict[str, Set[DispatchCandidate]]) -> Dict[str, DispatchCandidate]:
        ...

    @abstractmethod
    def get_grid_ids(self) -> Set[str]:
        ...

    @abstractmethod
    def state_value(self, grid_id: str) -> float:
        ...

    @abstractmethod
    def update_state_value(self, grid_id: str, delta: float) -> None:
        ...


class ScoredCandidate:
    def __init__(self, candidate: DispatchCandidate, score: float):
        self.candidate = candidate
        self.score = score

    def __repr__(self):
        return f'{self.candidate}|{self.score}'


class Sarsa(Dispatcher):
    def __init__(self, alpha, gamma, idle_reward):
        super().__init__(alpha, gamma, idle_reward)
        # Expected gain from each driver in (location)
        self.state_values = Dispatcher._init_state_values()

    def dispatch(self, drivers: Dict[str, Driver], requests: Dict[str, Request],
                 candidates: Dict[str, Set[DispatchCandidate]]) -> Dict[str, DispatchCandidate]:
        # Rank candidates based on incremental driver value improvement
        ranking = []  # type: List[ScoredCandidate]
        order_drivers = dict()
        for candidate in set(c for cs in candidates.values() for c in cs):  # type: DispatchCandidate
            request = requests[candidate.request_id]
            driver = drivers[candidate.driver_id]
            if candidate.request_id not in order_drivers:
                order_drivers[candidate.request_id] = set()
            order_drivers[candidate.request_id].add(candidate.driver_id)
            v0 = self.state_value(driver.location)  # Value of the driver current position
            v1 = self.state_value(request.end_loc)  # Value of the proposed new position
            expected_reward = completion_rate(candidate.distance) * request.reward
            if expected_reward > 0:
                # Best incremental improvement (get the ride AND improve driver position)
                update = expected_reward + self.gamma * v1 - v0
                ranking.append(ScoredCandidate(candidate, update))
        
        order_driver_num = collections.defaultdict(float)
        for order_id in order_drivers.keys():
            order_driver_num[order_id] = len(order_drivers[order_id])
            
        # Assign drivers
        assigned_driver_ids = set()  # type: Set[str]
        dispatch = dict()  # type: Dict[str, DispatchCandidate]
        ranking = sorted(ranking,key=lambda  x : x.score, reverse = True)
        for scored in sorted(ranking, key=lambda x: (x.score, -order_driver_num[x.candidate.request_id]), reverse=True):# type: ScoredCandidate
            candidate = scored.candidate
            if candidate.request_id in dispatch or candidate.driver_id in assigned_driver_ids:
                continue
            assigned_driver_ids.add(candidate.driver_id)
            request = requests[candidate.request_id]
            dispatch[request.request_id] = candidate

            # Update value at driver location
            driver = drivers[candidate.driver_id]
            self.update_state_value(driver.location, self.alpha * scored.score)

        # Reward (negative) for idle driver positions
        for driver in drivers.values():
            if driver.driver_id in assigned_driver_ids:
                continue
            v0 = self.state_value(driver.location)
            # TODO: idle transition probabilities Expected SARSA
            v1 = self.state_value(driver.location)  # Assume driver hasn't moved if idle
            update = self.idle_reward + self.gamma * v1 - v0
            self.update_state_value(driver.location, self.alpha * update)

        return dispatch

    def get_grid_ids(self) -> Set[str]:
        return set(self.state_values.keys())

    def state_value(self, grid_id: str) -> float:
        return self.state_values[grid_id]

    def update_state_value(self, grid_id: str, delta: float) -> None:
        self.state_values[grid_id] += delta


class Dql(Dispatcher):
    def __init__(self, alpha, gamma, idle_reward):
        super().__init__(alpha, gamma, idle_reward)
        # self.student = Dispatcher._init_state_values()
        # self.teacher = Dispatcher._init_state_values()
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deep_value.h5')
        self.dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dict.pkl')
        
        self.teacher = load_model(self.model_path)
        self.student = load_model(self.model_path)       
        self.grid_dict = self._init_state_values()
        self.timestamp = 0
        self.Batch_grid = []
        self.Batch_time = []
        self.Batch_value = []

    def dispatch(self, drivers: Dict[str, Driver], requests: Dict[str, Request],
                 candidates: Dict[str, Set[DispatchCandidate]]) -> Dict[str, DispatchCandidate]:
        #  Flip a coin
        if random.random() < 0.5:
            self.student, self.teacher = self.teacher, self.student
        # Rank candidates
        updates = dict()  # type: Dict[Tuple[str, str], ScoredCandidate]
        ranking = []  # type: List[ScoredCandidate]
        for candidate in set(c for cs in candidates.values() for c in cs):  # type: DispatchCandidate
            # Teacher provides the destination position value

            request = requests[candidate.request_id]
            self.timestamp = max(request.request_ts, self.timestamp)
            self.timestamp = timestamp2timegapindex(self.timestamp)
            #-----------------------
            space = self.grid_dict[request.end_loc]
            v1 = self.teacher.predict([np.array([space]),np.array([self.timestamp])])[0][0]
            #-----------------------
            # v1 = self.teacher[request.end_loc]

            # Compute student update

            driver = drivers[candidate.driver_id]
            #-----------------------
            space = driver.location
            # time = timestamp2timegapindex(request.request_ts)
            v0 = self.student.predict([np.array([self.grid_dict[space]]),np.array([self.timestamp])])[0][0]
            #-----------------------
            # v0 = self.student[driver.location]
            expected_reward = completion_rate(candidate.distance) * request.reward
            update = expected_reward + self.gamma * v1 - v0
            updates[(candidate.request_id, candidate.driver_id)] = ScoredCandidate(candidate, update)

            # Joint Ranking for actual driver assignment
            v1 = self.state_value(self.grid_dict[request.end_loc], self.timestamp) # teacher's value + student's value
            expected_gain = expected_reward + self.gamma * v1
            ranking.append(ScoredCandidate(candidate, expected_gain))

        # Assign drivers
        assigned_driver_ids = set()  # type: Set[str]
        dispatch = dict()  # type: Dict[str, DispatchCandidate]
        for scored in sorted(ranking, key=lambda x: x.score, reverse=True):  # type: ScoredCandidate
            candidate = scored.candidate
            if candidate.request_id in dispatch or candidate.driver_id in assigned_driver_ids:
                continue
            assigned_driver_ids.add(candidate.driver_id)

            request = requests[candidate.request_id]
            driver = drivers[candidate.driver_id]
            dispatch[request.request_id] = candidate

        
        return dispatch

    def get_grid_ids(self) -> Set[str]:
        return self.grid_dict.values()

    def state_value(self, grid_id: str, time) -> float:
        return self.student.predict([np.array([grid_id]),np.array([time])])[0][0] + self.teacher.predict([np.array([grid_id]),np.array([time])])[0][0]

    def update_state_value(self) -> None:
        self.student.fit([np.array(self.Batch_grid),np.array(self.Batch_time)],np.array(self.Batch_value),batch_size = 32)
        self.Batch_grid = []
        self.Batch_time = []
        self.Batch_value = []

def completion_rate(distance_meters: float) -> float:
    return 1 - max(min(CANCEL_DISTANCE_FIT(distance_meters), 1), 0)