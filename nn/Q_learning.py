#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 05:39:58 2017

@author: mjb
"""

"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
from collections import namedtuple
import Settings.arguments as arguments
import Settings.constants as constants
import Settings.game_settings as game_settings
import random

class QLearning:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.iter_count = 0
        self.q_table = pd.DataFrame(np.zeros((30000,5)),columns=self.actions)
        self.memory = ReplayMemory(40000)

    def select_action(self, observation):
        self.check_state_exist(observation)
        temperature = 1 / (1 + 0.02 * np.sqrt(self.iter_count))
        # action selection
        state_action = self.q_table.ix[observation, :].apply(lambda q : np.exp(q/temperature))
        state_action = state_action.div(state_action.sum())
        action = np.random.uniform()
        for i in range(game_settings.actions_count):
            action = action - state_action.ix[i]
            if action <= 0:
                return arguments.LongTensor([[i]])

        return action

    def batch_learn(self):
        batch = self.memory.sample(30)
        for i in range(len(batch)):
            s, a, s_, r = batch[i]
            self.check_state_exist(s)
            if s_:
                self.check_state_exist(s_)
            q_predict = self.q_table.ix[s, a]
            if s > 0:
                q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
            else:
                q_target = r  # next state is terminal
            self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update
            
    
    def learn(self,s,a,s_,r):
        q_predict = self.q_table.ix[s, a]
        if s_ > 0:
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update



    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = len(self.memory) if batch_size > len(self.memory) else batch_size
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
 