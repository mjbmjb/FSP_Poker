#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 00:08:11 2017

@author: mjb
"""
import torch
import random

import Settings.constants as constants
import Settings.game_settings as game_settings

from nn.dqn import *
from nn.table_sl import TableSL
from nn.net_sl import SLOptim

from Tree.game_state import GameState
from Tree.game_state import Action
from nn.sim_env import SimEnv

env = SimEnv()

class SixPlayerMachine:
    
    def __init__(self):
        self.dqn_optim = DQNOptim()
        self.net_sl = [SLOptim()] * game_settings.player_count
        
        
    
    
    def load_model(self, iter_time):
        iter_str = str(iter_time)
        # load rl model (only the net)
#        self.dqn_optim.model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '.rl'))
#        self.dqn_optim.target_net.load_state_dict(self.dqn_optim.model.state_dict())
        
        # load sl model
        for i in range(game_settings.player_count):
            self.net_sl[i].model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '_' + str(i) +'_' + '.sl'))
        
        
    # return int action['action:  ,'raise_amount':  ]
    def compute_action(self, state):
        
        # convert tensor for rl
        state_tensor = env.state2tensor(state)
        
        # !!!! the return action is a longTensor[[]]
#        action_id = (self.table_sl.select_action(state) if random.random() > arguments.eta \
#                 else self.dqn_optim.select_action(state_tensor))[0][0]
#        action_id = self.table_sl.select_action(state)[0][0]
        action_id = self.net_sl[state.current_player].select_action(state_tensor)[0][0]

        actions = env.get_vaild_action(state)
        
        if action_id >= len(actions):
            action_id = len(actions) - 1
        
        action = actions[action_id]
        
        return action