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
from nn.maddpg import MADDPG

from Tree.game_state import GameState
from Tree.game_state import Action
from nn.sim_env import SimEnv


env = SimEnv()

class SixPlayerMachine:
    
    def __init__(self):
        self.net_rl = [DQNOptim()] * game_settings.player_count
        self.net_sl = [SLOptim()] * game_settings.player_count
        self.maddpg = MADDPG()
        
    
    
    def load_model(self, iter_time):
        iter_str = str(iter_time)
        # load rl model (only the net)
#        self.dqn_optim.model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '.rl'))
#        self.dqn_optim.target_net.load_state_dict(self.dqn_optim.model.state_dict())

        for i in range(game_settings.player_count):
            # load maddpg
            if arguments.rl_model == 'maddpg':
                self.maddpg.load("../Data/Model/Iter:" + iter_str)
                self.maddpg.steps_done = iter_time * 100
                self.net_sl[i].model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '_' + str(0) +'_' + '.sl'))
            else:
                # TODO fix i
                self.net_sl[i].model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '_' + str(0) +'_' + '.sl'))
                self.net_rl[i].model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '_' + str(0) +'_' + '.rl'))
                self.net_rl[i].steps_done = self.net_rl[i].EPS_DECAY * 10


    # return int action['action:  ,'raise_amount':  ]
    def compute_action(self, state):
        
        # convert tensor for rl
        state_tensor = env.state2tensor(state)
        
        # !!!! the return action is a longTensor[[]]
#        action_id = (self.table_sl.select_action(state) if random.random() > arguments.eta \
#                 else self.dqn_optim.select_action(state_tensor))[0][0]
#        action_id = self.table_sl.select_action(state)[0][0]
        if arguments.rl_model == 'maddpg':
            action_id = self.maddpg.select_action(state.current_player, state_tensor)[0][0]
        else:
            action_id = self.net_sl[state.current_player].select_action(state_tensor)[0][0]

        actions = env.get_vaild_action(state)
        
        if action_id >= len(actions):
            action_id = len(actions) - 1
        
        action = actions[action_id]
        
        return action