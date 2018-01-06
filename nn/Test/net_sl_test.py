#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 05:16:45 2017

@author: mjb
"""

import sys
sys.path.append('../')

import torch
import numpy as np

import Settings.arguments as arguments
import Settings.game_settings as game_settings
import Settings.constants as constants
from Player.six_player_machine import SixPlayerMachine
from ACPC.six_acpc_game import SixACPCGame
from Tree.game_state import GameState
from Tree.game_state import Action
from nn.sim_env import SimEnv

from nn.dqn import *
from nn.table_sl import TableSL
from nn.net_sl import SLOptim


iter_str = str(2000)

net_sl = SLOptim()

state = GameState()
call = Action(atype=constants.actions.ccall,amount=0)
rrasie = Action(atype=constants.actions.rraise,amount=1000)
fold = Action(atype=constants.actions.fold,amount=0)
    
state.do_action(call)
state.do_action(rrasie)
state.do_action(call)
state.do_action(fold)
#state.do_action(fold)
#state.do_action(call)
for i in range(21):
    state.do_action(rrasie)
print(state.bets)

env = SimEnv()

state_tensor = env.state2tensor(state)

test_memory = np.load(arguments.WORK_PATH+'/nn/Test/sl_memory_test.npy')
net_sl.memory.memory = test_memory.tolist()

for i in range(10000):
    net_sl.optimize_model()
    if i % 100 == 0:
        print("episod:"+str(i))
        print(torch.exp(net_sl.model(Variable(state_tensor)))) 
