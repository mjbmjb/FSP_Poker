#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:37:03 2017

@author: mjb
"""
import sys
sys.path.append('../../')

import torch
import numpy as np
import Settings.arguments as arguments
import Settings.game_settings as game_settings
import Settings.constants as constants
# from Player.six_player_machine import SixPlayerMachine
# from ACPC.six_acpc_game import SixACPCGame
from Tree.game_state import GameState
from Tree.game_state import Action
from nn.sim_env import SimEnv

from nn.dqn import *
from nn.table_sl import TableSL
from nn.net_sl import SLOptim

iter_str = str(100000)

net_sl = SLOptim()
net_rl = DQNOptim()

net_sl.model.load_state_dict(torch.load(arguments.WORK_PATH+'/Data/Model/Iter:' + iter_str + '_' + str(0) +'_' + '.sl'))
net_sl.model.eval()
net_rl.model.load_state_dict(torch.load(arguments.WORK_PATH+'/Data/Model/Iter:' + iter_str + '_' + str(0) +'_' + '.rl'))
net_rl.model.eval()

state = GameState()
call = Action(atype=constants.actions.ccall,amount=0)
rrasie = Action(atype=constants.actions.rraise,amount=1000)
fold = Action(atype=constants.actions.fold,amount=0)

env = SimEnv()


def make_data(size=10000):
    cat = []
    data = []

    for _ in range(size):
        state.street = 1
        state.current_player = 0

        state.hole = torch.LongTensor(6,1).fill_(0)
        state.hole[state.current_player][0] = np.random.randint(5)
        # board = torch.LongTensor([6,30,31,38,43])
        state.board = torch.LongTensor([6])

        state.bets = arguments.LongTensor(np.random.randint(arguments.stack, size=6))
        state.street = 1
        state.current_player = 0

        state.terminal = True
        state_tensor = env.state2tensor(state)

        cat.append((state.hole[state.current_player].item(),state.bets.clone()))
        data.append(state_tensor)

    return cat, data

cat, data = make_data(10000)
hole_tar, bets_tar = list(zip(*cat))
#print(state.bets)

forward_data = []

for state_tensor in data:
    forward_data.append(net_sl.model.forward_fc(Variable(state_tensor)).data)

forward_data = np.vstack(forward_data)
