#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:37:03 2017

@author: mjb
"""
import sys
sys.path.append('/home/mjb/Nutstore/deepStack/')

import torch

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

iter_str = str(10000)

net_sl = [SLOptim()] * game_settings.player_count
net_rl = [DQNOptim()] * game_settings.player_count

for i in range(game_settings.player_count):
    net_sl[i].model.load_state_dict(torch.load(arguments.WORK_PATH+'/Data/Model/Iter:' + iter_str + '_' + str(i) +'_' + '.sl'))
    net_sl[i].model.eval()
    net_rl[i].model.load_state_dict(torch.load(arguments.WORK_PATH+'/Data/Model/Iter:' + iter_str + '_' + str(i) +'_' + '.rl'))
    net_rl[i].model.eval()
state = GameState()
call = Action(atype=constants.actions.ccall,amount=0)
rrasie = Action(atype=constants.actions.rraise,amount=20000)
fold = Action(atype=constants.actions.fold,amount=0)

hole = torch.LongTensor([[40,41],[50,51],[4,5],[8,9],[44,45],[48,49]])
#board = torch.LongTensor([6,30,31,38,43])
board = torch.LongTensor([6,30,31,39,43])

state.bets = arguments.LongTensor([10000,10000,10000,10000,10000,10000])
state.street = 3
state.current_player = 0

state.hole = hole
state.board = board

state.do_action(rrasie)
state.do_action(fold)
state.do_action(fold)
state.do_action(fold)
state.do_action(call)
#state.do_action(call)
state.street = 2


#print(state.bets)

env = SimEnv()

#print(state.hole)
#print(state.board)
#print(state.get_terminal_value() )
#print(state.board)
state_tensor = env.state2tensor(state)
#state_tensor = torch.randn(1,136)

#print('rl___________')
#for i in range(game_settings.player_count):
#    print(net_sl[i].model(Variable(state_tensor)))
#    print("Action:" + str(net_sl[i].select_action(state_tensor)))

net_sl_test = SLOptim()
action = arguments.LongTensor(1).fill_(1)
for i in range(10000):
    net_sl_test.memory.push(state_tensor,action)
for i in range(1000):
    net_sl_test.optimize_model()
#    net_sl_test.plot_error_vis(i)
net_sl_test.model.eval()
print(torch.exp(net_sl_test.model(Variable(state_tensor)).data))

hole = torch.LongTensor([[0,1],[4,5],[8,9]])
board = torch.LongTensor([6,30,31,38,43])

params = net_sl_test.model.state_dict()
weight = params['fc3.weight'].cpu().numpy()
