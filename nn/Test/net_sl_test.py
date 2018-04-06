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
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

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

# net_sl.model.cpu()

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
# print(state.bets)

env = SimEnv()

state_tensor = env.state2tensor(state)

LR = [0.01,0.001,0.0001,0.00001]

def load_memory(episoid = 2800):
    memory = ReplayMemory(100000)
    # sl_test_memory = np.load(arguments.WORK_PATH + '/Data/Model/Iter:' + str(episoid) + '_0_slm.npy')
    # for state, policy in zip(*sl_test_memory):
    #     net_sl.memory.push(arguments.Tensor(state).unsqueeze(0), arguments.LongTensor([policy]))
    # file name Iter:100000_0_.sl.npy
    rl_test_memory = np.load(arguments.WORK_PATH + '/Data/Model/Iter:' + str(episoid) + '_0_rlm_.npy')

    # rl memory is state | action | next_state | reward
    for state, action, next_state, reward in rl_test_memory.transpose():
        memory.push(state, action, next_state, reward)
    # TODO finish sl

    return memory


def test(net):
    for i in range(10000):
        net.optimize_model()
        if i % 100 == 0:
            net.target_net.load_state_dict(net.model.state_dict())
        if i % 10 == 0:
            net.plot_error_vis(i)
            net.model.eval()
            # print("episod:"+str(i))
            # print(net.model(Variable(state_tensor)))

    # test dead relu
    current_params = parameters_to_vector(net.model.parameters())
    dead_num = np.count_nonzero(np.less(current_params.data.cpu().numpy(), 0))
    print('dead: %d' % dead_num)
    return dead_num

def test_dead(memory, model='rl'):
    net = None
    dead = []
    for lr in LR:
        if model == 'rl':
            net = DQNOptim(lr=lr)
            net.memory = memory
            dead.append(test(net))
        else:
            raise NotImplementedError
    return list(zip(LR, dead))


if __name__ == '__main__':
    memory = load_memory()
    dead = test_dead(memory, 'rl')