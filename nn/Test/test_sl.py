#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 20:31:09 2017

@author: mjb
"""
import sys
sys.path.append('/home/mjb/Nutstore/deepStack/')

import Settings.game_settings as game_settings
import Game.card_to_string as card_to_string
import Settings.arguments as arguments
from Tree.tree_builder import PokerTreeBuilder
from Tree.tree_visualiser import TreeVisualiser
import Settings.constants as constants
from Player.player_machine import PlayerMachine
from nn.dqn import *
from nn.state import GameState
import torch


def dfs_fill_table(node, table, dqnmodel, builder):
    if node.terminal:
        return
#    if node.current_player == constants.players.chance:
#        node.table = arguments.Tensor([])
#        node.rl = arguments.Tensor([])
#        children = node.children
#        for child in children:
#            dfs_fill_table(child,table, dqnmodel, builder)
#        return
            
    # sl
    all_table = table[node.node_id,:,:]
#    print(node.node_id)
    for i in range(all_table.size(0)):
        all_table[i,:] = all_table[i,:] / all_table[i,:].sum()
    
    node.table = all_table
    
    #rl
    for i in range(game_settings.card_count):
        state = GameState()
        state.node = node
        state.private = [arguments.Tensor([i]),arguments.Tensor([i])]
        state_tensor = builder.statenode_to_tensor(state)
        node.rl = torch.cat((node.rl,dqnmodel(Variable(state_tensor, volatile=True)).data),0)
        
    
    children = node.children
    for child in children:
        dfs_fill_table(child,table, dqnmodel, builder)


model_num = 25000

builder = PokerTreeBuilder()

params = {}

params['root_node'] = {}
params['root_node']['board'] = card_to_string.string_to_board('')
params['root_node']['street'] = 0
params['root_node']['current_player'] = constants.players.P1
params['root_node']['bets'] = arguments.Tensor([100, 100])
params['limit_to_street'] = False
tree = builder.build_tree(params)

table_sl = torch.load('/home/mjb/Nutstore/deepStack/Data/Model/Iter:' + str(model_num) + '.sl')
    
dqn = DQN()
if torch.cuda.is_available():
    dqn.cuda()

dqn.load_state_dict(torch.load('/home/mjb/Nutstore/deepStack/Data/Model/Iter:' + str(model_num) + '.rl'))

#dfs_fill_table(tree, table_sl, dqn, builder)

acc_list = []
acc_node = {}
builder.acc_node(tree, acc_node, acc_list)
state = GameState()
state.private = [arguments.Tensor([0]),arguments.Tensor([2])]
state.node = acc_node['858']
print(dqn(Variable(builder.statenode_to_tensor(state))))

#visualiser = TreeVisualiser()
#visualiser.graphviz(tree,"table_sl:" + str(model_num))
