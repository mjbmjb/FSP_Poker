#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 00:27:06 2017

@author: mjb
"""

import sys
sys.path.append('../')

import torch
import Settings.game_settings as game_settings
import Game.card_to_string as card_to_string
import Settings.arguments as arguments
from Tree.tree_builder import PokerTreeBuilder
from Tree.tree_visualiser import TreeVisualiser
import Settings.constants as constants



builder = PokerTreeBuilder()

params = {}

params['root_node'] = {}
params['root_node']['board'] = card_to_string.string_to_board('')
params['root_node']['street'] = 0
params['root_node']['current_player'] = 2
params['root_node']['bets'] = arguments.Tensor([50, 100, 0])
params['limit_to_street'] = False
params['root_node']['action_taken'] = torch.ByteTensor([1,1,0])
tree = builder.build_tree(params)

acc_list = []
acc_node = {}
builder.acc_node(tree, acc_node, acc_list)
print(max(acc_list))
print(builder.node_id_acc)

#visualiser = TreeVisualiser()

#visualiser.graphviz(tree, "simple_tree")
