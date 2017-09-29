#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 05:34:11 2017

@author: mjb
"""

import Settings.game_settings as game_settings
import Game.card_to_string as card_to_string
from Game.card_tools import card_tools
card_tools = card_tools()
import Settings.arguments as arguments
from Tree.tree_builder import PokerTreeBuilder
from Tree.tree_visualiser import TreeVisualiser
from Tree.tree_builder import Node
from Tree.strategy_filling import StrategyFilling
import Settings.constants as constants


builder = PokerTreeBuilder()

params = {}
params['root_node'] = {}
params['root_node']['board'] = card_to_string.string_to_board('Ks')
params['root_node']['street'] = 1
params['root_node']['current_player'] = constants.players.P1
params['root_node']['bets'] = arguments.Tensor([100, 100])
params['limit_to_street'] = True

tree = builder.build_tree(params)

filling = StrategyFilling()

range1 = card_tools.get_uniform_range(params['root_node']['board'])
range2 = card_tools.get_uniform_range(params['root_node']['board'])

filling.fill_uniform(tree)


starting_ranges = arguments.Tensor(constants.players_count, game_settings.card_count)
starting_ranges[0].copy_(range1)
starting_ranges[1].copy_(range2)

#tree_values = TreeValues()
#tree_values:compute_values(tree, starting_ranges)

#print('Exploitability: ' + tree.exploitability + '[chips]' )

visualiser = TreeVisualiser()
visualiser.graphviz(tree,"strategy_fill")
