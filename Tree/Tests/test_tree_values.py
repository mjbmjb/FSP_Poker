#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 01:52:24 2017

@author: mjb
"""

import sys
sys.path.append('/home/mjb/Nutstore/deepStack/')
import torch
import Settings.game_settings as game_settings
import Game.card_to_string as card_to_string
import Settings.arguments as arguments
from Game.card_tools import card_tools
card_tools = card_tools()
from Tree.strategy_filling import StrategyFilling
from Tree.tree_builder import PokerTreeBuilder
from Tree.tree_visualiser import TreeVisualiser
from Tree.tree_values import TreeValues
import Settings.constants as constants

class ValuesTester:
    def dfs_fill_table(self, node, table, builder):
        if node.terminal:
            return
        if node.current_player == constants.players.chance:
            node.table = arguments.Tensor([])
            node.rl = arguments.Tensor([])
            children = node.children
            for child in children:
                self.dfs_fill_table(child,table, builder)
            return
                
        # sl
        all_table = table[node.node_id,:,0:len(node.children)]
        node.table = torch.transpose(all_table.clone(),0,1)
        
    #    print(node.node_id)
        for i in range(node.table.size(1)):
            node.table[:,i].div_(node.table[:,i].sum())
        
        node.strategy = node.table.clone()
    
#        print(node.strategy)
        children = node.children
        for child in children:
            self.dfs_fill_table(child,table, builder)
    
    def test(self, table_sl, model_num):
    
        builder = PokerTreeBuilder()
        
        params = {}
        
        params['root_node'] = {}
        params['root_node']['board'] = card_to_string.string_to_board('Ks')
        params['root_node']['street'] = 1
        params['root_node']['current_player'] = constants.players.P1
        params['root_node']['bets'] = arguments.Tensor([100, 100])
        params['limit_to_street'] = False
        
        tree = builder.build_tree(params)
        
#        table_sl = torch.load('/home/mjb/Nutstore/deepStack/Data/Model/Iter:' + str(model_num) + '.sl')
        #constract the starting range
        filling = StrategyFilling()

        range1 = card_tools.get_uniform_range(params['root_node']['board'])
        range2 = card_tools.get_uniform_range(params['root_node']['board'])

        filling.fill_uniform(tree)


        starting_ranges = arguments.Tensor(constants.players_count, game_settings.card_count)
        starting_ranges[0].copy_(range1)
        starting_ranges[1].copy_(range2)
        
        self.dfs_fill_table(tree, table_sl,builder)
        
        tree_values = TreeValues()
        tree_values.compute_values(tree, starting_ranges)
        
        
        
        print('Exploitability: ' + str(tree.exploitability) + '[chips]' )
        
        visualiser = TreeVisualiser()
        visualiser.graphviz(tree,'test_values')
        
if __name__ == '__main__':
    tester = ValuesTester()
    table_sl = torch.load('/home/mjb/Nutstore/deepStack/Data/Model/Iter:' + str(100000) + '.sl')
    tester.test(table_sl.clone(), 100000)