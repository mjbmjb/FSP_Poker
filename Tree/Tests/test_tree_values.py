#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 01:52:24 2017

@author: mjb
"""

import sys
sys.path.append('../')
import torch
from torch.autograd import Variable
import Settings.game_settings as game_settings
import Game.card_to_string as card_to_string
import Settings.arguments as arguments
from Game.card_tools import card_tools
card_tools = card_tools()

from Tree.strategy_filling import StrategyFilling
from Tree.tree_builder import PokerTreeBuilder
from Tree.tree_visualiser import TreeVisualiser
from Tree.tree_values import TreeValues
from nn.state import GameState
from nn.net_sl import SLOptim
from nn.dqn import DQNOptim
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
            
    def dfs_fill_strategy(self, agent_sl, node, builder):
        if node.terminal:
            return
        if node.current_player == constants.players.chance:
            node.table = arguments.Tensor([])
            node.rl = arguments.Tensor([])
            children = node.children
            for child in children:
                self.dfs_fill_strategy(agent_sl, child, builder)
            return
            
        #sl
        for card in range(game_settings.card_count):
            state = GameState()
            for player in range(game_settings.player_count):
                state.private.append(arguments.Tensor([card]))
            state.node = node
            tensor = builder.statenode_to_tensor(state)
            strategy = agent_sl.model(Variable(tensor)).data[0][0:len(node.children)]
            if isinstance(agent_sl, DQNOptim):
#                print(strategy)
                max_ix = strategy.lt(strategy.max())
                strategy[max_ix] = 0.0001
                strategy[1-max_ix] = 1
            strategy.div_(strategy.sum())
            node.strategy[:,card] = strategy

        children = node.children
        for child in children:
            self.dfs_fill_strategy(agent_sl, child, builder)
    
    def test(self, table_sl):
    
        builder = PokerTreeBuilder()
        
        params = {}
        
        params['root_node'] = {}
        params['root_node']['board'] = card_to_string.string_to_board('')
        params['root_node']['street'] = 0
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


        starting_ranges = arguments.Tensor(game_settings.player_count, game_settings.card_count)
        starting_ranges[0].copy_(range1)
        starting_ranges[1].copy_(range2)
        
        table_sl.model.eval()
#        self.dfs_fill_table(tree, table_sl,builder)
        self.dfs_fill_strategy(table_sl,tree, builder)
        
        tree_values = TreeValues()
        tree_values.compute_values(tree, starting_ranges)
        
        
        
        print('Exploitability: ' + str(tree.exploitability.item()) + '[chips]' )
        return tree.exploitability.item()
#        visualiser = TreeVisualiser()
#        visualiser.graphviz(tree,'test_values')
        
if __name__ == '__main__':
    tester = ValuesTester()
#    table_sl = torch.load('/home/mjb/Nutstore/deepStack/Data/Model/Iter:' + str(100000) + '.sl')
#    tester.test(table_sl.clone())
    optim = SLOptim()
    tester.test(optim)
