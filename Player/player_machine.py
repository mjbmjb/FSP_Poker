#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 02:41:33 2017

@author: mjb
"""
import torch
import random
from nn.dqn import *
from nn.table_sl import TableSL
from Tree.tree_builder import PokerTreeBuilder
import Settings.arguments as arguments
import Settings.constants as constants
import Settings.game_settings as game_settings
from Game.bet_sizing import BetSizing
from nn.net_sl import SLOptim

class PlayerMachine:
    
    def __init__(self):
        self.dqn_optim = DQNOptim()
        self.table_sl = TableSL()
        self.net_sl = [SLOptim(),SLOptim()]
        
        
    
    
    def load_model(self, iter_time, is_table=False):
        iter_str = str(iter_time)
        # load rl model (only the net)
        self.dqn_optim.model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '_0_'+ '.rl'))
        self.dqn_optim.target_net.load_state_dict(self.dqn_optim.model.state_dict())
        
        # load sl model
        if is_table:
            self.table_sl.strategy = torch.load('../Data/Model/Iter:' + iter_str + '.sl')
        else:
            self.net_sl[0].model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '_0_' + '.sl'))
            self.net_sl[1].model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '_1_' + '.sl'))
        
    # return int action['action:  ,'raise_amount':  ]
    def compute_action(self, state):
        
        # convert tensor for rl
        builder = PokerTreeBuilder()
        state_tensor = builder.statenode_to_tensor(state)
        
        # !!!! the return action is a longTensor[[]]
#        action_id = (self.table_sl.select_action(state) if random.random() > arguments.eta \
#                 else self.dqn_optim.select_action(state_tensor))[0][0]
#        action_id = self.table_sl.select_action(state)[0][0]
        action_id = self.net_sl[state.node.current_player].select_action(state_tensor).item()
#        print('_____________')
#        print(action_id)
#        print('_____________')
        # action['action:  ,'raise_amount':  ]
        action = {}
        
        #fold 
        if action_id == 0:
            action['action'] = constants.acpc_actions.fold
        # call        
        elif action_id == 1 or action_id >= game_settings.actions_count:
            action['action'] = constants.acpc_actions.ccall
        #raise
        elif action_id > 1:
            # get possible to determine the raising size
            bet_sizding = BetSizing(arguments.Tensor(arguments.bet_sizing))
            possible_bets = bet_sizding.get_possible_bets(state.node)
            if possible_bets.dim() != 0:
                possible_bet = possible_bets[:,state.node.current_player]
            else:
                action['action'] = constants.acpc_actions.ccall
                return action

            raise_action_id = action_id - 2# to override fold and call action
            # node possible bet in this state so call
            action['action'] = constants.acpc_actions.rraise
            if(len(possible_bet) <= raise_action_id):            
                action['raise_amount'] = possible_bet[len(possible_bet) - 1].item()
            else:
                action['raise_amount'] = possible_bet[raise_action_id].item() # to override fold and call action
        else:
            assert(False)#invaild actions

        return action
            
            