#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 06:52:02 2017

@author: mjb
"""
import torch
import Settings.game_settings as game_settings
import Game.card_to_string as card_to_string

from Game.card_tools import card_tools
card_tools = card_tools()

import Settings.arguments as arguments
import Settings.constants as constants
from Game.bet_sizing import BetSizing
from Game.Evaluation.evaluator import Evaluator
from Tree.tree_builder import PokerTreeBuilder
from Tree.strategy_filling import StrategyFilling

import random_card_generator
from nn.state import GameState

import random

filling = StrategyFilling()
builder = PokerTreeBuilder()
al_nn = []
rl_nn = []
evaluator = Evaluator()

class Env:

    def __init__(self):
        params = {}
        params['root_node'] = {}
        params['root_node']['board'] = card_to_string.string_to_board('')
        params['root_node']['street'] = 0
        params['root_node']['current_player'] = constants.players.P1
        params['root_node']['bets'] = arguments.Tensor([100, 100])
        params['limit_to_street'] = False
        builder = PokerTreeBuilder()
        self.root_node = builder.build_tree(params)
#        print(self.builder.node_id_acc)
        filling.fill_uniform(self.root_node)
        self.state = GameState()
        self._cached_terminal_equities = {}
        
    def reset(self):
        self.state = GameState()
        pri_card = random_card_generator.generate_cards(game_settings.private_count * 2)
        self.state.private.append(pri_card[0:game_settings.private_count])
        self.state.private.append(pri_card[game_settings.private_count:])
        self.state.node = self.root_node
        
        
    def _chance_next(self, this_node, state):
        rannum = random.random()
        hand_id = int(state.private[this_node.current_player][0])
        chance_strategy = this_node.strategy[:,hand_id]
        for i in range(len(chance_strategy)):
            if rannum <= sum(chance_strategy[0: i+1]):
                return this_node.children[i]
                
    
    #@return next_node, reward, terminal
    def step(self, agent, state, action):
        parent_node = state.node
        
        
        # TODO grasp if action if invaild
        if action[0][0] >= len(state.node.children):
            action[0][0] = 1
        # fold in first round is invaild
        if action[0][0] == 0 and parent_node.bets[0] == 100 and parent_node.bets[1] == 100:
            action[0][0] = 1

#        assert (action < 4)
        next_node = state.node.children[action[0][0]]
        if next_node.current_player == constants.players.chance:
            next_node = self._chance_next(next_node, state)
                
    #    next_state reward
        next_state = GameState()
        next_state.node = next_node
        next_state.private = state.private
        reward = parent_node.bets[parent_node.current_player] - next_node.bets[parent_node.current_player]
        # if we choose to reach the terminal we should observe the ternimal value    
        terminal = False
        
        #if chance node and the acting player is player 0 (who is act first at every round)
        if next_node.current_player == parent_node.current_player:
            return next_state, next_state, reward, terminal
        if next_node.terminal:
            reward = reward + self._get_terminal_value(next_state, parent_node.current_player)
            terminal = True
            next_state = None
            real_next_state = next_state

        else:
            next_state_tensor = builder.statenode_to_tensor(next_state)
#            oppo_action = int((agent.sl.select_action(next_state) if random.random() > arguments.eta \
#                          else agent.rl.select_action(next_state_tensor))[0][0])
            oppo_action = int(agent.sl.select_action(next_state)[0][0])  
            if oppo_action >= len(next_node.children):
                oppo_action = 1
            real_next_node = next_node.children[oppo_action]

            if real_next_node.current_player == constants.players.chance:
                real_next_node = self._chance_next(real_next_node, state)                

            assert(real_next_node != None)
            real_next_state = GameState()
            real_next_state.node = real_next_node
            real_next_state.private = state.private
            if real_next_node.terminal:
                reward = reward + self._get_terminal_value(real_next_state, parent_node.current_player)
                real_next_state = None

            

            
#        print(parent_node.node_id)
#        print(next_node.node_id)
#        if real_next_node:
#            print(real_next_node.node_id)
#        else:
#            print('None')
#        print(reward)
#        self.process_log(state, real_next_node, action, reward)
        return next_state, real_next_state, reward, terminal
                           
        #[0,1,1,1] means the second action
        
#    def _get_ternimal_equity(self, node):
#        if len(self._cached_terminal_equities == 0):
#            cached = TerminalEquity()
#            cached.set_board(node.board)
#            self._cached_terminal_equities[node.board_string] = cached
#        cached = self._cached_terminal_equities[node.board_string]
#        return cached

#    # return vaule FloatTensor(2)
#    def _get_terminal_value(self, state):
#        node = state.node
#        assert(node.terminal)
#
#        if node.type == constants.node_types.terminal_fold:
#            #ternimal fold
#            value = - node.bets[node.current_player]
#        elif node.type == constants.node_types.terminal_call:
#            # show down
#            player_hand = arguments.Tensor(state.private[node.current_player].tolist() + node.board.tolist())
#            player_strength = evaluator.evaluate(player_hand, -1)
#            oppo_hand = arguments.Tensor(state.private[1 - node.current_player].tolist() + node.board.tolist())
#            oppo_strength = evaluator.evaluate(oppo_hand, -1)
#            
#            if player_strength < oppo_strength:
#                value = node.bets[1 - node.current_player]
#            else:
#                value = -node.bets[node.current_player]
#        else:
#            assert(False)# not a vaild terminal node
#            
#        return value


        # return vaule FloatTensor(2)
    def _get_terminal_value(self, state, player):
        # oppo take the action and lead to this terminal node 
        
        node = state.node
        assert(node.terminal)
        value = arguments.Tensor([0,0])
        
        if node.type == constants.node_types.terminal_fold:
        #ternimal fold   
            value[node.current_player] = node.bets.sum()

        elif node.type == constants.node_types.terminal_call:
            # show down
            player_hand = arguments.Tensor(state.private[node.current_player].tolist() + node.board.tolist())
            player_strength = evaluator.evaluate(player_hand, -1)
            oppo_hand = arguments.Tensor(state.private[1 - node.current_player].tolist() + node.board.tolist())
            oppo_strength = evaluator.evaluate(oppo_hand, -1)
            
            # the one take call lose
            if player_strength < oppo_strength:
                value[node.current_player] = node.bets.sum()
            else:
                value[1-node.current_player] = node.bets.sum()
        else:
            assert(False)# not a vaild terminal node
            
        return value[player]
            
    def _al_action(self, state):
        
        # get possible bets in the node
        possible_bets = get_possible_actions()
        actions_count = possible_bets.size(0)
        
        # get the strategy
        
        
        assert(math.abs(1 - hand_strategy.sum()) < 0.001)
        # sample the action 
        action = strategy.cumsum(0).gt(random.random())
        
        return action
    
        
    def process_log(self, state, real_next_node, action, reward):
        node = state.node
        with open('state_action.log','a') as fout:
            fout.write('player: ' + str(node.current_player) + '\n' +
                       'private, public cards: ' + str(state.private[0][0]) +','+str(state.private[1][0])+ node.board_string + '\n' +
                       'bets: ' + str(node.bets[0]) + ',' + str(node.bets[1]) + '\n' 
                       'actions: ' + str(action[0][0]) + '\n'
                       'rewards:' + str(reward) + '\n'+
                       'terminal: ' + str(real_next_node.terminal) + '\n\n')
            