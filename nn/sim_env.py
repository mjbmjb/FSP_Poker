#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 06:52:02 2017

@author: mjb
"""
import sys
sys.path.append('/home/mjb/Nutstore/deepStack/')

import torch
import Settings.game_settings as game_settings
import Game.card_to_string as card_to_string

import Settings.arguments as arguments
import Settings.constants as constants

from Tree.game_state import GameState
from Tree.game_state import Action

import random
import copy
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class SimEnv:

    def __init__(self):
        self.memory = [[] for i in range(game_settings.player_count)]
        
    def reset(self):
        self.memory = [[] for i in range(game_settings.player_count)]
        
        

                
    # state: GameState action: int 
    #@return next_node, terminal
    def step(self, state, action):
        current_player = state.current_player
        current_bet = state.bets[current_player]
        
        vaild_action = self.get_vaild_action(state)
        
        # if action is invaild
        if action >= len(vaild_action):
            action = len(vaild_action) - 1
        
        action_tuple = vaild_action[action] 
        
        # copy the current state, may be slow
        next_state = copy.deepcopy(state)
 
        next_state.do_action(action_tuple)
        
        reward = arguments.Tensor([current_bet - next_state.bets[current_player]])
        terminal = next_state.terminal
        
        self.store_memory(current_player, state, action, next_state, reward)
        # only for debug
#        self.store_memory(current_player, state, action_tuple, next_state, reward)
        
        if next_state.terminal: 
            terminal_value = next_state.get_terminal_value()
            for record in self.memory:
                if len(record) > 0:
                    record[-1].reward.add_(terminal_value[record[-1].state.current_player])
            next_state = None
        
        return next_state, terminal
    
    def store_memory(self, current_player, *args):
        self.memory[current_player].append(Transition(*args))
        


   # return the list of the vaild action given the state of the current_player
    def get_vaild_action(self, state):
        vaild_action = []
        # if bet is max, cannot fold
        if state.bets[state.current_player] < state.bets.max():
            vaild_action.append(Action(atype=constants.actions.fold,amount=0))
        # add call
        vaild_action.append(Action(atype=constants.actions.ccall,amount=0))
        # if the max bet is not the same as the stack size
        if state.bets.max() < arguments.stack:
            raise_bucket_len  = int((state.max_no_limit_raise_to - state.min_no_limit_raise_to) / arguments.bet_bucket)
            for i in range(arguments.bet_bucket):
                vaild_action.append(Action(atype=constants.actions.rraise,amount=state.min_no_limit_raise_to+raise_bucket_len*i))
            # if greater than stack, change it to stack
            if vaild_action[-1].amount > arguments.stack:
                vaild_action[-1] = Action(atype=constants.actions.rraise,amount=arguments.stack)
                
        return vaild_action
    
    def _cards_to_tensor(self, cards):
        tensor = arguments.Tensor(game_settings.card_count).fill_(0)
    
        for i in range(cards.size(0)):
            if cards[i] > 0:
                tensor[int(cards[i])] = 1
        return tensor
    
    # |street (16 * 1 or 0) | current_position (24 * 1 or 0) | bets | hole cards|board cards|
    def state2tensor(self, state):
        if (state == None):
          return None
    
        # transform street [0,1] means the first street
        street_tensor = arguments.Tensor(constants.streets_count).fill_(0)
        street_tensor[int(state.street)] = 1
      
        #position_tensor
        position_tensor = arguments.Tensor(constants.streets_count).fill_(state.current_player)
      
                    
        # transform bets
        bet_tensor = arguments.Tensor(arguments.bet_bucket * game_settings.player_count).fill_(0)
        for i in range(game_settings.player_count):
            bet_tensor[i*arguments.bet_bucket + int((state.bets[state.current_player]-1) / arguments.bet_bucket_len)] = 1

      
#      print(node.bets)
#      print(bet_player_tensor)
#      print(bet_oppo_tensor)
            
      
      # transform hand(private and board)
#      print(len(state.private))
        private_tensor = self._cards_to_tensor(state.hole[state.current_player])
        board_tensor = self._cards_to_tensor(state.board)
      
        #transform hand strengen

      
        # street: 1-2 position 3 bets 4-5 private 
        return_tensor = torch.unsqueeze(torch.cat((street_tensor, position_tensor,
                                         bet_tensor, private_tensor, board_tensor,) , 0), 0)
        
        return return_tensor
        
    def process_log(self, state, real_next_node, action, reward):
        node = state.node
        with open('state_action.log','a') as fout:
            fout.write('player: ' + str(node.current_player) + '\n' +
                       'private, public cards: ' + str(state.private[0][0]) +','+str(state.private[1][0])+ node.board_string + '\n' +
                       'bets: ' + str(node.bets[0]) + ',' + str(node.bets[1]) + '\n' 
                       'actions: ' + str(action[0][0]) + '\n'
                       'rewards:' + str(reward) + '\n'+
                       'terminal: ' + str(real_next_node.terminal) + '\n\n')
    
if __name__ == '__main__':
    env = SimEnv()
    call = Action(atype=constants.actions.ccall,amount=0)
    rrasie = Action(atype=constants.actions.rraise,amount=500)
    fold = Action(atype=constants.actions.fold,amount=0)
    

    state = GameState()
    terminal = state.terminal
    
    while not terminal:
        action_list = env.get_vaild_action(state)
        next_state, terminal = env.step(state,1)
        state = next_state
        
 