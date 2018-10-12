#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 06:52:02 2017

@author: mjb
"""
#%%
import sys
sys.path.append('../')

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

#Transition = namedtuple('Transition',
#                        ('state', 'action', 'next_state', 'reward'))
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward'))

class SimEnv:

    def __init__(self, distributed = False):
        self.distributed = distributed
        
        self.memory = [[] for i in range(game_settings.player_count)]
        
    def reset(self):
        self.memory = [[] for i in range(game_settings.player_count)]
        
        

                
    # state: GameState action: int 
    #@return next_node, terminal
    def step(self, state, action, is_rl=False):
        pot_size = state.bets.sum()
        current_player = state.current_player
        current_bet = state.bets[current_player]
        
        vaild_action = self.get_vaild_action(state)
        
        action_taken = action.item()
        # if action is invaild
        if action_taken >= len(vaild_action):
            action_taken = len(vaild_action) - 1
#            print(action)
        action_tuple = vaild_action[action_taken] 
        
        # copy the current state, may be slow
#        print(state.action_string)
        next_state = copy.deepcopy(state)
        state.next = next_state
        next_state.prev = state
 
        next_state.do_action(action_tuple)
       
        reward = arguments.Tensor([current_bet - next_state.bets[current_player]]) if not self.distributed else arguments.Tensor([0])
        
        terminal = next_state.terminal
        terminal_value = None
        
        # TODO !!!!! here we store action not action_taken
#        self.store_memory(current_player, state, action, next_state, reward)
#         action[0][0] = action_taken
        if is_rl:
            self.store_memory(current_player, state, action, reward)
#        assert(reward[0] < 10 and reward[0] > -10)
        # only for debug
#        self.store_memory(current_player, state, action_tuple, next_state, reward)
        
        if next_state.terminal: 
            terminal_value = next_state.get_terminal_value()
            for record in self.memory:
                if len(record) > 0:
                    record_player = record[-1].state.current_player
                    if self.distributed:
                        record[-1].reward.add_((terminal_value[record_player] - next_state.bets[record_player]).float())
                    else:
                        record[-1].reward.add_(terminal_value[record_player].float())

            # for multi agent
            terminal_value = terminal_value - next_state.bets
                    
            # fix the small and big bind
            if len(self.memory[0]) > 0 and len(self.memory[1]) > 0 and not self.distributed:
                self.memory[0][-1].reward.sub_(50)
                self.memory[1][-1].reward.sub_(100)
#                self.memory[0][-1].reward.sub_(0.3)
#                self.memory[1][-1].reward.sub_(0.6)
            next_state = None
        
        return next_state, terminal, action_taken, terminal_value

    # @return GameState: next_state, Boolean terminal, list(Tensor) state_a, list(LongTensor) state_a,
    #         Tensor(player-N) reward
    def step_r(self, start_state, rl, sl, eta=arguments.eta):
        # End if all the agents acted
        n_acted = 0
        n_active = start_state.active.sum()
        # 暂时用全0替代吧 （log时为零，就没有梯度了） 现在用的是onehot
        action_a = [arguments.Tensor(1,game_settings.actions_count).fill_(0)] * game_settings.player_count
        state_a = [arguments.Tensor(1,arguments.dim_obs).fill_(0)] * game_settings.player_count
        reward = arguments.LongTensor(game_settings.player_count).fill_(0)

        state = start_state
        while n_acted < n_active:
            current_player = state.current_player
            state_tensor = self.state2tensor(state)

            is_rl = random.random() > arguments.eta # 0 sl 1 rl
            if is_rl:
                action, onehot_a = rl.select_action(current_player,state_tensor)
            else:
                action, onehot_a = sl.select_action(state_tensor)

            next_state, terminal, action_taken, terminal_value = self.step(state, action, True)

            if is_rl:
                sl.memory.push(state_tensor, action)

            # action[0][0] = action_taken
            state_a[current_player] = state_tensor
            action_a[current_player] = onehot_a

            if terminal:
                return next_state, terminal, state_a, action_a, terminal_value
            state = next_state
            n_acted += 1

        return next_state, terminal, state_a, action_a, reward




    def store_memory(self, current_player, *args):
        self.memory[current_player].append(Transition(*args))

   # return the list of the vaild action given the state of the current_player
    def get_vaild_action(self, state):
        vaild_action = []
        # if bet is max, cannot fold
#        if state.bets[state.current_player] < state.bets.max():
        vaild_action.append(Action(atype=constants.actions.fold,amount=0))
        # add call
        vaild_action.append(Action(atype=constants.actions.ccall,amount=0))
        # if the max bet is not the same as the stack size
#        if state.bets.max() < arguments.stack:
#            raise_bucket_len  = int((state.max_no_limit_raise_to - state.min_no_limit_raise_to) / arguments.bet_bucket)
#            for i in range(arguments.bet_bucket):
#                vaild_action.append(Action(atype=constants.actions.rraise,amount=state.min_no_limit_raise_to+raise_bucket_len*i))
#            # if greater than stack, change it to stack
#            if vaild_action[-1].amount > arguments.stack:
#                vaild_action[-1] = Action(atype=constants.actions.rraise,amount=arguments.stack)

        #  F C 1/4P 1/2P P A
        if state.bets.max() < arguments.stack:
            pot_times = arguments.pot_times
            pot_size = state.bets.sum()
            
            for times in pot_times:
                raise_size = int(times * pot_size)
                if raise_size < state.min_no_limit_raise_to: continue
                # greater than big bind and smaller than satck
                if raise_size > 100 and raise_size < arguments.stack:
                    vaild_action.append(Action(atype=constants.actions.rraise, amount=raise_size))
            # all in 
#            vaild_action.append(Action(atype=constants.actions.rraise,amount=arguments.stack))
#        print(vaild_action)
        return vaild_action
    
    def _cards_to_tensor(self, cards):
        tensor = arguments.Tensor(game_settings.card_count).fill_(0)
        for i in range(cards.size(0)):
            if cards[i] >= 0:
                tensor[int(cards[i])] = 1
        return tensor
    
    # |street (16 * 1 or 0) | current_position (24 * 1 or 0) | bets | hole cards|board cards|
    def state2tensor(self, state):
        if state is None:
          return None
        # print(state.action_string)

        # transform street [0,1] means the first street # 4 /32 0-31
        # TODO make one bit to test the dimision of input
        street_tensor = arguments.Tensor(constants.streets_count * 1).fill_(0)
        street_tensor[int(state.street)* 1: int(state.street+1)* 1] = 1
      
        # position_tensor # /48 32-80
        position_tensor = arguments.Tensor(game_settings.player_count * 1).fill_(0)
        position_tensor[state.current_player* 1: (state.current_player+1)* 1] = 1
      
        # active tensor / 6 81-86
        active_tensor = arguments.Tensor(game_settings.player_count)
        active_tensor[state.active] = 1

        # transform bets 60 87-146
        bet_tensor = arguments.Tensor(arguments.bet_bucket * game_settings.player_count).fill_(0)
        for i in range(game_settings.player_count):
            bet_tensor[i*arguments.bet_bucket + int((state.bets[i]-1) / arguments.bet_bucket_len)]= 1
            
        # ransform pot 60 87-146
        pot_size = state.bets.max().item()
        pot_tensor = arguments.Tensor(len(arguments.pot_times) * game_settings.player_count).fill_(0)
        for i in range(game_settings.player_count):
            for j in range(len(arguments.pot_times)):
                if state.bets[i] < arguments.pot_times[j] * pot_size:
                    pot_tensor[i * len(arguments.pot_times) + j] = 1
                    break

        # betting history
        betting_his = state.action_taken.view(-1).type(arguments.Tensor)

#      print(node.bets)
#      print(bet_player_tensor)
#      print(bet_oppo_tensor)

       # transform hand(private and board) 52
#      print(len(state.private)) 52
        private_tensor = self._cards_to_tensor(state.hole[state.current_player])
        board_tensor = self._cards_to_tensor(state.board)
      
        #transform hand strengen
        # street: 1-2 position 3 bets 4-5 private 
        return_tensor = torch.unsqueeze(torch.cat((street_tensor, position_tensor, active_tensor,
                                         bet_tensor, pot_tensor, betting_his, private_tensor, board_tensor,) , 0), 0)
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
#%%
# if __name__ == '__main__':
#     env = SimEnv()
#     call = Action(atype=constants.actions.ccall,amount=0)
#     rrasie = Action(atype=constants.actions.rraise,amount=500)
#     fold = Action(atype=constants.actions.fold,amount=0)
#
#
#     state = GameState()
#     terminal = state.terminal
#
#     state_tensor = env.state2tensor(state)
#     #%%
#     t = torch.stack([state_tensor,state_tensor])
#     from nn.maddpg import MADDPG
#     maddpg = MADDPG()
#     a = maddpg.select_action(t)
#
# # while not terminal:
# #     action_list = env.get_vaild_action(state)
# #     print(action_list)
# #     env.step_r(state, 1)
# #     next_state, terminal, action= env.step(state,torch.LongTensor([[1]]))
# #     state = next_state

 