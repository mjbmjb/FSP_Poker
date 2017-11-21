#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 06:52:02 2017

@author: mjb
"""
import torch
import Settings.game_settings as game_settings
import Game.card_to_string as card_to_string

import Settings.arguments as arguments
import Settings.constants as constants

from Tree.game_state import GameState
from Tree.game_state import Action

import random


class SimEnv:

    def __init__(self):
        self.state = GameState()
        
    def reset(self):
        self.state = GameState()
        
        

                
    
    #@return next_node, reward, terminal
    def step(self, agent, state, action):
        self.state.do_action(action)
        
        

   # return the list of the vaild action given the state of the current_player
    def get_vaild_action(self, state):
        vaild_action = []
        # if bet is max, cannot fold
        if state.bets[state.current_player] < state.bets.max():
            vaild_action.append(Fold = Action(atype=constants.actions.fold,amount=0))
        # add call
        vaild_action.append(Action(atype=constants.actions.ccall,amount=0))
        # if the max bet is not the same as the stack size
        if state.bets.max() < arguments.stack:
            raise_bucket_len  = int((state.max_no_limit_raise_to - state.min_no_limit_raise_to) / arguments.bet_bucket)
            for i in range(arguments.bet_bucket):
                vaild_action.append(Rrasie = Action(atype=constants.actions.rraise,amount=state.max_no_limit_raise_to+raise_bucket_len))
            # if greater than stack, change it to stack
            vaild_action[-1].amount = arguments.stack if vaild_action[-1].amount > arguments.stack else vaild_action[-1].amount
                
                
        return vaild_action
        
    def process_log(self, state, real_next_node, action, reward):
        node = state.node
        with open('state_action.log','a') as fout:
            fout.write('player: ' + str(node.current_player) + '\n' +
                       'private, public cards: ' + str(state.private[0][0]) +','+str(state.private[1][0])+ node.board_string + '\n' +
                       'bets: ' + str(node.bets[0]) + ',' + str(node.bets[1]) + '\n' 
                       'actions: ' + str(action[0][0]) + '\n'
                       'rewards:' + str(reward) + '\n'+
                       'terminal: ' + str(real_next_node.terminal) + '\n\n')
            