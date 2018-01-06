#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 00:23:01 2017

@author: mjb
"""
import sys
sys.path.append('../')


import Settings.game_settings as game_settings
import Settings.arguments as arguments
import Settings.constants as constants

from Tree.game_state import GameState
from Tree.game_state import Action

from ACPC.message_parser import MessageParser

import re

class MsgToState:
    
    def __init__(self, message):
        self.message_parser = MessageParser(message)
        
        self.state = GameState()
        
        # hole, [['2c', '2d'],...] for 6 players
        self.hole = self.message_parser.hole
        # board ['2c', '2d', '2h', ...] for at most 5 board cards
        self.board = self.message_parser.board
        self.viewing_player = self.message_parser.get_position()
        self.betting_string = self.message_parser.get_betting_string(rd=None)
        self.board_string = self.message_parser.get_board_string(rd=None)
        # a two-dimension list, dim-1 is round, dim-2 is i-th action of a round
        self.betting_action = self.get_betting_action()
        # a one-dimension list, store each {action string} as an element
        self.action_list = []
        # store each {action object} of all round actions in a one-dimension list
        for each_round_action in self.betting_action:
            for action in each_round_action:
                self.action_list.append(self._parse_action(action))
            
        # first set training flag to false
        self.state.train = False
        # update [hole] and [board]
        self.state.hole = self.message_parser.get_hole_card(position=None)
        self.state.board = self.message_parser.get_board_card(rd=None)
        
        
        # after setting up basic data structure, start to do each action and update data structure
        cnt = 0
        for action in self.action_list:
#            if cnt == 18:
#                mjb = 1
            cnt+=1
            self.state.do_action(action)
        
    def _parse_action(self, action_str):
        # call
        if action_str[0] == 'c':
            return Action(atype=constants.actions.ccall,amount=0)
        # fold
        elif action_str[0] == 'f':
            return Action(atype=constants.actions.fold,amount=0)
        # raise
        elif action_str[0] == 'r':
            return Action(atype=constants.actions.rraise,amount=int(action_str[1:]))
        else:
            assert(False)
            
        
    
    # split betting string into single betting actions
    # if rd=None, by default, handle betting string of all rounds
    def get_betting_action(self, rd=None):
        pattern = re.compile(r'r\d+|f|c')
        if rd is not None:
            string = self.betting_string[rd]
            current_round_action = []
            # parse string into sigle action string
            for m in pattern.finditer(string):
                current_round_action.append(m.group())
            return current_round_action
        else:
            betting_action = []
            for string in self.betting_string:
                # parse string into single action string
                current_round_action = []
                for m in pattern.finditer(string):
                    current_round_action.append(m.group())
                betting_action.append(current_round_action)
            return betting_action


    
if __name__ == '__main__':
    s = "MATCHSTATE:1:5:r200fccff:|4h2s||||"
    m2s = MsgToState(s)
    print(m2s.get_betting_action())
    print(m2s.action_list)
    