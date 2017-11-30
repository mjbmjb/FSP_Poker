#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 01:30:35 2017

@author: mjb
"""
import torch

import Settings.game_settings as game_settings
import Settings.arguments as arguments
import Settings.constants as constants
from Tree.game_state import GameState
from Tree.game_state import Action

def test_five_card(state):
    call = Action(atype=constants.actions.ccall,amount=0)
    rrasie = Action(atype=constants.actions.rraise,amount=20000)
    fold = Action(atype=constants.actions.fold,amount=0)
    
    hole = torch.LongTensor([[40,41],[50,51],[4,5],[8,9],[44,45],[48,49]])
    #board = torch.LongTensor([6,30,31,38,43])
    board = torch.LongTensor([6,30,31,39,43])
    
    state.bets = arguments.LongTensor([10000,10000,10000,10000,10000,10000])
    state.street = 3
    state.current_player = 0
    
    state.hole = hole
    state.board = board
    
    state.do_action(rrasie)
    state.do_action(fold)
    state.do_action(fold)
    state.do_action(fold)
    state.do_action(call)
    #state.do_action(call)
    state.street = 3