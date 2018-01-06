#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 05:12:49 2017

@author: mjb
"""
import sys
sys.path.append('../')


from sys import argv


import Settings.arguments as arguments
from Player.player_machine import PlayerMachine
from ACPC.acpc_game import ACPCGame

player_machine = PlayerMachine()
player_machine.load_model(argv[3])
#acpc_game = ACPCGame(["MATCHSTATE:1:16:r724:|Ah"])
acpc_game = ACPCGame(None)
acpc_game.connect(argv[1], int(argv[2]))

last_state = None
last_node = None

while True:
    _, state = acpc_game.get_next_situation()
    
    adviced_action = player_machine.compute_action(state)
    
    acpc_game.play_action(adviced_action)
    



 