#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 00:08:38 2017

@author: mjb
"""

import sys
sys.path.append('../')


from sys import argv


import Settings.arguments as arguments
from Player.six_player_machine import SixPlayerMachine
from ACPC.six_acpc_game import SixACPCGame

player_machine = SixPlayerMachine()
player_machine.load_model(argv[3])
#acpc_game = SixACPCGame(["MATCHSTATE:3:63:ccfcr18914cfcr20000fcc///:|Ac8d||Jd2s||8h7h/Kc5c6c/Ad/Jh"])
#acpc_game = SixACPCGame(["MATCHSTATE:2:22:cfcr5722r11771cccc/ccccr12969fccr15663fcc/:||3c6d|||/Th9c5d/Ks"])

acpc_game = SixACPCGame(None)
acpc_game.connect(argv[1], int(argv[2]))

last_state = None
last_node = None

raction = []

while True:
    state = acpc_game.get_next_situation()
    
    adviced_action = player_machine.compute_action(state)
    
    acpc_game.play_action(adviced_action)
    
    if adviced_action.atype != -1 and adviced_action.atype != -2:
        print("Raise Action: " + str(adviced_action))
        raction.append(adviced_action)
