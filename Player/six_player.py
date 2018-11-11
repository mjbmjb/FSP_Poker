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

import argparse
def get_args():
  parser = argparse.ArgumentParser(description='Six_Player')
  parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='host (default: 127.0.0.1)')
  parser.add_argument('--port', type=int, default= 20000,
                        help='port (default: False)')
  parser.add_argument('--model_num', type=int, default=200,
                        help='model_num (default: 200)')

  args = parser.parse_args()
  return args

args = get_args()

player_machine = SixPlayerMachine()
player_machine.load_model(args.model_num)
acpc_game = SixACPCGame(["MATCHSTATE:1:299:c:|Ks"])
#acpc_game = SixACPCGame(["MATCHSTATE:2:22:cfcr5722r11771cccc/ccccr12969fccr15663fcc/:||3c6d|||/Th9c5d/Ks"])

if arguments.evalation:
    acpc_game = SixACPCGame(None)
    acpc_game.connect(args.host, args.port)
    print('connection finished')

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

    # print(raction)