#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 19:31:47 2018

@author: mjb
"""

import sys
sys.path.append('../../')

import Settings.game_settings as game_settings
from itertools import count
import random

from ctypes import cdll, c_int
dll = cdll.LoadLibrary( "/home/mjb/Nutstore/deepStack/DLL/handstrength.so")
print(';;')
for t in count():
    if (t % 1000 == 0):
        print(t)
    card_stack = list(range(52))
    random.shuffle(card_stack)
#    print(card_stack)
    board = (c_int * 1)()
    for i in range(1):
        board[i] = card_stack.pop()
        # set hole
    hole = ((c_int * 1) * 6)()
    for i in range(6):
        for j in range(1):
            hole[i][j] = card_stack.pop()
                 # init hs
    hs = (c_int * 6)()
        # hs store the hand strength       
#    print(board)
    dll.evalShowdown_1(board, hole, 6, hs)
#    del board
#    del hole
#    del hs

    
    

    