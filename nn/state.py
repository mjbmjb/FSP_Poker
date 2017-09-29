#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 00:14:47 2017

@author: mjb
"""
import Settings.game_settings as game_settings
import Settings.arguments as arguments
import Settings.constants as constants
from Tree.tree_builder import PokerTreeBuilder, Node

class GameState:
    def __init__(self):
        self.private = []
        self.private_string = ""
        self.node = Node
