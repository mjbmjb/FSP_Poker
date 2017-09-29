#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 01:30:35 2017

@author: mjb
"""


import Settings.game_settings as game_settings
import Game.card_to_string as card_to_string
import Settings.arguments as arguments
from Tree.tree_builder import PokerTreeBuilder
from Tree.tree_visualiser import TreeVisualiser
import Settings.constants as constants
from nn.state import GameState

from nn.env import Env

builder = PokerTreeBuilder()
env = Env()
env.reset()

tensor = builder.statenode_to_tensor(env.state)

print(tensor.size(0))