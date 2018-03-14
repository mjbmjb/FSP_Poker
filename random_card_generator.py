#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 04:12:34 2017

@author: mjb
"""

# Samples random card combinations.
# @module random_card_generator

import torch
import Settings.arguments as arguments
import Settings.constants as constants
import Settings.game_settings as game_settings
import random

# Samples a random set of cards.
# 
# Each subset of the deck of the correct size is sampled with 
# uniform probability.
#
def generate_cards( count ):
  #marking all used cards
  used_cards = torch.ByteTensor(game_settings.card_count).zero_()
  
  out = arguments.IntTensor(count)
  #counter for generated cards
  generated_cards_count = 0
  while(generated_cards_count < count):
    card = random.randint(0, game_settings.card_count - 1)
    if ( used_cards[card] == 0 ): 
      out[generated_cards_count] = card
      generated_cards_count = generated_cards_count + 1
      used_cards[card] = 1
  return out
