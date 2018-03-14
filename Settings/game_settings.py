#!/usr/bin/env python2
# #*# coding: utf#8 #*#
"""
Created on Sun Aug 20 00:33:16 2017

@author: mjb
"""

# Game constants which define the game played by DeepStack.
# @module game_settings



# the number of card suits in the deck
suit_count = 2
# the number of card ranks in the deck
rank_count = 3
# the total number of cards in the deck
card_count = suit_count * rank_count;
# the number of public cards dealt in the game (revealed after the first
# betting round)
board_card_count = 1;
board_card_num = [0,1]
# the number of players in the game 
player_count = 2
# mjb the num of raises
raises_count = 4
# mjb the num of actions
actions_count = 7
# mjb private
private_count = 1
