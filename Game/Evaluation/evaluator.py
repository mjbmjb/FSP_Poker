#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 18:55:01 2017

@author: mjb
"""
# Evaluates hand strength in Leduc Hold'em and variants.
# 
# Works with hands which contain two or three cards, but assumes that
# the deck contains no more than two cards of each rank (so three-of-a-kind
# is not a possible hand).
# 
# Hand strength is given as a numerical value, where a lower strength means
# a stronger hand: high pair < low pair < high card < low card
# @module evaluator

import math
import Settings.game_settings as game_settings
import Game.card_to_string as card_to_string

from Game.card_tools import card_tools
card_tools = card_tools()

import Settings.arguments as arguments

class Evaluator:
    # Gives a strength representation for a hand containing two cards.
    # @param hand_ranks the rank of each card in the hand
    # @return the strength value of the hand
    # @
    def evaluate_two_card_hand(self, hand_ranks):
        #check for the pair 
        hand_value = None    
        if hand_ranks[0] == hand_ranks[1]:
            #hand is a pair
            hand_value = hand_ranks[0]
        else:
            #hand is a high card    
            hand_value = hand_ranks[0] * game_settings.rank_count + hand_ranks[1]    
        return hand_value
    
    # Gives a strength representation for a hand containing three cards.
    # @param hand_ranks the rank of each card in the hand
    # @return the strength value of the hand
    # @
    def evaluate_three_card_hand(self, hand_ranks):
        hand_value = None
        #check for the pair 
        if hand_ranks[0] == hand_ranks[1]:   
            #paired hand, value of the pair goes first, value of the kicker goes second
            hand_value = hand_ranks[0] * game_settings.rank_count + hand_ranks[2]
        elif hand_ranks[1] == hand_ranks[2]: 
            #paired hand, value of the pair goes first, value of the kicker goes second
            hand_value = hand_ranks[1] * game_settings.rank_count + hand_ranks[0]
        else:
            #hand is a high card    
            hand_value = hand_ranks[0] * game_settings.rank_count * game_settings.rank_count + hand_ranks[1] * game_settings.rank_count + hand_ranks[2]   
        return hand_value
    
    
    # Gives a strength representation for a two or three card hand.
    # @param hand a vector of two or three cards
    # @param[opt] impossible_hand_value the value to return if the hand is invalid
    # @return the strength value of the hand, or `impossible_hand_value` if the 
    # hand is invalid
    def evaluate(self, hand, impossible_hand_value):
      assert(hand.max() <= game_settings.card_count and hand.min() >= 0)#, 'hand:es not correspond to any cards' )
      impossible_hand_value = impossible_hand_value or -1
      if not card_tools.hand_is_possible(hand):
        return impossible_hand_value
      
      #we are not interested in the hand suit - we will use ranks instead of cards
      hand_ranks = hand.clone()
      for i in range(hand_ranks.size(0)): 
        hand_ranks[i] = card_to_string.card_to_rank(hand_ranks[i])
      hand_ranks = hand_ranks.sort()
      if hand.size(0) == 2:
        return self.evaluate_two_card_hand(hand_ranks[0])
      elif hand.size(0) == 3:
#        print hand_ranks
        return self.evaluate_three_card_hand(hand_ranks[0])
        
      elif hand.size(0) == 1:
          # !!!!!!mjb  for all-in in the first round, only have one private card
        return hand[0]
      else:
        assert(False)#, 'unsupported size of hand!' )
    
    # Gives strength representations for all private hands on the given board.
    # @param board a possibly empty vector of board cards
    # @param impossible_hand_value the value to assign to hands which are invalid 
    # on the board
    # @return a vector containing a strength value or `impossible_hand_value` for
    # every private hand
    def batch_eval(self, board, impossible_hand_value = -1):

        hand_values = arguments.Tensor(game_settings.card_count).fill_(-1)
        if board.dim() == 0: 
            for hand in range(game_settings.card_count): 
                hand_values[hand] = math.floor((hand -1 ) / game_settings.suit_count ) + 1
        else:
            board_size = board.size(0)
            assert(board_size == 1 or board_size == 2)#, 'Incorrect board size for Leduc' )
            whole_hand = arguments.Tensor(board_size + 1)
            whole_hand[0:-1].copy_(board)
            for card in range(game_settings.card_count): 
                whole_hand[-1] = card; 
                hand_values[card] = self.evaluate(whole_hand, impossible_hand_value)
        return hand_values
#    for debug
def main():
    import torch
    e = evaluator()
    print (e.batch_eval(torch.Tensor([]),-1))
    
