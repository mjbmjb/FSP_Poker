#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 20:44:52 2017

@author: mjb
"""

# Converts between string and numeric representations of cards.
# @module card_to_string_conversion

import sys
sys.path.append('/home/mjb/Nutstore/deepStack/')
import Settings.game_settings as game_settings
import Settings.arguments as arguments


#All possible card suits - only the first 2 are used in Leduc Hold'e
#suit_table = ['h', 's', 'c', 'd']
suit_table = ['h', 's']

#All possible card ranks - only the first 3-4 are used in Leduc Hold'em and 
# variants.
rank_table = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

# Gets the suit of a card.
# @param card the numeric representation of the card
# @return the index of the suit
def card_to_suit(card):
  return card % game_settings.suit_count


# Gets the rank of a card.
# @param card the numeric representation of the card
# @return the index of the rank
def card_to_rank(card):
  return int(card / game_settings.suit_count )



# Holds the string representation for every possible card, indexed by its 
# numeric representation.
card_to_string_table = {}
for card in range(game_settings.card_count): 
  rank_name = rank_table[card_to_rank(card)]
  suit_name = suit_table[card_to_suit(card)]
  card_to_string_table[card] =  rank_name + suit_name


# Holds the numeric representation for every possible card, indexed by its 
# string representation.
string_to_card_table = {}
for card in range(game_settings.card_count): 
  string_to_card_table[card_to_string_table[card]] = card

 
# Converts a card's numeric representation to its string representation.
# @param card the numeric representation of a card
# @return the string representation of the card
def card_to_string(card):
  assert(card >= 0 and card <= game_settings.card_count )
  return card_to_string_table[card]

# Converts several cards' numeric representations to their string 
# representations.
# @param cards a vector of numeric representations of cards
# @return a string containing each card's string representation, concatenated
def cards_to_string(cards):
  if cards.dim() == 0:
    return ""
  
  out = ""
  for card in range(cards.size(0)):
    out = out + card_to_string(cards[card])
  return out


# Converts a card's string representation to its numeric representation.
# @param card_string the string representation of a card
# @return the numeric representation of the card
def string_to_card(card_string):
  card = string_to_card_table[card_string]
  assert(card >= 0 and card <= game_settings.card_count )
  return card


# Converts a string representing zero or one board cards to a 
# vector of numeric representations.
# @param card_string either the empty string or a string representation of a 
# card
# @return either an empty tensor or a tensor containing the numeric 
# representation of the card
def string_to_board(card_string):
  
  if card_string == '':
    return arguments.Tensor([])
 
  
  return arguments.Tensor([string_to_card(card_string)])
  
def card_to_tensor(card):
    pass
    
    

