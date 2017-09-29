#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 19:08:07 2017

@author: mjb
"""

# Evaluates player equities at terminal nodes of the game's public tree.
# @classmod terminal_equity

import torch

import Settings.arguments as arguments
import Settings.constants as constants
import Settings.game_settings as game_settings
from Game.card_tools import card_tools
from Game.Evaluation.evaluator import Evaluator

card_tools = card_tools()

class TerminalEquity:

    # Constructor
    def __init__(self):
        self.set_board(torch.Tensor())
    
    # Constructs the matrix that turns player ranges into showdown equity.
    # 
    # Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay` is the equity
    # for the first player when no player folds.
    # 
    # @param board_cards a non-empty vector of board cards
    # @param call_matrix a tensor where the computed matrix is stored
    # @local
    def get_last_round_call_matrix(self, board_cards, call_matrix):
      assert(board_cards.size(0) == 1 or board_cards.size(0) == 2)#, 'Only Leduc and extended Leduc are now supported' )
      
      e = Evaluator()
      strength = e.batch_eval(board_cards)
      #handling hand stregths (winning probs);
      strength_view_1 = strength.view(game_settings.card_count, 1).expand_as(call_matrix)
      strength_view_2 = strength.view(1, game_settings.card_count).expand_as(call_matrix)
      
      #mjb greater is 1
      call_matrix.copy_(torch.gt(strength_view_1, strength_view_2))
      #mjb lower is -1
      call_matrix.sub_(torch.lt(strength_view_1, strength_view_2).type_as(call_matrix))
    
      self._handle_blocking_cards(call_matrix, board_cards)
    
    # Zeroes entries in an equity matrix that correspond to invalid hands.
    # 
    # A hand is invalid if it shares any cards with the board.
    #
    # @param equity_matrix the matrix to modify
    # @param board a possibly empty vector of board cards
    # @local
    def _handle_blocking_cards(self, equity_matrix, board):
      possible_hand_indexes = card_tools.get_possible_hand_indexes(board)
      possible_hand_matrix = possible_hand_indexes.view(1, game_settings.card_count).expand_as(equity_matrix)
      equity_matrix.mul(possible_hand_matrix)
      possible_hand_matrix = possible_hand_indexes.view(game_settings.card_count,1).expand_as(equity_matrix)
      equity_matrix.mul_(possible_hand_matrix)
    
    # Sets the evaluator's fold matrix, which gives the equity for terminal
    # nodes where one player has folded.
    # 
    # Creates the matrix `B` such that for player ranges `x` and `y`, `x'By` is the equity
    # for the player who doesn't fold
    # @param board a possibly empty vector of board cards
    # @local
    def _set_fold_matrix(self, board):
      self.fold_matrix = arguments.Tensor(game_settings.card_count, game_settings.card_count)
      self.fold_matrix.fill_(1)
      #setting cards that block each other to zero - exactly elements on diagonal in leduc variants
      self.fold_matrix.sub_(torch.eye(game_settings.card_count).type_as(self.fold_matrix))
      self._handle_blocking_cards(self.fold_matrix, board)
    
    # Sets the evaluator's call matrix, which gives the equity for terminal
    # nodes where no player has folded.
    # 
    
    
    # For nodes in the last betting round, creates the matrix `A` such that for player ranges
    # `x` and `y`, `x'Ay` is the equity for the first player when no player folds. For nodes
    # in the first betting round, gives the weighted average of all such possible matrices.
    #
    # @param board a possibly empty vector of board cards
    # @local
    def _set_call_matrix(self, board):
      street = card_tools.board_to_street(board)
      self.equity_matrix = arguments.Tensor(game_settings.card_count, game_settings.card_count).zero_()
      
      if street == 0:
        #iterate through all possible next round streetss
        next_round_boards = card_tools.get_second_round_boards()
        boards_count = next_round_boards.size(0)
        next_round_equity_matrix = arguments.Tensor(game_settings.card_count, game_settings.card_count)
        for board in range(boards_count):
          self.get_last_round_call_matrix(next_round_boards[board], next_round_equity_matrix)
          self.equity_matrix.add_(next_round_equity_matrix)
        #averaging the values in the call matrix
        weight_constant = game_settings.board_card_count == 1 and 1/(game_settings.card_count -2) or 2/((game_settings.card_count -2) * (game_settings.card_count -3 ))
        self.equity_matrix.mul_(weight_constant)
      elif  street == 1:
        #for last round we just return the matrix
        self.get_last_round_call_matrix(board, self.equity_matrix)
      else:
        #impossible street
        assert(False)#, 'impossible street');
    
    # Sets the board cards for the evaluator and creates its internal data structures.
    # @param board a possibly empty vector of board cards
    def set_board(self, board):
        self._set_call_matrix(board)
        self._set_fold_matrix(board)
    
    # Computes (a batch of) counterfactual values that a player achieves at a terminal node
    # where no player has folded.
    # 
    # @{set_board} must be called before this function.
    #
    # @param ranges a batch of opponent ranges in an NxK tensor, where N is the batch size
    # and K is the range size
    # @param result a NxK tensor in which to save the cfvs
    def call_value(self, ranges, result ):
      result.add_(ranges.mm(self.equity_matrix))
    
    # Computes (a batch of) counterfactual values that a player achieves at a terminal node
    # where a player has folded.
    # 
    # @{set_board} must be called before this function.
    #
    # @param ranges a batch of opponent ranges in an NxK tensor, where N is the batch size
    # and K is the range size
    # @param result A NxK tensor in which to save the cfvs. Positive cfvs are returned, and
    # must be negated if the player in question folded.
    def fold_value(self, ranges, result ):
      result.add_(ranges.mm(self.fold_matrix))
    
    # Returns the matrix which gives showdown equity for any ranges.
    # 
    # @{set_board} must be called before this function.
    #
    # @return For nodes in the last betting round, the matrix `A` such that for player ranges
    # `x` and `y`, `x'Ay` is the equity for the first player when no player folds. For nodes
    # in the first betting round, the weighted average of all such possible matrices.
    def get_call_matrix(self):
      return self.equity_matrix
    
    # Computes the counterfactual values that both players achieve at a terminal node
    # where no player has folded.
    # 
    # @{set_board} must be called before this function.
    #
    # @param ranges a 2xK tensor containing ranges for each player (where K is the range size)
    # @param result a 2xK tensor in which to store the cfvs for each player
    def tree_node_call_value(self, ranges, result ):
      assert(ranges.dim() == 2)
      assert(result.dim() == 2)
      self.call_value(ranges[0].view(1,  -1), result[1].view(1,  -1)) 
      self.call_value(ranges[1].view(1,  -1), result[0].view(1,  -1))
    
    # Computes the counterfactual values that both players achieve at a terminal node
    # where either player has folded.
    #
    # @{set_board} must be called before this function.
    #
    # @param ranges a 2xK tensor containing ranges for each player (where K is the range size)
    # @param result a 2xK tensor in which to store the cfvs for each player
    # @param folding_player which player folded
    def tree_node_fold_value(self, ranges, result, folding_player):
      assert(ranges.dim() == 2)
      assert(result.dim() == 2)
      self.fold_value(ranges[0].view(1,  -1), result[1].view(1,  -1)) 
      self.fold_value(ranges[1].view(1,  -1), result[0].view(1,  -1))
      
      result[folding_player].mul_(-1)
