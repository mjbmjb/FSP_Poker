#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 01:30:35 2017

@author: mjb
"""

# Builds a public tree for Leduc Hold'em or variants.
# 
# Each node of the tree contains the following fields:
# 
# * `node_type`: an element of @{constants.node_types} (if applicable)
# 
# * `street`: the current betting round
# 
# * `board`: a possibly empty vector of board cards
# 
# * `board_string`: a string representation of the board cards
# 
# * `current_player`: the player acting at the node
# 
# * `bets`: the number of chips that each player has committed to the pot
#
# * `pot`: half the pot size, equal to the smaller number in `bets`
#
# * `children`: a list of children nodes
# @classmod tree_builder

import torch
import Settings.game_settings as game_settings
import Game.card_to_string as card_to_string
from Game.Evaluation.evaluator import Evaluator
from Game.card_tools import card_tools
card_tools = card_tools()

import Settings.arguments as arguments
import Settings.constants as constants
from Game.bet_sizing import BetSizing


class Node:
    def __init__(self):
        self.current_player = -2
        self.type = ""
        self.street = -1
        self.board  = ""
        self.board_string = ""
        self.bets = arguments.Tensor(2)
        self.pot = 0
        self.parent = Node
        self.children = []
        self.terminal = False
        self.actions = []
        self.bet_sizing = []       
        self.node_id = 0
        self.strategy = arguments.Tensor([])
        
        self.table = arguments.Tensor([])
        self.rl = arguments.Tensor([])
        
class PokerTreeBuilder:

    # Constructor
    def __init__(self):
        self.node_id_acc = 0
    
    
    # Creates the child node after a call which transitions between betting 
    # rounds.
    # @param parent_node the node at which the transition call happens
    # @return a list containing the child node
    # @local
    def _get_children_nodes_transition_call(self, parent_node):
    
      chance_node = Node()
      self.node_id_acc = self.node_id_acc + 1
      
      chance_node.node_id = self.node_id_acc
      chance_node.type = constants.node_types.chance_node
      chance_node.street = parent_node.street
      chance_node.board= parent_node.board
      chance_node.board_string = parent_node.board_string
      chance_node.current_player = constants.players.chance  
      chance_node.bets = parent_node.bets.clone()
    
      return [chance_node]
    
    # Creates the children nodes after a chance node.
    # @param parent_node the chance node
    # @return a list of children nodes
    # @local
    def _get_children_nodes_chance_node(self, parent_node):
      assert(parent_node.current_player == constants.players.chance)
      
      if self.limit_to_street:
        return []
    
      next_boards = card_tools.get_second_round_boards()
      next_boards_count = next_boards.size(0)
    
      subtree_height = -1
      children = []
      #mjb the chance node's child differ with the different board card 
      #1.0 iterate over the next possible boards to build the corresponding subtrees
      for i in range(next_boards_count):
        next_board = next_boards[i]
        next_board_string = card_to_string.cards_to_string(next_board)
    
        child = Node()
        self.node_id_acc = self.node_id_acc + 1
    
        child.node_id = self.node_id_acc
        child.type = constants.node_types.inner_node
        child.parent = parent_node
        child.current_player = constants.players.P1
        child.street = parent_node.street + 1
        child.board = next_board
        child.board_string = next_board_string
        child.bets = parent_node.bets.clone()
    
        children.append(child)
    
      return children
    
    # Fills in additional convenience attributes which only depend on existing
    # node attributes.
    # @param node the node
    # @local
    def _fill_additional_attributes(self, node):
      node.pot = node.bets.min()
    
    # Creates the children nodes after a player node.
    # @param parent_node the chance node
    # @return a list of children nodes
    # @local
    def _get_children_player_node(self, parent_node):
      assert(parent_node.current_player != constants.players.chance)
    
      children = []
      
      #1.0 fold action
      fold_node = Node()
      self.node_id_acc = self.node_id_acc + 1
      
      fold_node.node_id = self.node_id_acc
      fold_node.type = constants.node_types.terminal_fold
      fold_node.terminal = True
      fold_node.current_player = 1 - parent_node.current_player
      fold_node.street = parent_node.street 
      fold_node.board = parent_node.board
      fold_node.board_string = parent_node.board_string
      fold_node.bets = parent_node.bets.clone()
      children.append(fold_node)
      
      #2.0 check action
      if parent_node.current_player == constants.players.P1 and (parent_node.bets[0] == parent_node.bets[1]):
        check_node = Node()
        self.node_id_acc = self.node_id_acc + 1
        
        check_node.node_id = self.node_id_acc
        check_node.type = constants.node_types.check
        check_node.terminal = False
        check_node.current_player = 1 - parent_node.current_player
        check_node.street = parent_node.street 
        check_node.board = parent_node.board
        check_node.board_string = parent_node.board_string
        check_node.bets = parent_node.bets.clone()
        children.append(check_node)
      #transition call
      elif parent_node.street == 0 and ( (parent_node.current_player == constants.players.P2 and \
                                            parent_node.bets[0] == parent_node.bets[1]) or \
                                           (parent_node.bets[0] != parent_node.bets[1] and \
                                            max(parent_node.bets) < arguments.stack) ):
        chance_node = Node()
        self.node_id_acc = self.node_id_acc + 1
        
        chance_node.node_id = self.node_id_acc
        chance_node.type = constants.node_types.chance_node
        chance_node.street = parent_node.street
        chance_node.board = parent_node.board
        chance_node.board_string = parent_node.board_string
        chance_node.current_player = constants.players.chance  
        chance_node.bets = parent_node.bets.clone().fill_(parent_node.bets.max())
        children.append(chance_node)
      else:
      #2.0 terminal call - either last street or allin
        terminal_call_node = Node()
        self.node_id_acc = self.node_id_acc + 1
        
        terminal_call_node.node_id = self.node_id_acc
        terminal_call_node.type = constants.node_types.terminal_call
        terminal_call_node.terminal = True
        terminal_call_node.current_player = 1 - parent_node.current_player
        terminal_call_node.street = parent_node.street 
        terminal_call_node.board = parent_node.board
        terminal_call_node.board_string = parent_node.board_string
        terminal_call_node.bets = parent_node.bets.clone().fill_(parent_node.bets.max())
        children.append(terminal_call_node)
    
      #3.0 bet actions    
      possible_bets = self.bet_sizing.get_possible_bets(parent_node)
      
      if possible_bets.dim() != 0:
        assert(possible_bets.size(1) == 2)
        
        for i in range(possible_bets.size(0)):
          child = Node()
          self.node_id_acc = self.node_id_acc + 1
          
          child.node_id = self.node_id_acc
          child.parent = parent_node
          child.current_player = 1 - parent_node.current_player
          child.street = parent_node.street 
          child.board = parent_node.board
          child.board_string = parent_node.board_string
          child.bets = possible_bets[i]
          children.append(child)
      
      return children
    
    # Creates the children after a node.
    # @param parent_node the node to create children for
    # @return a list of children nodes
    # @local
    def _get_children_nodes(self, parent_node):
    
      #is this a transition call node (leading to a chance node)?
      call_is_transit = parent_node.current_player == constants.players.P2 and \
                                                      parent_node.bets[0] == parent_node.bets[1] and \
                                                      parent_node.street < constants.streets_count
      
      chance_node = parent_node.current_player == constants.players.chance

      chance_node = parent_node.current_player != 0 and parent_node.current_player != 1
      #transition call -> create a chance node
      if  parent_node.terminal:
        return []
      #chance node
      elif chance_node:
        return self._get_children_nodes_chance_node(parent_node)
      #inner nodes -> handle bet sizes
      else:
        return self._get_children_player_node(parent_node)
    
      assert(False)
    
    # Recursively build the (sub)tree rooted at the current node.
    # @param current_node the root to build the (sub)tree from
    # @return `current_node` after the (sub)tree has been built
    # @local
    def _build_tree_dfs(self, current_node):
      
      self._fill_additional_attributes(current_node)
      children = self._get_children_nodes(current_node)
      current_node.children = children
      
      depth = 0
#      if len(children) == 0:
#          print len(children)
      current_node.actions= []
      for i in range(len(children)):   
#        assert(children[i].node_id != 857)
        children[i].parent = current_node
        self._build_tree_dfs(children[i])
        depth = max(depth, children[i].depth)
        if i == 0:
          current_node.actions.append(constants.actions.fold)
        elif i == 1:
          current_node.actions.append(constants.actions.ccall)
        else:
          current_node.actions.append(children[i].bets.max())
      current_node.depth = depth + 1


      
      return current_node
     
    
    # Builds the tree.
    # @param params table of tree parameters, containing the following fields:
    # 
    # * `street`: the betting round of the root node
    # 
    # * `bets`: the number of chips committed at the root node by each player
    # 
    # * `current_player`: the acting player at the root node
    # 
    # * `board`: a possibly empty vector of board cards at the root node
    # 
    # * `limit_to_street`: if `true`, only build the current betting round
    # 
    # * `bet_sizing` (optional): a @{bet_sizing} object which gives the allowed
    # bets for each player 
    # @return the root node of the built tree
    def build_tree(self, params):
      root = Node()
      #.current_player necessary stuff from the root_node not to touch the input
      root.street = params['root_node']['street']
      root.bets = params['root_node']['bets'].clone()
      root.current_player = params['root_node']['current_player']
      root.board = params['root_node']['board'].clone()
      root.board_string = card_to_string.cards_to_string(root.board)
      
      params['bet_sizing'] = params['bet_sizing'] if 'bet_sizing' in params else BetSizing(arguments.Tensor(arguments.bet_sizing))
    
      assert(params['bet_sizing'])
    
      self.bet_sizing = params['bet_sizing']
      self.limit_to_street = params['limit_to_street']
    
      self._build_tree_dfs(root)
      
#      mjb 
#      strategy_filling = StrategyFilling()
#      strategy_filling.fill_uniform(root)
      
      return root

    def statenode_to_tensor(self, state):
#      tensor = arguments.Tensor(constants.player_count, \
#                                constants.streets_count, \
#                                constants.raises_count, \
#                                constants.acions_count, \
#                                constants.card_count * 2).fill_(0)
      if (state == None):
          return None
      node = state.node
    
      # transform street [0,1] means the first street
      street_tensor = arguments.Tensor(constants.streets_count).fill_(0)
      street_tensor[int(node.street)] = 1
      
      #position_tensor
      position_tensor = arguments.Tensor(4).fill_(node.current_player)
      
                    
      # transform #detpth# and bets
      bet_player_tensor = arguments.Tensor(arguments.bet_bucket).fill_(0)
      bet_player_tensor[int((node.bets[node.current_player]-1) / arguments.bet_bucket_len)] = 1
      bet_oppo_tensor = arguments.Tensor(arguments.bet_bucket).fill_(0)                 
      bet_oppo_tensor[int((node.bets[1-node.current_player]-1) / arguments.bet_bucket_len)] = 1
#      print(node.bets)
#      print(bet_player_tensor)
#      print(bet_oppo_tensor)
            
      
      # transform hand(private and board)
#      print(len(state.private))
      assert(len(state.private) == 2)
      private_tensor = card_tools.hand_to_tensor(arguments.Tensor(state.private[node.current_player]))
      board_tensor = card_tools.hand_to_tensor(node.board)
      
      #transform hand strengen
      player_hand = arguments.Tensor(state.private[node.current_player].tolist() + node.board.tolist())
      evaluator = Evaluator()
      player_strength = evaluator.evaluate(player_hand, -1)
      strength_tensor = arguments.Tensor([player_strength])
      
      # street: 1-2 position 3 bets 4-5 private 
      return_tensor = torch.unsqueeze(torch.cat((street_tensor, position_tensor,
                                         bet_player_tensor, bet_oppo_tensor, private_tensor, board_tensor,
                                         strength_tensor) , 0), 0)
#      print("private:" + str(state.private[node.current_player]))
#      print("board:" + node.board_string)
      
#      print(return_tensor)
      return return_tensor
      
    def acc_node(self, tree_root, acc_node, acc_list):
        assert(not str(tree_root.node_id) in acc_node)
        acc_node[str(tree_root.node_id)] = tree_root
        acc_list.append(tree_root.node_id)
        if tree_root.terminal:
            return
        for child in tree_root.children:
            self.acc_node(child, acc_node, acc_list)
         
         
         
         
         