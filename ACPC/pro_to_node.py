#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 07:05:17 2017

@author: mjb
"""
import Settings.game_settings as game_settings
import Settings.arguments as arguments
import Settings.constants as constants
import Game.card_to_string as card_to_string
from Tree.tree_builder import Node
from nn.state import GameState
import re

class ProToNode:
    def __init__(self):
        self.position = None
        self.hand_id = None
        # actions a list of intenger representation of actions
        self.actions = []
        self.actions_raw = []

        # the string of board and private cards
        self.board = None
        self.hand_p1 = ''
        self.hand_p1 = ''
        
    # Parses a list of actions from a string representation.
    # @param actions a string representing a series of actions in ACPC format
    # @return a list of actions, each of which is a table with fields:
        # 
        # * `action`: an element of @{constants.acpc_actions}
    # 
    # * `raise_amount`: the number of chips raised (if `action` is rraise)
    # mjb @return a list of parsed actions in one street
    def _parse_actions(self, actions):
        out = []
        
        actions_reminder = actions
        
        while actions_reminder != '':
            parsed_chunk = ''
            if actions_reminder.startswith('c'):
                out.append({'action': constants.acpc_actions.ccall})
                parsed_chunk = 'c'
            elif actions_reminder.startswith('r'):
                m = re.search(r'(\d+)', actions_reminder)
                amo_str = m.group(1)
                raise_amount = int(amo_str)
                out.append({'action': constants.acpc_actions.rraise, \
                            'raise_amount': raise_amount})
                parsed_chunk = constants.acpc_actions.rraise + amo_str
            elif actions_reminder.startswith('f'):
                out.append({'action': constants.acpc_actions.fold})
                parsed_chunk = 'f'
            else:
                assert(False)
            
            assert(len(parsed_chunk) > 0)
            actions_reminder = actions_reminder[len(parsed_chunk):]
        return out
    
    # Parses a set of parameters that represent a poker state, from a string
    # representation.
    # @param state a string representation of a poker state in ACPC format
    # @return fill the filed of self
    # 
    # * `position`: the acting player
    # 
    # * `hand_id`: a numerical id for the hand
    # 
    # * `actions`: a list of actions which reached the state, for each 
    # betting round - each action is a table with fields:
    # 
    # * `action`: an element of @{constants.acpc_actions}
    # 
    # * `raise_amount`: the number of chips raised (if `action` is rraise)
    # 
    # * `actions_raw`: a string representation of actions for each betting round
    # 
    # * `board`: a string representation of the board cards
    # 
    # * `hand_p1`: a string representation of the first player's private hand
    # 
    # * `hand_p2`: a string representation of the second player's private hand
    # @
        
    def _parse_state(self, state):
        #MATCHSTATE:0:99:cc/r8146c/cc/cc:4cTs|Qs9s/9h5d8d/6c/6d'
         
        m = re.search(r'MATCHSTATE:(\d):(\d+):([^:]*):(\S*)', state)
        position, hand_id, actions, cards = m.groups()
        
        print('position: ', position)
        print('actions: ', actions)
        print('cards: ', cards)
  
        #cc/r8146c/cc/cc
        m = re.search("([^/]*)/?([^/]*)/?([^/]*)/?([^/]*)", actions)
        preflop_actions, flop_actions, turn_actions, river_actions = m.groups()
        
        
        print('preflop_actions: ', preflop_actions)
        print('flop_actions: ', flop_actions)
        
        #4cTs|Qs9s/9h5d8d/6c/6d
        m = re.search("([^\|]*)\|?([^/]*)/?([^/]*)/?([^/]*)/?([^/]*)", cards)
        hand_p1, hand_p2, flop, turn, river = m.groups()
        print('hand_sb: ', hand_p1)
        print('hand_bb: ', hand_p2)
        print('flop: ', flop)
      
        self.position = int(position)
        self.hand_id = hand_id
        
        self.actions.append(self._parse_actions(preflop_actions))
        self.actions.append(self._parse_actions(flop_actions))
        
        self.actions_raw.append(preflop_actions)
        self.actions_raw.append(flop_actions)
        
        self.board = flop
  
        
        self.hand_p1 = hand_p1
        self.hand_p2 = hand_p2
        return 
        
        
   # Processes a list of actions for a betting round.
   # @param actions a list of actions (see @{_parse_actions})
   # @param street the betting round on which the actions takes place
   # `street`, and `index` are added to each action.
   # @return a list of actions
   # @
        
    def _convert_actions_street(self, actions, street,all_actions):
        
        street_first_player = constants.players.P1
        
        for i in range(len(actions)):
            acting_player = 0 if i % 2 == street_first_player else 1 - street_first_player
            
            action = {}
            action['player'] = acting_player
            action['street'] = street
            action['index'] = len(all_actions)
            action['action'] = actions[i]['action']
            if actions[i]['action'] ==  constants.acpc_actions.rraise:
                action['raise_amount'] = actions[i]['raise_amount']
            
            all_actions.append(action)
        
    # Processes all actions.
    # @param actions a list of actions for each betting round
    # @return a of list actions, processed with @{_convert_actions_street} and
    # concatenated
    # @
    def _convert_actions(self, actions):
        all_actions = []  
        for street in range(constants.streets_count):
            self._convert_actions_street(actions[street], street,all_actions)
  
        return all_actions
        
        
    # Further processes a parsed state into a format understandable by DeepStack.
    # @param parsed_state a parsed state returned by @{_parse_state}
    # @return a table of state parameters, with the fields:
    # 
    # * `position`: which player DeepStack is (element of @{constants.players})
    # 
    # * `current_street`: the current betting round
    # 
    # * `actions`: a list of actions which reached the state, for each 
    # betting round - each action is a table with fields:
    # 
    #     * `action`: an element of @{constants.acpc_actions}
    # 
    #     * `raise_amount`: the number of chips raised (if `action` is rraise)
    # 
    # * `actions_raw`: a string representation of actions for each betting round
    # 
    # * `all_actions`: a concatenated list of all of the actions in `actions`,
    # with the following fields added:
    # 
    #     * `player`: the player who made the action
    # 
    #     * `street`: the betting round on which the action was taken
    # 
    #     * `index`: the index of the action in `all_actions`
    # 
    # * `board`: a string representation of the board cards
    # 
    # * `hand_id`: a numerical id for the current hand
    # 
    # * `hand_string`: a string representation of DeepStack's private hand
    # 
    # * `hand_id`: a numerical representation of DeepStack's private hand
    # 
    # * `acting_player`: which player is acting (element of @{constants.players})
    # 
    # * `bet1`, `bet2`: the number of chips committed by each player
    # @
    def _process_parsed_state(self):
        # 1.0 figure out the current street
        current_street = len(self.board) / 2
        print('current_street:' + str(current_street))
        
        # all_actions is a list
        # 2.0 convert actions to player actions
        all_actions = self._convert_actions(self.actions)
        print('all_actions:')
        print(all_actions)
        
        # 3.0 current board
        board = self.board
        print('board:' + str(board))
        
        out = {}
        #TODO
        out['position'] = self.position
        out['current_street'] = current_street
        out['actions'] = self.actions
        out['actions_raw'] = self.actions_raw
        out['all_actions'] = all_actions
        out['board'] = board
        out['hand_number'] = self.hand_id
        
        if out['position'] == constants.players.P1:
            out['hand_string'] = self.hand_p1
        else:
            out['hand_string'] = self.hand_p2  
        out['hand_id'] = card_to_string.string_to_card(out['hand_string'])
        
        
        acting_player = self._get_acting_player(out)
        out['acting_player'] = acting_player
        
        # 5.0 compute bets
        bet1, bet2 = self._compute_bets(out)
        assert(bet1 <= bet2)
  
        if out['position'] == constants.players.P1:
            out['bet1'] = bet1
            out['bet2']= bet2
        else:
            out['bet1'] = bet2
            out['bet2'] = bet1

        return out
        
    # Computes the number of chips committed by each player at a state.
    # @param processed_state a table containing the fields returned by
    # @{_process_parsed_state}, except for `bet1` and `bet2`
    # @return the number of chips committed by the first player
    # @return the number of chips committed by the second player
    # @
    def _compute_bets(self, processed_state):
        if processed_state['acting_player'] == -1:
            return -1, -1
  

        first_p1_action = {'action':constants.acpc_actions.rraise, 'raise_amount':arguments.ante, 'player':constants.players.P1, 'street':0}
        first_p2_action = {'action':constants.acpc_actions.rraise, 'raise_amount':arguments.ante, 'player':constants.players.P2, 'street':1}
  
        # mjb modified
        last_action = first_p2_action
        prev_last_action = first_p1_action
  
        prev_last_bet = first_p2_action
  
        for i in range(len(processed_state['all_actions'])):    
            action = processed_state['all_actions'][i]
            assert(action['player'] == constants.players.P1 or action['player'] == constants.players.P2)
    
            prev_last_action = last_action
            # mjb cause action has different type
            last_action = action

            if action['action'] == constants.acpc_actions.rraise and i <= (len(processed_state['all_actions']) - 3):
                prev_last_bet = action
  
        bet1 = None
        bet2 = None
  
        if last_action['action'] == constants.acpc_actions.rraise and prev_last_action['action'] == constants.acpc_actions.rraise:
            bet1 = prev_last_action['raise_amount']
            bet2 = last_action['raise_amount']
        else:
            if last_action['action'] == constants.acpc_actions.ccall and prev_last_action['action'] == constants.acpc_actions.ccall:
                bet1 = prev_last_bet['raise_amount']
                bet2 = prev_last_bet['raise_amount']
            else:
    
                #either ccal/rraise or rraise/ccal situation
                # TODO
#                assert(last_action['player'] != prev_last_action['player'])
      
                #rraise/ccall
                if last_action['action'] == constants.acpc_actions.ccall:
                    assert(prev_last_action['action'] == constants.acpc_actions.rraise and prev_last_action['raise_amount'])
                    bet1 = prev_last_action['raise_amount']
                    bet2 = prev_last_action['raise_amount']
                else:
                    #call/rraise
        
                    assert(last_action['action'] == constants.acpc_actions.rraise and last_action['raise_amount'])
                    bet1 = prev_last_bet['raise_amount']
                    bet2 = last_action['raise_amount']

        assert(bet1 != None)
        assert(bet2 != None)

        print("bet1 :"+ str(bet1))
        print("bet2 :"+ str(bet2))
  
        return bet1, bet2

    # Gives the acting player at a given state.
    # @param processed_state a table containing the fields returned by
    # @{_process_parsed_state}, except for `accting_player`, `bet1`, and `bet2`
    # @return the acting player, as defined by @{constants.players}
    # @
    def _get_acting_player(self, processed_state):
  
        if len(processed_state['all_actions']) == 0 :
            assert(processed_state['current_street'] == 0)
            return constants.players.P1
  
        last_action = processed_state['all_actions'][len(processed_state['all_actions']) - 1]
  
        #has the street changed since the last action?
        if last_action['street'] != processed_state['current_street']:
            return constants.players.P1
  
        #is the hand over?
        if last_action['action'] == constants.acpc_actions.fold:
            return -1
  
        if processed_state['current_street'] == 1 and len(processed_state['actions'][1]) >= 2 and last_action['action'] == constants.acpc_actions.ccall:
            return -1  
        
        #there are some actions on the current street
        #the acting player is the opponent of the one who made the last action
        return 1 - last_action['player']

    # Turns a string representation of a poker state into a table understandable 
    # by DeepStack.
    # @param state a string representation of a poker state, in ACPC format
    # @return a table of state parameters, with the fields:
    # 
    # * `position`: which player DeepStack is (element of @{constants.players})
    # 
    # * `current_street`: the current betting round
    # 
    # * `actions`: a list of actions which reached the state, for each 
    # betting round - each action is a table with fields:
    # 
    #     * `action`: an element of @{constants.acpc_actions}
    # 
    #     * `raise_amount`: the number of chips raised (if `action` is rraise)
    # 
    # * `actions_raw`: a string representation of actions for each betting round
    # 
    # * `all_actions`: a concatenated list of all of the actions in `actions`,
    # with the following fields added:
    # 
    #     * `player`: the player who made the action
    # 
    #     * `street`: the betting round on which the action was taken
    # 
    #     * `index`: the index of the action in `all_actions`
    # 
    # * `board`: a string representation of the board cards
    # 
    # * `hand_id`: a numerical id for the current hand
    # 
    # * `hand_string`: a string representation of DeepStack's private hand
    # 
    # * `hand_id`: a numerical representation of DeepStack's private hand
    # 
    # * `acting_player`: which player is acting (element of @{constants.players})
    # 
    # * `bet1`, `bet2`: the number of chips committed by each player
    def parse_state(self, state):
 
        self._parse_state(state)
        processed_state = self._process_parsed_state()
  
        return processed_state

    # Gets a representation of the public tree node which corresponds to a
    # processed state.
    # @param processed_state a processed state representation returned by 
    # @{parse_state}
    # @return a table representing a public tree node, with the fields:
    # 
    # * `street`: the current betting round
    # 
    # * `board`: a (possibly empty) vector of board cards
    # 
    # * `current_player`: the currently acting player
    # 
    # * `bets`: a vector of chips committed by each player
    def parsed_state_to_nodestate(self, processed_state):
        node = Node()
        node.street = processed_state['current_street']
        node.board = card_to_string.string_to_board(processed_state['board'])
        node.current_player = processed_state['acting_player']
        node.bets = arguments.Tensor([processed_state['bet1'], processed_state['bet2']])
        
        state = GameState()
        state.node = node
        
        #TODO mjb private card been hardcode
        state.private = [-1 for i in range(game_settings.player_count)]
        state.private[node.current_player] = arguments.Tensor([processed_state['hand_id']])
  
        return state
    
    # Converts an action taken by DeepStack into a string representation.
    # @param adviced_action the action that DeepStack chooses to take, with fields
    # 
    # * `action`: an element of @{constants.acpc_actions}
    # 
    # * `raise_amount`: the number of chips to rraise (if `action` is rraise)
    # @return a string representation of the action
    # @
    def _bet_to_protocol_action(self, adviced_action):
  
        if adviced_action['action'] == constants.acpc_actions.ccall:
            return "c"
        elif adviced_action['action'] == constants.acpc_actions.fold:
            return "f"
        elif adviced_action['action'] == constants.acpc_actions.rraise:
            return "r" + str(adviced_action['raise_amount'])
        else:
            assert(False)

    # Generates a message to send to the ACPC protocol server, given DeepStack's
    # chosen action.
    # @param last_message the last state message sent by the server
    # @param adviced_action the action that DeepStack chooses to take, with fields
    # 
    # * `action`: an element of @{constants.acpc_actions}
    # 
    # * `raise_amount`: the number of chips to rraise (if `action` is rraise)
    # @return a string messsage in ACPC format to send to the server
    def action_to_message(self, last_message, adviced_action):
  
        out = last_message.replace('\n','')
  
        protocol_action = self._bet_to_protocol_action(adviced_action)
        
  
        out = out + ":" + protocol_action
  
        return out
