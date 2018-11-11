#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 03:53:35 2017

@author: mjb
"""
import sys
sys.path.append('../')

import Settings.constants as constants
import Settings.arguments as arguments
import Settings.game_settings as game_settings
import re
import random

from collections import namedtuple

Action = namedtuple('Action', ['atype', 'amount'])

from ctypes import cdll, c_int
import platform
if platform.uname()[0] == 'Windows':
    dll = cdll.LoadLibrary(arguments.WORK_PATH + "DLL/handstrength.dll")
else:
    dll = cdll.LoadLibrary(arguments.WORK_PATH + "DLL/handstrength.so")


class GameState(object):
    
    
    def __init__(self):
        self.current_player = game_settings.start_player
        self.atype = ""
        self.street = 0
        self.board  = arguments.IntTensor(game_settings.board_card_count).fill_(-1)
        self.hole = arguments.IntTensor(game_settings.player_count,game_settings.private_count).fill_(-1)
        self.board_string = ""    
        self.used_card = arguments.ByteTensor(game_settings.card_count).fill_(0)

        self.terminal = False
        self.actions = []
        self.bet_sizing = []       
        
        self.bets = arguments.LongTensor(game_settings.player_count).fill_(0)
        self.bets[0] = 100
        self.bets[1] = 100
        
        self.active = arguments.ByteTensor(game_settings.player_count).fill_(1)
        self.fold = arguments.ByteTensor(game_settings.player_count).fill_(0)
        self.allin = arguments.ByteTensor(game_settings.player_count).fill_(0)
        self.action_taken = arguments.ByteTensor(game_settings.player_count,
                                                 constants.streets_count,
                                                 game_settings.raises_count * 2 + 1).fill_(0) # 1 means 1 fold
        self.pot = 150
        self.max_bet = 100
        self.call_number = 0
        self.action_string = ''
        
        
        
        self.min_no_limit_raise_to = 2 * 100
        self.max_no_limit_raise_to = arguments.stack
        
        # deal cards
        # self.card_stack = list(range(game_settings.card_count,0,-1))
        self.card_stack = list(range(game_settings.card_count))
        random.shuffle(self.card_stack)
        self._deal_hole()
        
        self.next = GameState
        self.prev = GameState
        self.train = True
    
    def do_action(self, action):
        # [1.0] do action, update player [bets] and [active]
        if action.atype == constants.actions.ccall:
            self.bets[self.current_player] = self.max_bet
            self.call_number += 1
            # if current player called ALLIN action -> not active
            if self.max_bet >= arguments.stack:
                assert(self.max_bet == arguments.stack)
                self.active[self.current_player] = 0
                self.allin[self.current_player] = 1
            self.action_string += 'c'
            self._his_add('c')
         #if current player folded -> not active

        elif action.atype == constants.actions.fold:

            self.active[self.current_player] = 0

            self.fold[self.current_player] = 1

            self.action_string += 'f'
            self._his_add('f')

        else: # must be a raise action

            self.call_number = 1

            # a raise action happened, min_no_limit_raise_to need to be updated

            if action.amount + action.amount - max(self.bets) > self.min_no_limit_raise_to:

                self.min_no_limit_raise_to = action.amount + action.amount - max(self.bets)

            # make sure it <= arguments.stack

            self.min_no_limit_raise_to = min([self.min_no_limit_raise_to, arguments.stack])

            # update {max_bet}

            self.max_bet = action.amount

            self.bets[self.current_player] = action.amount

            # if current player raised to stack size -> not active

            if action.amount >= arguments.stack:
                assert(action.amount == arguments.stack)
                self.active[self.current_player] = 0

                self.allin[self.current_player] = 1

            self.action_string += 'r' + str(action.amount)
            self._his_add('r')

        # if all players choose all in, then game ends, which no active players

        if self.active.sum().item() == 0:

            self.terminal = True

            return
        
        # if only one player is active, and it's spent >= self.max_bet
        if self.active.sum().item() == 1:

            if self.bets[self.active].sum() >= self.max_bet:
                self.terminal = True
                return

        # [3.0] if all active player bets same amount, which means they are reaching next street

        amount_set = set()

        for p, amount in zip(self.active, self.bets):

            if p:

                amount_set.add(amount.item())

        next_street_reaching_flag = len(amount_set) == 1 and self.call_number == game_settings.player_count - self.fold.sum()



        if next_street_reaching_flag:

            # reset call number

            self.call_number = 0

            # we are going to reach next street

            # if current street == 4, then there is no more next street and the game ends here

            if self.street == constants.streets_count - 1:

                self.terminal = True



            # there are next street

            else:

                # update street

                self.street += 1

                # update min_no_limit_raise_to

                self.min_no_limit_raise_to += self.max_bet

                self.min_no_limit_raise_to = min([self.min_no_limit_raise_to, arguments.stack])

                # find next active player from seat 0

                next_player = 0

                while not self.active[next_player]:

                    next_player = (next_player + 1) % game_settings.player_count

                self.current_player = next_player
                
                # if the board hasn't set(when using in player) deal board cards
                if self.train:
                    self._deal_board(self.street)

        else:
            # we are still at current street

            # update {current player}, find next active player

            # if more than one active player left, find next active player

            if self.active.sum().item() > 1:

                # game is not finished yet

                # find next active player

                next_player = (self.current_player + 1) % game_settings.player_count

                while not self.active[next_player]:

                    next_player = (next_player + 1) % game_settings.player_count

                self.current_player = next_player

            elif self.active.sum().item() == 1:

                # game may finish now

                # if there is no all-in player, which means other players all folded, only one player left

                if self.allin.sum().item() == 0:

                    # game ends

                    self.terminal = True

                # else, there are all-in player, the only one player who is active is the next current player

                else:

                    self.current_player = self.active.tolist().index(True)

            else:

                # active player number is 0, which means they are at least one player all-in, the rest are folded

                # game is finished

                self.terminal = True
                
                
    # split betting string into single betting actions

    # if rd=None, by default, handle betting string of all streets
    def get_betting_action(self, rd=None):

        pattern = re.compile(r'r\d+|f|c')

        if rd is not None:

            string = self.betting_string[rd]

            current_street_action = []

            # parse string into sigle action string

            for m in pattern.finditer(string):

                current_street_action.append(m.group())

            return current_street_action

        else:

            betting_action = []

            for string in self.betting_string:

                # parse string into single action string

                current_street_action = []

                for m in pattern.finditer(string):

                    current_street_action.append(m.group())

                betting_action.append(current_street_action)

            return betting_action
     
    # return the ByteTensor of 0 and 1 , 1 means the winner 
    # @show_player bytetensor which 1 indicts which player need to showdown
    def _get_showdown_winner(self, show_player):
#        print('game_state._get_showdown_winner')
        assert(self.terminal)
        # set board
        board = (c_int * game_settings.board_card_count)()
        board_size = c_int(sum(game_settings.board_card_num))
        for i in range(game_settings.board_card_count):
            assert(self.board[i] >= 0)
            board[i] = self.board[i]
        # set hole
        if game_settings.private_count > 1:
            hole = ((c_int * game_settings.private_count) * game_settings.player_count)()
            for i in range(game_settings.player_count):
                for j in range(game_settings.private_count):
                    hole[i][j] = self.hole[i][j]
        elif game_settings.private_count == 1:
            hole = (c_int * game_settings.player_count)()
            for i in range(game_settings.player_count):
                hole[i] = self.hole[i][0]
        else:
            assert (False)
        # init hs
        hs = (c_int * game_settings.player_count)()
        # hs store the hand strength
        if game_settings.private_count == 1:
            dll.evalShowdown_1(board, board_size, hole, game_settings.player_count, hs)
        else:
            dll.evalShowdown(board, board_size, hole, game_settings.player_count, hs)
        
        hst = arguments.LongTensor(game_settings.player_count)
        for i in range(game_settings.player_count):
           hst[i] = hs[i]
        # print(self.board,self.hole,hst)
        del board
        del hole
        del hs
        return hst == hst[show_player].max()
  
    def _deal_hole(self):
        for i in range(game_settings.player_count):
            for j in range(game_settings.private_count):
                self.hole[i][j] = self.card_stack.pop()
    
    # deal card by street
    def _deal_board(self, street):
        board_num = sum(game_settings.board_card_num[:street+1])
        pre_board_num = board_num - game_settings.board_card_num[street]
        for i in range(pre_board_num, board_num):
            self.board[i] = self.card_stack.pop()
        
    def _his_add(self, type):
        if type == 'r':
            for i in range(game_settings.raises_count):
                if self.action_taken[self.current_player,self.street,i].item() == 0:
                    self.action_taken[self.current_player,self.street,i] = 1
                    return
        if type == 'c':
            for i in range(game_settings.raises_count, game_settings.raises_count*2 - 1):
                if self.action_taken[self.current_player,self.street,i].item() == 0:
                    self.action_taken[self.current_player,self.street,i] = 1
                    return
        if type == 'f':
            self.action_taken[self.current_player, self.street,-1] = 1
            return
        # FIXME there wiil be 11101 in raise history
        # raise NotImplementedError

    def get_terminal_value(self):
        assert(self.terminal)
        terminal_value = arguments.LongTensor(game_settings.player_count).fill_(0)
        max_bet = self.bets.max()
        sum_bet = self.bets.sum()
        show_player = (max_bet == self.bets)
        sp_num = show_player.sum()
        
        # state end with fold(means only one player holds the max bets)
        if sp_num == 1:
            terminal_value[self.bets.tolist().index(max_bet)] = sum_bet
        elif sp_num > 1:
            for street in range(self.street + 1, constants.streets_count):
                self._deal_board(street)
            winner = self._get_showdown_winner(show_player)
            terminal_value[winner] = self.bets.sum() / winner.sum().item()
        else:
            assert(True)
        return terminal_value


        
        
# if __name__ == '__main__':
#     state = GameState()
#     call = Action(atype=constants.actions.ccall,amount=0)
#     rrasie = Action(atype=constants.actions.rraise,amount=1000)
#     rrasie1 = Action(atype=constants.actions.rraise,amount=2000)
#     fold = Action(atype=constants.actions.fold,amount=0)
#
#
#
#
#     state.do_action(call)
#     state.do_action(rrasie)
#     state.do_action(call)
#     state.do_action(fold)
#     print(state.street)
#     print(state.board)
#     for i in range(5):
#         state.do_action(call)
#     state.do_action(rrasie)
#     print(state.street)
#     print(state.board)
#     for i in range(5):
#         state.do_action(call)
#     print(state.street)
#     print(state.board)
#     state.do_action(fold)
#
# #    ter = state.get_terminal_value()