import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.optim as optim
import random
import Settings.arguments as arguments
import Settings.game_settings as game_settings

from copy import deepcopy
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.init import normal, calculate_gain, kaiming_normal
from collections import namedtuple

from nn.make_env import make_env


class SimEnv:

    def __init__(self, distributed=False):
        self.distributed = distributed

        self.memory = [[] for i in range(game_settings.player_count)]

    def reset(self):
        self.memory = [[] for i in range(game_settings.player_count)]

    # state: GameState action: int
    # @return next_node, terminal
    def step(self, state, action, is_rl=False):
        pot_size = state.bets.sum()
        current_player = state.current_player
        current_bet = state.bets[current_player]

        vaild_action = self.get_vaild_action(state)

        action_taken = action.item()
        # if action is invaild
        if action_taken >= len(vaild_action):
            action_taken = len(vaild_action) - 1
        #            print(action)
        action_tuple = vaild_action[action_taken]

        # copy the current state, may be slow
        #        print(state.action_string)
        next_state = copy.deepcopy(state)
        state.next = next_state
        next_state.prev = state

        next_state.do_action(action_tuple)

        reward = arguments.Tensor(
            [current_bet - next_state.bets[current_player]]) if not self.distributed else arguments.Tensor([0])

        terminal = next_state.terminal
        terminal_value = None

        # TODO !!!!! here we store action not action_taken
        #        self.store_memory(current_player, state, action, next_state, reward)
        #         action[0][0] = action_taken
        if is_rl:
            self.store_memory(current_player, state, action, reward)
        #        assert(reward[0] < 10 and reward[0] > -10)
        # only for debug
        #        self.store_memory(current_player, state, action_tuple, next_state, reward)

        if next_state.terminal:
            terminal_value = next_state.get_terminal_value()
            for record in self.memory:
                if len(record) > 0:
                    record_player = record[-1].state.current_player
                    if self.distributed:
                        record[-1].reward.add_(terminal_value[record_player] - next_state.bets[record_player])
                    else:
                        record[-1].reward.add_(terminal_value[record_player])

            # for multi agent
            terminal_value = terminal_value - next_state.bets

            # fix the small and big bind
            if len(self.memory[0]) > 0 and len(self.memory[1]) > 0 and not self.distributed:
                self.memory[0][-1].reward.sub_(50)
                self.memory[1][-1].reward.sub_(100)
            #                self.memory[0][-1].reward.sub_(0.3)
            #                self.memory[1][-1].reward.sub_(0.6)
            next_state = None

        return next_state, terminal, action_taken, terminal_value

