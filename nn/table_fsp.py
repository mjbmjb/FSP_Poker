#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 05:15:26 2017

@author: mjb
"""

import sys
sys.path.append('../')
#import cProfilev

import random
import torch
import numpy as np
import Settings.arguments as arguments
import Settings.constants as constants
import Settings.game_settings as game_settings
from itertools import count
from nn.env import Env
from nn.dqn import DQN
from nn.dqn import DQNOptim
from nn.table_sl import TableSL
from nn.Q_learning import QLearning
from nn.state import GameState
from Tree.tree_builder import PokerTreeBuilder
from Tree.Tests.test_tree_values import ValuesTester
from collections import namedtuple
builder = PokerTreeBuilder()

num_episodes = 10
env = Env()
value_tester = ValuesTester()
action = np.arange(5)

Agent = namedtuple('Agent',['rl','sl'])

agent0 = Agent(rl=QLearning(action),sl=TableSL())
agent1 = Agent(rl=QLearning(action),sl=TableSL())
table_sl = agent0.sl
agents = [agent0,agent1]

def load_model(dqn_optim, iter_time):
    iter_str = str(iter_time)
    # load rl model (only the net)
    dqn_optim.model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '.rl'))
    # load sl model
    table_sl.strategy = torch.load('../Data/Model/Iter:' + iter_str + '.sl')

def save_model(table, episod):
    path = '../Data/Model/'
    sl_name = path + "Iter:" + str(episod) + '.sl'
    rl_name = path + "Iter:" + str(episod)
    memory_name = path + 'Iter:' + str(episod)   
    # save sl strategy
    torch.save(table.strategy, sl_name)
    # save rl strategy
    # 1.0 save the prarmeter
    torch.save(agent0.rl.model.state_dict(), rl_name + '_0_' + '.rl')
    torch.save(agent1.rl.model.state_dict(), rl_name + '_1_' + '.rl')
    # 2.0 save the memory of DQN
    np.save(memory_name, np.array(agent0.rl.memory.memory))
    np.save(memory_name, np.array(agent1.rl.memory.memory))

def save_table_csv(table):
    with open('../Data/table.csv', 'a') as fout:
        for i in range(table.size(0)):
            fout.write(str(table[i].sum()))
            fout.write(',')
        fout.write('\n')
    
    


def get_action(state, current_player ,flag):
    # flag = 0 sl flag = 1 rl
    action = table_sl.select_action(state) if flag == 0 else agents[current_player].rl.select_action(state)
    return action

def state_to_id(state, terminal):
    state_id = state.node.node_id
    hand_id = int(state.private[state.node.current_player][0])
    if terminal:
        return -int(state_id * 5 + hand_id)
    else:
        return int(state_id * 5 + hand_id)
    
def sample_sim(n):
    for i_episode in range(n):

        # Initialize the environment and state
        env.reset()
        state = env.state
        for t in count():
            current_player = state.node.current_player
            action = agents[current_player].sl.select_action(state)
                
            next_state, real_next_state, reward, done = env.step(agents[1-current_player], state, action)
            
            # Store the transition in reforcement learning memory Mrl
            agents[current_player].rl.memory.push(state_to_id(state), action[0][0], state_to_id(real_next_state), reward)

            # Move to the next state
            state = next_state
    

            if done:

                break
    
def sample_alt(m):
    for player in range(2):
        for i_episode in range(m):

            # Initialize the environment and state
            env.reset()
            state = env.state
            for t in count():
                current_player = state.node.current_player
                if current_player == player:
                    action = agents[current_player].rl.select_action(state_to_id(state))
                else:
                    action = agents[current_player].sl.select_action(state)
                    
                next_state, real_next_state, reward, done = env.step(agents[1-current_player], state, action)
                
                if current_player == player:
                    # Store the transition in reforcement learning memory Mrl
                    agents[current_player].rl.memory.push(state_to_id(state), action[0][0], state_to_id(real_next_state), reward)
                    table_sl.store(state, action)
                # Move to the next state
                state = next_state
                
                if done:
    
                    break
    
    
######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` variable. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes.

time_start = 0
#@profile
def main():
    import time
    time_start = time.time()
    for iter_count in range(1000):
        sample_sim(20)
        rl0_memory = agents[0].rl.memory.memory
        rl1_memory = agents[1].rl.memory.memory

        for player in range(2):
            agents[player].rl.batch_learn()
#            agents[player].rl.optimize_model()
#           agents[player].rl.plot_error_vis(iter_count)
#            if agents[player].rl.steps_done > 0 and agents[player].rl.steps_done % 300 == 0:
#                    agents[player].rl.target_net.load_state_dict(agents[player].rl.model.state_dict())
        rl0_table = agents[0].rl.q_table
        rl1_table = agents[1].rl.q_table      

        sample_alt(10)
        for player in range(2):
            agents[player].sl.update_strategy()
            agents[player].rl.iter_count = agents[player].rl.iter_count + 1
            
        
        if iter_count % 300 == 0:
            
            strategy = agents[0].sl.strategy.clone().add_(agents[1].sl.strategy)
#            save_model(strategy,arguments.epoch_count)
            value_tester.test(strategy.clone(), arguments.epoch_count)
            
    print('Complete')
    print((time.time() - time_start))
#    dqn_optim.plt.ioff()
#    dqn_optim.plt.show()



            
