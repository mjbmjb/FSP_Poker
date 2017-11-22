#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 05:15:26 2017

@author: mjb
"""

import sys
sys.path.append('/home/mjb/Nutstore/deepStack/')
#import cProfilev

import random
import torch
import numpy as np
import Settings.arguments as arguments
import Settings.constants as constants
import Settings.game_settings as game_settings
from itertools import count

from nn.sim_env import SimEnv
from nn.dqn import DQN
from nn.dqn import DQNOptim
from nn.net_sl import SLOptim


from Tree.game_state import GameState
from collections import namedtuple

num_episodes = 10
env = SimEnv()

Agent = namedtuple('Agent',['rl','sl','ID'])

Agents = []
for i in range(game_settings.player_count):
    Agents.append(Agent(rl=DQNOptim(),sl=SLOptim(),ID=i))


def load_model(dqn_optim, iter_time):
    iter_str = str(iter_time)
#    # load rl model (only the net)
#    dqn_optim.model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '.rl'))
#    # load sl model
#    table_sl.strategy = torch.load('../Data/Model/Iter:' + iter_str + '.sl')
#
def save_model(episod):
    path = '../Data/Model/'
    sl_name = path + "Iter:" + str(episod)
    rl_name = path + "Iter:" + str(episod)
    memory_name = path + 'Iter:' + str(episod)
    
    for agent in Agents:
        # save sl strategy
    #    torch.save(table_sl.strategy, sl_name)
        torch.save(agent.sl.model.state_dict(), sl_name + '_'+str(agent.ID)+'_' + '.sl')
        # save rl strategy
        # 1.0 save the prarmeter
        torch.save(agent.rl.model.state_dict(), rl_name + '_'+str(agent.ID)+'_' + '.rl')
        # 2.0 save the memory of DQN
        np.save(memory_name + '_'+str(agent.ID)+'_' + '.memory', np.array(agent.rl.memory.memory))

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

def store_memory(env, agents):
    # store the memory
    for record_list in env.memory:
        for record in record_list:                   
            state, action, next_state, reward = record
            # convert to tensor
            state_tensor = env.state2tensor(state)
            next_state_tensor = env.state2tensor(next_state)
            action_tensor = arguments.LongTensor([[action]])
            reward.div_(arguments.stack)
            assert(type(action) == type(1))
            agents[state.current_player].rl.memory.push(state_tensor, action_tensor, next_state_tensor, reward)

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
    table_update_num = 0
    
    if arguments.load_model:
        load_model(dqn_optim, arguments.load_model_num)
    
    for i_episode in range(arguments.epoch_count + 1):
        
        # Initialize the environment and state
        env.reset()
        state = GameState()
        for t in count():
            state_tensor = env.state2tensor(state)
            current_player = state.current_player
            # Select and perform an action
#            print(state_tensor.size(1))
            assert(state_tensor.size(1) == 136)
            
            flag = 0 if random.random() > arguments.eta else 1
            if flag == 0:
                # sl
                action = Agents[current_player].sl.select_action(state_tensor)
            elif flag == 1:
                #rl
                action = Agents[current_player].rl.select_action(state_tensor)
            else:
                assert(False)
                
            next_state, done = env.step(state, action[0][0])
            
            if flag == 1:
                # if choose sl store tuple(s,a) in supervised learning memory Msl
                state_tensor = env.state2tensor(state)
                action_tensor = action[0]
                Agents[current_player].sl.memory.push(state_tensor, action_tensor)

            
            if done:
                store_memory(env, Agents)
                for agent in Agents:
                    if agent.rl.steps_done > 0 and agent.rl.steps_done % 300 == 0:
                        agent.rl.target_net.load_state_dict(agent.rl.model.state_dict())
                    if len(agent.rl.memory.memory) > agent.rl.memory.capacity / 10 or True:
                        agent.rl.optimize_model()
                        agent.sl.optimize_model()
#                    agent.rl.plot_error_vis(i_episode)
                    if i_episode % 100 == 0:
                        print("rl")
                        print(len(agent.rl.memory))
                break
            

         
            state = next_state
    

#    dqn_optim.plot_error()
#    global LOSS_ACC
#    LOSS_ACC = dqn_optim.error_acc
    # save the model
    if arguments.load_model:
        i_episode = i_episode + arguments.load_model_num
            
    save_model(i_episode)
    print('Complete')
    print((time.time() - time_start))
#    dqn_optim.plt.ioff()
#    dqn_optim.plt.show()

if __name__ == '__main__':
#    cProfile.run(main())
    main()


            
