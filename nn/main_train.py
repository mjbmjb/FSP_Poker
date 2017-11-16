#!/usr/bin/env python2
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
from nn.env import Env
from nn.dqn import DQN
from nn.dqn import DQNOptim
from nn.net_sl import SLOptim
from nn.table_sl import TableSL
from nn.state import GameState
from Tree.tree_builder import PokerTreeBuilder
from Tree.Tests.test_tree_values import ValuesTester
from collections import namedtuple
builder = PokerTreeBuilder()

num_episodes = 10
env = Env()
value_tester = ValuesTester()

Agent = namedtuple('Agent',['rl','sl'])

agent0 = Agent(rl=DQNOptim(),sl=SLOptim())
agent1 = Agent(rl=DQNOptim(),sl=SLOptim())
table_sl = agent0.sl
agents = [agent0,agent1]

def load_model(dqn_optim, iter_time):
    iter_str = str(iter_time)
    # load rl model (only the net)
    dqn_optim.model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '.rl'))
    # load sl model
    table_sl.strategy = torch.load('../Data/Model/Iter:' + iter_str + '.sl')

def save_model(episod):
    path = '../Data/Model/'
    sl_name = path + "Iter:" + str(episod)
    rl_name = path + "Iter:" + str(episod)
    memory_name = path + 'Iter:' + str(episod)   
    # save sl strategy
#    torch.save(table_sl.strategy, sl_name)
    torch.save(agent0.sl.model.state_dict(), sl_name + '_0_' + '.sl')
    torch.save(agent1.sl.model.state_dict(), sl_name + '_1_' + '.sl')
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
    total_reward = 0.0
    table_update_num = 0
    
    if arguments.load_model:
        load_model(dqn_optim, arguments.load_model_num)
    
    for i_episode in range(arguments.epoch_count + 1):
        agents[0], agents[1] = agents[1], agents[0] 
        
        # choose policy 0-sl 1-rl
        flag = 0 if random.random() > arguments.eta else 1
        
        # Initialize the environment and state
        env.reset()
        state = env.state
        for t in count():
            state_tensor = builder.statenode_to_tensor(state)
            current_player = state.node.current_player
            # Select and perform an action
#            print(state_tensor.size(1))
            assert(state_tensor.size(1) == 27)
            
            if flag == 0:
                # sl
                action = agents[current_player].sl.select_action(state_tensor)
            elif flag == 1:
                #rl
                action = agents[current_player].rl.select_action(state_tensor)
            else:
                assert(False)
                
            next_state, real_next_state, reward, done = env.step(agents[1-current_player], state, action)
#            reward = reward / 2400.0
            
            # transform to tensor
            real_next_state_tensor = builder.statenode_to_tensor(real_next_state)
            
            action_tensor = action

            # Store the transition in reforcement learning memory Mrl
            agents[current_player].rl.memory.push(state_tensor, action_tensor, real_next_state_tensor, arguments.Tensor([reward]))
                
            training_flag = False

            if len(agents[current_player].rl.memory.memory) >= agents[current_player].rl.memory.capacity:

                training_flag = True
                if flag == 1:
                    # if choose sl store tuple(s,a) in supervised learning memory Msl
                    agents[current_player].sl.memory.push(state_tensor, action_tensor[0])
                    table_update_num = table_update_num + 1
                    if True or table_update_num >= arguments.sl_update_num:
#                        agents[current_player].sl.update_strategy()
                        agents[current_player].sl.optimize_model()
#                        agents[current_player].sl.plot_error_vis(i_episode)
                        table_update_num = 0
                
                # Perform one step of the optimization (on the target network)
                agents[current_player].rl.optimize_model() 
                # Move to the next state
            state = next_state
    
            # update the target net work
            if agents[current_player].rl.steps_done > 0 and agents[current_player].rl.steps_done % 300 == 0 and training_flag:
                agents[current_player].rl.target_net.load_state_dict(agents[current_player].rl.model.state_dict())
#                agents[current_player].rl.plot_error_vis(i_episode)
#                agents[current_player].sl.plot_error_vis(i_episode)
            
#            if i_episode % 1000 == 0 and training_flag:
#                print(len(agents[current_player].rl.memory.memory))
#               agents[current_player].rl.plot_error_vis(i_episode)
            
            if done:
#                if(i_episode % 100 == 0 and training_flag):
#                    agents[current_player].rl.plot_error_vis(i_episode)
                if(i_episode % arguments.save_epoch == 0 and training_flag):
                    save_model(i_episode)
                    value_tester.test(agents[current_player].sl)
#                    value_tester.test(agents[current_player].sl.strategy.clone(), i_episode)
#                    save_table_csv(table_sl.strategy)
#                dqn_optim.episode_durations.append(t + 1)
#                dqn_optim.plot_durations()
                break
    print(len(agents[current_player].rl.memory.memory))    
#    dqn_optim.plot_error()
#    global LOSS_ACC
#    LOSS_ACC = dqn_optim.error_acc
    # save the model
    if arguments.load_model:
        i_episode = i_episode + arguments.load_model_num
            
            
    print('Complete')
    print((time.time() - time_start))
#    dqn_optim.plt.ioff()
#    dqn_optim.plt.show()

if __name__ == '__main__':
#    cProfile.run(main())
    main()


            
