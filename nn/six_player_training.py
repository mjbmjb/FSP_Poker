#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 05:15:26 2017

@author: mjb
"""

import sys
sys.path.append('../')
#sys.path.append('/home/carc/mjb/deepStack/')

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#import cProfilev

import random
import torch
import numpy as np
import Settings.arguments as arguments
import Settings.constants as constants
import Settings.game_settings as game_settings
from itertools import count
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

from nn.sim_env import SimEnv
from nn.dqn import DQNOptim
from nn.reinforce import ReinforceOptim
from nn.net_sl import SLOptim
from nn.dqn import Transition as RL_TRAN
from nn.net_sl import Transition as SL_TRAN

from Tree.game_state import GameState
from collections import namedtuple

num_episodes = 10
env = SimEnv(True)

Agent = namedtuple('Agent',['rl','sl','ID'])
Reward = [0] * game_settings.player_count

rl_model = {'dqn':DQNOptim, 'reinforce':ReinforceOptim}
sl_model = {'mlr':SLOptim}

Agents = []
n_agent = game_settings.player_count if arguments.multi_agent else 1
for i in range(n_agent):
    Agents.append(Agent(rl=rl_model[arguments.rl_model](),sl=sl_model[arguments.sl_model](),ID=i))


def load_model(iter_time):
    iter_str = str(iter_time)
#    # load rl model (only the net)
    for i in range(game_settings.player_count):
        Agents[i].rl.model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '_' + str(i) +'_' + '.rl'))
        Agents[i].rl.target_net.load_state_dict(Agents[i].rl.model.state_dict())
        
        Agents[i].sl.model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '_' + str(i) +'_' + '.sl'))
        

def save_model(episod):
    path = "../Data/Model/Iter:" + str(episod)
    
    for agent in Agents:
        # save sl strategy
    #    torch.save(table_sl.strategy, sl_name)  
        sl_model = agent.sl.model.state_dict()
        rl_model = agent.rl.model.state_dict()
        if arguments.cpu_store:
            sl_model = agent.sl.model.cpu().state_dict()
            rl_model = agent.rl.model.cpu().state_dict()

            
    
        torch.save(sl_model, path + '_'+str(agent.ID)+'_' + '.sl')
        # save rl strategy
        # 1.0 save the prarmeter
        torch.save(rl_model, path + '_'+str(agent.ID)+'_' + '.rl')
        # TODO add rl memory save
        # 2.0 save the memory of sl
        save_memory(path + '_')

        if arguments.gpu:
            agent.sl.model.cuda()
            agent.rl.model.cuda()

def save_memory(path):
    for agent in Agents:
        # if len(agent.sl.memory.memory) > 0:
        #     sl_memory = SL_TRAN(*zip(*agent.sl.memory.memory))
        #     np.save(path + str(agent.ID) + '_slm_', sl_memory)

            # sl_state = np.vstack(sl_memory.state)
            # sl_action = np.vstack(sl_memory.policy)
            # sl_whole = np.append(sl_state, sl_action, axis=1)
            # np.save(path, sl_whole)
            # np.save(memory_name + '_' + str(agent.ID) + '_' + '.rl', np.array(agent.rl.memory.memory))
        if len(agent.rl.memory.memory) > 0:
            rl_memory = RL_TRAN(*zip(*agent.rl.memory.memory))
            np.save(path + str(agent.ID) + '_rlm_', rl_memory)
            # rl_state = np.vstack(rl_memory.state)
            # rl_reward = np.vstack(rl_memory.reward)
            # rl_action = np.vstack(rl_memory.action)
            # rl_whole = np.hstack((rl_state,rl_reward,rl_action))
            #
            # rl_nexts = np.array(rl_memory.next_state) # cause there are None value in next_state
            # np.save(path+'_ns_',rl_nexts)
            # np.save(path, rl_whole)


def save_table_csv(table):
    with open('../Data/table.csv', 'a') as fout:
        for i in range(table.size(0)):
            fout.write(str(table[i].sum()))
            fout.write(',')
        fout.write('\n')
    
    


def get_action(state, current_player ,flag):
    # flag = 0 sl flag = 1 rl
    action = table_sl.select_action(state) if flag == 0 else Agent.rl.select_action(state)
    return action


street_list = []
def store_memory(env, agents):
    # store the memory
    for record_list in env.memory:
        for i in range(len(record_list)):                   
            state, action, reward = record_list[i]
#            if reward[0] > 10 or reward[0] < -10:
#                assert(False)
#                print(reward[0])
            # the next state is the state when agents action next time
            if i+1 >= len(record_list):
                next_state = None
            else:
                next_state,_,_ = record_list[i+1]
#            if reward[0] > 0:
#                print(reward[0])
            # convert to tensor
            # street_list.append(state.street)
            state_tensor = env.state2tensor(state)
            action_tensor = arguments.LongTensor([[action]])
            next_state_tensor = env.state2tensor(next_state)
#            reward.div_(arguments.stack*game_settings.player_count)
            
#            agents[state.current_player].rl.memory.push(state_tensor, action_tensor, next_state_tensor, reward)
#            print(reward[0])
            agents.rl.memory.push(state_tensor, action_tensor, next_state_tensor, reward.unsqueeze(1) / arguments.stack)
            
def record_reward(agents, env, rewards):
    iter_count = 0
    for record_list in env.memory:
        state, _, _, reward = record_list[-1]
        rewards[agents[state.current_player].ID]  = rewards[agents[state.current_player].ID] + reward
        iter_count = iter_count + 1
    assert(iter_count == game_settings.player_count)
    

def print_dead(net):
    current_params = parameters_to_vector(net.model.parameters())
    dead_num = np.count_nonzero(np.less(current_params.data.cpu().numpy(), 0))
    if isinstance(net, DQNOptim):
        print('rl_dead: %d' % dead_num)
    else:
        print('sl_dead: %d' % dead_num)
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
# @profile
def main():
    import time
    time_start = time.time()
    table_update_num = 0
    
    if arguments.load_model:
        load_model(arguments.load_model_num)
    Agent = Agents[0]
    
    for i_episode in range(arguments.epoch_count + 1):
#        random.shuffle(Agents)
        
        # Initialize the environment and state
        env.reset()
        state = GameState()
        flag = 0 if random.random() > arguments.eta else 1
#        import nn.Test.test_env as test
#        test.test_five_card(state)
        
        for t in count():
#            Agent = Agents[state.current_player]
            state_tensor = env.state2tensor(state)
            current_player = state.current_player
            # Select and perform an action
#            print(state_tensor.size(1))
            assert(state_tensor.size(1) == 70)

            if flag == 0:
                # sl
#                action = Agents[current_player].sl.select_action(state_tensor)
                action = Agents[0].sl.select_action(state_tensor)
#                print("SL Action:"+str(action[0][0]))
            elif flag == 1:
                #rl
#                action = Agents[current_player].rl.select_action(state_tensor)
                action = Agents[0].rl.select_action(state_tensor, state.current_player)
#                print("RL Action:"+str(action[0][0]))
            else:
                assert(False)

            is_rl = flag == 1 and arguments.rl_model =='reinforce'
            next_state, done, action_taken = env.step(state, action[0][0], is_rl)
            action[0][0] = action_taken 
            
            if flag == 1:
#                if Agents[0].rl.steps_done > Agents[0].rl.EPS_DECAY * 3:
                if Agent.rl.steps_done > arguments.sl_start:
                    # if choose sl store tuple(s,a) in supervised learning memory Msl
                    # state_tensor = env.state2tensor(state)
                    action_tensor = action[0]
                    Agent.sl.memory.push(state_tensor, action_tensor)
#                print('Action:' + str(action[0][0]))

            
            if done:
                if arguments.rl_model == 'dqn':
                    store_memory(env, Agent)
    #                for agent in Agents:
                    if (Agents[0].rl.steps_done - 1) % 200 == 0: # remember that at first the two net shall be the same
                        Agents[0].rl.target_net.load_state_dict(Agents[0].rl.model.state_dict())
                        Agents[0].rl.steps_done += 1 # track always assign problem
                    if len(Agents[0].rl.memory.memory) >= Agents[0].rl.memory.capacity/10 and i_episode % arguments.rl_update == 0:
                        Agents[0].rl.optimize_model()
                elif arguments.rl_model == 'reinforce':
                    Agents[0].rl.finish_episode(env.memory)
                else:
                    raise (NotImplementedError)

                if len(Agent.sl.memory.memory) >= Agent.sl.memory.capacity/10 and i_episode % arguments.sl_update == 0:
                    Agent.sl.optimize_model()
                if i_episode % 100 == 0 and i_episode > 0:
                    # Agent.sl.test(10)
                    Agent.rl.plot_error_vis(i_episode)
                    Agent.sl.plot_error_vis(i_episode)
                    # print dead relu
                    print_dead(Agent.rl)
                    print_dead(Agent.sl)
                    print('episod: %d' % (i_episode))
                                                        # print('episod: %d rl: %d,sl: %d' % (i_episode,\
                    #                                     len(Agents[0].rl.memory),\
                    #                                     len(Agent.sl.memory)))
                    # record the award
#                    record_reward(Agents, env, Reward)
                if i_episode != 0 and i_episode % arguments.save_epoch == 0:
                    save_model(i_episode)
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


            
