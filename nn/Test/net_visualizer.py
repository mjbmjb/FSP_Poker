# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pdb

# In[2]:


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:37:03 2017

@author: mjb
"""
import sys

sys.path.append('../../')

import torch
import numpy as np
import Settings.arguments as arguments
import Settings.game_settings as game_settings
import Settings.constants as constants
# from Player.six_player_machine import SixPlayerMachine
# from ACPC.six_acpc_game import SixACPCGame
from Tree.game_state import GameState
from Tree.game_state import Action
from nn.sim_env import SimEnv

from nn.dqn import *
from nn.table_sl import TableSL
from nn.net_sl import SLOptim
from nn.maddpg import MADDPG

import seaborn as sns
from sklearn import manifold
import pandas as pd

# In[3]:


# iter_num = [10000,20000,30000]
iter_num = [50000, 60000, 70000]


def load_model(episoid):
    net_sl = SLOptim()
    net_rl = DQNOptim()
    maddpg = MADDPG()

    #     net_sl.model.load_state_dict(torch.load( \
    #                                 arguments.WORK_PATH+'/Data/Model/Iter:' + str(episoid) + '_' + str(0) +'_' + '.sl'))
    #     net_sl.model.eval()
    #     net_rl.model.load_state_dict(torch.load( \
    #                                 arguments.WORK_PATH+'/Data/Model/Iter:' + str(episoid) + '_' + str(0) +'_' + '.rl'))
    #     net_rl.model.eval()

    for i in range(game_settings.player_count):
        maddpg.actors[i].load_state_dict(
            torch.load(arguments.WORK_PATH + '/Data/Model/Iter:' + str(episoid) + '_' + str(i) + '.actor'))
        maddpg.actors[i].load_state_dict(
            torch.load(arguments.WORK_PATH + '/Data/Model/Iter:' + str(episoid) + '_' + str(i) + '.actor'))
    return net_sl, net_rl, maddpg


state = GameState()
call = Action(atype=constants.actions.ccall, amount=0)
rrasie = Action(atype=constants.actions.rraise, amount=1000)
fold = Action(atype=constants.actions.fold, amount=0)

env = SimEnv()


def make_data(size=10000):
    cat = []
    data = []

    for _ in range(size):
        state.street = np.random.randint(2)
        state.current_player = np.random.randint(3)
        state.terminal = True

        state.hole = torch.LongTensor(3, 1).fill_(0)
        state.hole[state.current_player][0] = np.random.randint(10)
        # board = torch.LongTensor([6,30,31,38,43])
        state.board = torch.LongTensor([6])
        state.bets = arguments.LongTensor(np.random.randint(arguments.stack, size=3))
        state.do_action(call)
        state.do_action(rrasie)
        state.do_action(call)
        state.do_action(fold)

        state_tensor = env.state2tensor(state)

        cat.append((state.hole[state.current_player].item(), state.street))
        data.append(state_tensor)

    return cat, data


def plot_tsne(forward_data, target):
    tsne = manifold.TSNE(n_components=2, init='pca',
                         random_state=0, perplexity=50)
    fc1_x = tsne.fit_transform(forward_data)

    df_data = np.append(fc1_x, np.array(target).reshape((-1, 1)), axis=1)
    df = pd.DataFrame(columns=['x', 'y', 'tar'], data=df_data)
    #     df['tar'] = (df['tar'] / 4).astype(int)
    sns.lmplot(x='x', y='y', data=df,
               fit_reg=False,  # No regression line
               hue='tar',
               scatter_kws={"s": 5})  # Color by evolution stage
    return df


cat, data = make_data(5000)
hole_tar, street_tar = list(zip(*cat))

# 读取模型参数，并利用上一个cell生成的数据画tsne图

# In[ ]:


all_data = []
for it in iter_num:
    net_sl, net_rl, maddpg = load_model(it)
    net_sl = SLOptim()
    net_sl.model.eval()
    net_rl = DQNOptim()
    net_rl.model.eval()
    # print(state.bets)

    forward_data = []

    for state_tensor in data:
        forward_data.append(net_rl.model.forward_fc(Variable(state_tensor)).data)
    #         forward_data.append(maddpg.actors[0].forward_fc(Variable(state_tensor)).data)

    forward_data = np.vstack(forward_data)
    all_data.append(forward_data)

# In[ ]:


for data in all_data:
    plot_tsne(data, street_tar)

# In[ ]:


df_data[0].head(20)

# 测试网络 dead relu的个数

# In[ ]:


for it in iter_num:
    net_sl, net_rl, maddpg = load_model(it)
    for p in range(game_settings.player_count):
        current_params = parameters_to_vector(maddpg.actors[p].parameters())
        current_params = parameters_to_vector(maddpg.critics[p].parameters())
        dead_num = np.count_nonzero(np.less(current_params.data.cpu().numpy(), 0))
        print('total: %d dead: %d' % (current_params.shape[0], dead_num))

# In[ ]:


for state_tensor in data:
    print(maddpg.actors[2](Variable(state_tensor)).data)

