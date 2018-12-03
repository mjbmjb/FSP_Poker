#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 02:20:21 2017

@author: mjb
"""
import sys
sys.path.append('../')

import random
from collections import namedtuple
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.init import normal_

from sklearn.metrics import log_loss

import Settings.arguments as arguments
import Settings.game_settings as game_settings

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def weights_init(m):
    if isinstance(m, nn.Linear):
        normal_(m.weight.data,mean=0.0, std=0.1)
        normal_(m.bias.data,mean=0.0, std=0.1)

def reservoir_sample(memory, K):
    data = memory.memory
    sample = []
    for i in range(len(memory.memory)):
        if len(memory.memory) >= memory.capacity:
            record = memory.memory[(i + memory.position) % memory.capacity]
        else:
            record = memory.memory[i]
        if i < K:
            sample.append(record)
        elif i >= K and random.random() < K/float(i+1):
            replace = random.randint(0,len(sample)-1)
            sample[replace] = record
    # memory.memory = []
    # memory.position = 0
    
#    print('sample sl len:%d'%len(sample))
    return sample

class SLNet(nn.Module):
    def __init__(self, dim=arguments.dim_obs):
        super(SLNet, self).__init__()
        self.fc1 = nn.Linear(dim,512)
        self.fc1_bn = nn.BatchNorm1d(512)
        # self.fc2 = nn.Linear(64,64)
        # self.fc2_bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(512,512)
        self.fc3_bn = nn.BatchNorm1d(512)
        self.output = nn.Linear(512,5)
        self.logsoftmax = nn.functional.log_softmax
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc1_bn(x))
#         x = self.fc2(x)
#         x = F.dropout(x, p=0.2)
#         x = F.relu(self.fc2_bn(x))

        x = self.fc3(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc3_bn(x))
#        x = F.relu(x)
        output = self.output(x)
        output = self.logsoftmax(output,dim=1)
        return output

    def forward_fc(self, x):
        x = self.fc1(x)
        # x = F.relu(self.fc1_bn(x))

        # x = self.fc2(x)
        # x = F.dropout(x, p=0.2)
        # x = F.relu(self.fc2_bn(x))
        # x = self.fc3(x)
        # x = F.relu(self.fc3_bn(x))
        return x


Transition = namedtuple('Transition',
                        ('state', 'policy'))


class Memory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
        
class SLOptim:
    ######################################################################
    # Training
    # --------
    #
    # Hyperparameters and utilities
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # This cell instantiates our model and its optimizer, and defines some
    # utilities:
    #
    # -  ``Variable`` - this is a simple wrapper around
    #    ``torch.autograd.Variable`` that will automatically send the data to
    #    the GPU every time we construct a Variable.
    # -  ``select_action`` - will select an action accordingly to an epsilon
    #    greedy policy. Simply put, we'll sometimes use our model for choosing
    #    the action, and sometimes we'll just sample one uniformly. The
    #    probability of choosing a random action will start at ``EPS_START``
    #    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
    #    controls the rate of the decay.
    # -  ``plot_durations`` - a helper for plotting the durations of episodes,
    #    along with an average over the last 100 episodes (the measure used in
    #    the official evaluations). The plot will be underneath the cell
    #    containing the main training loop, and will update after every
    #    episode.
    #
    
    def __init__(self,state_dim=arguments.dim_obs, batch_size = arguments.batch_size):
        
        self.BATCH_SIZE = batch_size
        
        self.model = SLNet(dim=state_dim)
        
        # init weight and baise
        self.model.apply(weights_init)
        
        if use_cuda:
            self.model.cuda()
            
        if arguments.muilt_gpu:
            self.model = nn.DataParallel(self.model)

        # self.optimizer = optim.Adam(self.model.parameters(),lr=0.00001,weight_decay=0.0005)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0.0001)
        self.memory = Memory(50000)
        self.loss = nn.NLLLoss()
        self.max_grad_norm = 0.5
        
        
        self.steps_done = 0
        self.episode_durations = []
        self.error_acc = []
    
        
        self.viz = None
        self.win = None
        self.current_sum = 0.1
    
    
    # @return action LongTensor(1,1)
    def select_action(self, state):
        self.model.eval() # to use the batchNorm correctly
        policy = self.model(Variable(state)).data
        #convert log(softmax) to softmax
        policy = torch.exp(policy)
#        assert((policy >= 0).sum() == 7)
        m = Categorical(policy)
        action = m.sample().view((1, 1))
        one_hot = torch.eye(policy.size(1)).cuda()[action].squeeze(1)
        if arguments.gpu:
            return action.cuda(), one_hot
        return action , one_hot.cpu()

    
    def plot_error_vis(self, step):
        if self.steps_done == 0:
            return 
        if not self.viz:
            import visdom
            self.viz = visdom.Visdom()
            self.win = self.viz.line(X=np.array([self.steps_done]),
                                     Y=np.array([self.current_sum]))
        if step % 10000 == 0:
            self.viz.updateTrace(
                 X=np.array([self.steps_done]),
                 Y=np.array([self.current_sum]),
                 win=self.win)
        else:
            self.viz.line(
                 X=np.array([self.steps_done]),
                 Y=np.array([self.current_sum]),
                 win=self.win,
                 update='append')
        
    

    last_sync = 0
    
#    @profile
    def optimize_model(self):
        self.model.train()# to use the batchNorm correctly
        
        self.steps_done += 1
        global last_sync
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        if arguments.reservoir:
            transitions = reservoir_sample(self.memory, self.BATCH_SIZE)
        else:
            transitions = self.memory.sample(self.BATCH_SIZE)
        
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))
    

        state_batch = Variable(torch.cat(batch.state))
        with torch.no_grad():
            excepted_policy_batch = Variable(torch.cat(batch.policy)).squeeze()
    
        policy_batch = self.model(state_batch)
        
        output = self.loss(policy_batch, excepted_policy_batch)
        
#        print(len(loss.data))
#        print(excepted_policy_batch)
#        print(torch.exp(policy_batch[0:2]))
#        self.error_acc.append(loss.data.sum())
#        self.current_sum = (self.steps_done / (self.steps_done + 1.0)) * self.current_sum + loss.data[0]/(self.steps_done + 1)
#        print(self.steps_done)
#        print(self.current_sum)
        self.current_sum = output.data.item()
#        print(self.current_sum)
#         if self.current_sum > 0.8 and self.steps_done > 200:
            # print(excepted_policy_batch)

        # Optimize the model
        self.optimizer.zero_grad()
        output.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def test(self, batch_size=100):
        self.model.eval()
        if arguments.reservoir:
            transitions = reservoir_sample(self.memory, batch_size)
        else:
            transitions = self.memory.sample(batch_size)

        batch = Transition(*zip(*transitions))

        state_batch = Variable(torch.cat(batch.state))
        excepted_policy_batch = Variable(torch.cat(batch.policy))

        policy_batch = self.model(state_batch)

        log_loss()