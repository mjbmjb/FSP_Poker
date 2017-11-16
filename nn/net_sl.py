#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 02:20:21 2017

@author: mjb
"""
import sys
sys.path.append('/home/mjb/Nutstore/deepStack/')

import random
from collections import namedtuple
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import Settings.arguments as arguments
import Settings.game_settings as game_settings

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor



class SLNet(nn.Module):
    def __init__(self):
        super(SLNet, self).__init__()
        
        self.fc1 = nn.Linear(27,64)
        self.fc2 = nn.Linear(64,64)
        self.output = nn.Linear(64,4)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.output(x)
        output = self.softmax(output)
        return output

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
    
    def __init__(self):
        
        self.BATCH_SIZE = 256
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.00
        self.EPS_DECAY = 50
        
        self.model = SLNet()
        
        if use_cuda:
            self.model.cuda()
            
        if arguments.muilt_gpu:
            self.model = nn.DataParallel(self.model)

            
            
        self.optimizer = optim.ASGD(self.model.parameters(),lr=0.01)
        self.memory = Memory(100000)
        self.loss = nn.CrossEntropyLoss()
        
        
        self.steps_done = 0
        self.episode_durations = []
        self.error_acc = []
    
        
        self.viz = None
        self.win = None
        self.current_sum = 0.1
    
    # @return action LongTensor[[]]
    def select_action(self, state):
        policy = self.model(Variable(state)).data
        action = arguments.LongTensor([np.random.choice(np.arange(game_settings.actions_count),\
                                                         1,\
                                                         replace=False,\
                                                         p=policy.cpu().numpy()[0]).tolist()])
        return action
    
    
    def plot_error_vis(self, step):
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
        self.steps_done += 1
        global last_sync
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))
    

        state_batch = Variable(torch.cat(batch.state), volatile=False)
        excepted_policy_batch = Variable(torch.cat(batch.policy), volatile=False)
    
        policy_batch = self.model(state_batch)
        
        output = self.loss(policy_batch, excepted_policy_batch)
        
#        print(len(loss.data))
     
#        self.error_acc.append(loss.data.sum())
#        self.current_sum = (self.steps_done / (self.steps_done + 1.0)) * self.current_sum + loss.data[0]/(self.steps_done + 1)
#        print(self.steps_done)
#        print(self.current_sum)
        self.current_sum = output.data[0]
#        print(self.current_sum)

        # Optimize the model
        self.optimizer.zero_grad()
        output.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
