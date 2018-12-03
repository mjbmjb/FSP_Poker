#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 00:50:31 2017

@author: mjb
"""

# Parameters for DeepStack.
# @module arguments

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

#WORK_PATH = '/home/mjb/Nutstore/deepStack'
WORK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/'
# whether to run on GPU
# gpu = torch.cuda.is_available()
gpu = True
device = torch.device("cuda:0" if gpu else "cpu")
#
num_process = 15
# gpu = False
cpu_store = False
multi_agent = True
# list of pot-scaled bet sizes to use in tree

# dict of model
rl_model = 'dqn'
#rl_model = 'maddpg'
sl_model = 'mlr'

# culcate by how many step per episoid
rl_update = 100
sl_update = 100
batch_size = 128
# params for rl
gamma = 0.99

dim_obs = 133
# params for sl
sl_start = 50

# @field bet_sizing
bet_sizing = [1]

blind = 0
# server running the ACPC dealer
acpc_server = "localhost"
# server port running the ACPC dealer
acpc_server_port = 500
# the number of betting rounds in the game
#streets_count = 4
# the tensor datatype used for storing DeepStack's internal data
Tensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
IntTensor = torch.IntTensor
# the directory for data files
data_directory = 'Data/'
# the size of the game's ante, in chips
ante = 100
# the size of each player's stack, in chips
stack = 1200
# the number of iterations that DeepStack runs CFR for
cfr_iters = 1000
# pot times
pot_times = [0.3,0.5,0.75,1]
# the number of preliminary CFR iterations which DeepStack doesn't factor into the average strategy (included in cfr_iters)
cfr_skip_iters = 500
net = '{nn.Linear(input_size, 50), nn.PReLU(), nn.Linear(50, output_size)}'
loss_F = F.nll_loss
loss = nn.MSELoss()
dqn_init_policy = Tensor([0.25,0.25,0.25,0.25])
reservoir = True
# how often to save the model during training
save_epoch = 300
# how many epochs to train for
epoch_count = 200000
# how many solved poker situations are generated for use as training examples
train_data_count = 100
# how many solved poker situations are generated for use as validation examples
valid_data_count = 100
# learning rate for neural net training
learning_rate = 0.001
#
eta = 0.5

# table update number
sl_update_num = 128

evalation = True
load_model = True
load_model_num = 1250
muilt_gpu = False
display = True

bet_bucket = 5
bet_bucket_len = int(stack / bet_bucket)

C_PLAYER = True

assert(cfr_iters > cfr_skip_iters)
if gpu:
  Tensor = torch.cuda.FloatTensor
  LongTensor = torch.cuda.LongTensor
  ByteTensor = torch.cuda.ByteTensor
  IntTensor = torch.cuda.LongTensor

import argparse
def get_args():
  parser = argparse.ArgumentParser(description='RL')
  parser.add_argument('--evaluation', type=bool, default=False,
                        help='is evaluation (default: False)')
  parser.add_argument('--model_num', type=int, default=200,
                        help='model_num (default: 200)')

  args = parser.parse_args()
  return args