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

torch.set_default_tensor_type('torch.FloatTensor')

torch.manual_seed(1234)
#WORK_PATH = '/home/mjb/Nutstore/deepStack'
WORK_PATH = '/home/carc/mjb/deepStack/'
# whether to run on GPU
gpu = torch.cuda.is_available()
# gpu = False
cpu_store = False
multi_agent = True
# list of pot-scaled bet sizes to use in tree

# dict of model
# rl_model = 'dqn'
rl_model = 'maddpg'
sl_model = 'mlr'

# culcate by how many step per episoid
rl_update = 20
sl_update = 40

# params for rl
gamma = 0.99

dim_obs = 133
# params for sl
sl_start = 500

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
dqn_init_policy = Tensor([0.01,0.50,0.45,0.01,0.01,0.01,0.01])
reservoir = False
# how often to save the model during training
save_epoch = 10000
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

#table update number
sl_update_num = 128

#load model
load_model = False
load_model_num = 60000

muilt_gpu = False

bet_bucket = 5
bet_bucket_len = int(stack / bet_bucket)


assert(cfr_iters > cfr_skip_iters)
if gpu:
  Tensor = torch.cuda.FloatTensor
  LongTensor = torch.cuda.LongTensor
  ByteTensor = torch.cuda.ByteTensor
  IntTensor = torch.cuda.LongTensor

