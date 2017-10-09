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


# whether to run on GPU
gpu = torch.cuda.is_available()
# list of pot-scaled bet sizes to use in tree
# @field bet_sizing
bet_sizing = [1]
# server running the ACPC dealer
acpc_server = "localhost"
# server port running the ACPC dealer
acpc_server_port = 20000
# the number of betting rounds in the game
streets_count = 2
# the tensor datatype used for storing DeepStack's internal data
Tensor = torch.FloatTensor
LongTensor = torch.LongTensor
# the directory for data files
data_directory = 'Data/'
# the size of the game's ante, in chips
ante = 100
# the size of each player's stack, in chips
stack = 400
# the number of iterations that DeepStack runs CFR for
cfr_iters = 1000
# the number of preliminary CFR iterations which DeepStack doesn't factor into the average strategy (included in cfr_iters)
cfr_skip_iters = 500
# how many poker situations are solved simultaneously during data generation
gen_batch_size = 10
# how many poker situations are used in each neural net training batch
train_batch_size = 100
# path to the solved poker situation data used to train the neural net
data_path = '../Data/TrainSamples/PotBet/'
# path to the neural net model
model_path = '../Data/Models/PotBet/'
# the name of the neural net file
value_net_name = 'final'
# the neural net architecture
net = '{nn.Linear(input_size, 50), nn.PReLU(), nn.Linear(50, output_size)}'
loss_F = F.nll_loss
loss = nn.MSELoss()

# how often to save the model during training
save_epoch = 5000
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
sl_update_num = 5

#load model
load_model = False
load_model_num = 100000

muilt_gpu = False

bet_bucket = 4
bet_bucket_len = int(stack / bet_bucket)


assert(cfr_iters > cfr_skip_iters)
if gpu:
  Tensor = torch.cuda.FloatTensor
  LongTensor = torch.cuda.LongTensor


