# -*- coding: utf-8 -*-
"""
Reinforcement Learning (DQN) tutorial
=====================================
**Author**: `Adam Paszke <https://github.com/apaszke>`_


This tutorial shows how to use PyTorch to train a Deep Q Learning (DQN) agent
on the CartPole-v0 task from the `OpenAI Gym <https://gym.openai.com/>`__.

**Task**

The agent has to decide between two actions - moving the cart left or
right - so that the pole attached to it stays upright. You can find an
official leaderboard with various algorithms and visualizations at the
`Gym website <https://gym.openai.com/envs/CartPole-v0>`__.

.. figure:: /_static/img/cartpole.gif
   :alt: cartpole

   cartpole

As the agent observes the current state of the environment and chooses
an action, the environment *transitions* to a new state, and also
returns a reward that indicates the consequences of the action. In this
task, the environment terminates if the pole falls over too far.

The CartPole task is designed so that the inputs to the agent are 4 real
values representing the environment state (position, velocity, etc.).
However, neural networks can solve the task purely by looking at the
scene, so we'll use a patch of the screen centered on the cart as an
input. Because of this, our results aren't directly comparable to the
ones from the official leaderboard - our task is much harder.
Unfortunately this does slow down the training, because we have to
render all the frames.

Strictly speaking, we will present the state as the difference between
the current screen patch and the previous one. This will allow the agent
to take the velocity of the pole into account from one image.

**Packages**


First, let's import needed packages. Firstly, we need
`gym <https://gym.openai.com/docs>`__ for the environment
(Install using `pip install gym`).
We'll also use the following from PyTorch:

-  neural networks (``torch.nn``)
-  optimization (``torch.optim``)
-  automatic differentiation (``torch.autograd``)
-  utilities for vision tasks (``torchvision`` - `a separate
   package <https://github.com/pytorch/vision>`__).

"""


import math
import random
import numpy as np
from collections import namedtuple
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

import Settings.arguments as arguments
import Settings.game_settings as game_settings
from torch.nn.init import normal_

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classses:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def weights_init(m):
    if isinstance(m, nn.Linear):
        normal_(m.weight.data, mean=0.3, std=0.1)
        normal_(m.bias.data, mean=0, std=0.1)

#def reservoir_sample(data, K):
#    sample = []
#    for i,line in enumerate(data):
#        if i < K:
#            sample.append(line)
#        elif i >= K and random.random() < K/float(i+1):
#            replace = random.randint(0,len(sample)-1)
#            sample[replace] = line
#    return sample

class ReplayMemory(object):

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


######################################################################
# Now, let's define our model. But first, let quickly recap what a DQN is.
#
# DQN algorithm
# -------------
#
# Our environment is deterministic, so all equations presented here are
# also formulated deterministically for the sake of simplicity. In the
# reinforcement learning literature, they would also contain expectations
# over stochastic transitions in the environment.
#
# Our aim will be to train a policy that tries to maximize the discounted,
# cumulative reward
# :math:`R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t`, where
# :math:`R_{t_0}` is also known as the *return*. The discount,
# :math:`\gamma`, should be a constant between :math:`0` and :math:`1`
# that ensures the sum converges. It makes rewards from the uncertain far
# future less important for our agent than the ones in the near future
# that it can be fairly confident about.
#
# The main idea behind Q-learning is that if we had a function
# :math:`Q^*: State \times Action \rightarrow \mathbb{R}`, that could tell
# us what our return would be, if we were to take an action in a given
# state, then we could easily construct a policy that maximizes our
# rewards:
#
# .. math:: \pi^*(s) = \arg\!\max_a \ Q^*(s, a)
#
# However, we don't know everything about the world, so we don't have
# access to :math:`Q^*`. But, since neural networks are universal function
# approximators, we can simply create one and train it to resemble
# :math:`Q^*`.
#
# For our training update rule, we'll use a fact that every :math:`Q`
# function for some policy obeys the Bellman equation:
#
# .. math:: Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))
#
# The difference between the two sides of the equality is known as the
# temporal difference error, :math:`\delta`:
#
# .. math:: \delta = Q(s, a) - (r + \gamma \max_a Q(s', a))
#
# To minimise this error, we will use the `Huber
# loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts
# like the mean squared error when the error is small, but like the mean
# absolute error when the error is large - this makes it more robust to
# outliers when the estimates of :math:`Q` are very noisy. We calculate
# this over a batch of transitions, :math:`B`, sampled from the replay
# memory:
#
# .. math::
#
#    \mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta) 
#
# .. math::
#
#    \text{where} \quad \mathcal{L}(\delta) = \begin{cases}
#      \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
#      |\delta| - \frac{1}{2} & \text{otherwise.}
#    \end{cases}
#
# Q-network
# ^^^^^^^^^
#
# Our model will be a convolutional neural network that takes in the
# difference between the current and previous screen patches. It has two
# outputs, representing :math:`Q(s, \mathrm{left})` and
# :math:`Q(s, \mathrm{right})` (where :math:`s` is the input to the
# network). In effect, the network is trying to predict the *quality* of
# taking each action given the current input.
#

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(arguments.dim_obs, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        # self.fc2 = nn.Linear(64,64)
        # self.fc2_bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(512,512)
        self.fc3_bn = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc1_bn(x))
        # x = self.fc2(x)
        # x = F.relu(self.fc2_bn(x))
        x = self.fc3(x)
        x = F.relu(self.fc3_bn(x))
        
#        return self.out(x.view(x.size(0), -1))
        return self.out(x)

    def forward_fc(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc1_bn(x))
        # x = self.fc2(x)
        # x = F.relu(self.fc2_bn(x))
        x = self.fc3(x)
        x = F.relu(self.fc3_bn(x))
        return x


class DQNOptim:
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
    
    def __init__(self, lr=1e-4):
        
        self.BATCH_SIZE = 128
        self.GAMMA = 1.0
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 2000
        
        self.model = DQN()
        self.target_net = DQN()
        
        # init weight and baise
        self.model.apply(weights_init)
        self.target_net.apply(weights_init)
        
        if use_cuda:
            self.model.cuda()
            self.target_net.cuda()
            
        if arguments.muilt_gpu:
            self.model = nn.DataParallel(self.model)
            self.target_net = nn.DataParallel(self.target_net)
            
        self.lr = lr

        self.optimizer = optim.SGD(self.model.parameters(),self.lr,weight_decay=0.0001)
        self.memory = ReplayMemory(100000)
        
        
        self.steps_done = 0
        self.episode_durations = []
        self.error_acc = []
    
        
        self.viz = None
        self.win = None
        self.current_sum = 0.1
        
    
    # @return action LongTensor[[]]
    def select_action(self, state, *useless):
        self.model.eval() # to use the batchNorm correctly
        sample = random.random()
#        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
#            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        eps_threshold = 0.06 / np.sqrt(self.steps_done)
#        self.steps_done += 1
        if sample > eps_threshold:
            action = self.model(
                Variable(state)).data.max(1)[1].view((1, 1))
            action = arguments.LongTensor(action)

        else:
            m = Categorical(arguments.dqn_init_policy)
            action = m.sample().view((1,1))
        
        one_hot = torch.eye(game_settings.actions_count)[action.cpu()].squeeze(1)
        if arguments.gpu:
            return action.cuda(), one_hot.cuda()
        return action , one_hot
    
    
    def plot_error(self):
        self.plt.figure(2)
        self.plt.clf()
        error_acc_t = torch.FloatTensor(self.error_acc)
        self.plt.title('Training...')
        self.plt.xlabel('Episode')
        self.plt.ylabel('error')
        self.plt.plot(error_acc_t.numpy())
        # Take 100 episode averages and plot them too
#        if len(durations_t) >= 100:
#            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#            means = torch.cat((torch.zeros(99), means))
#            self.plt.plot(means.numpy())
    
        self.plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(self.plt.gcf())
    
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
        
    
    ######################################################################
    # Training loop
    # ^^^^^^^^^^^^^
    #
    # Finally, the code for training our model.
    #
    # Here, you can find an ``optimize_model`` function that performs a
    # single step of the optimization. It first samples a batch, concatenates
    # all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
    # :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
    # loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
    # state.
    
    
    last_sync = 0
    
#    @profile
    def optimize_model(self):
        self.model.train()# to use the batchNorm correctly
        
        self.steps_done += 1
        global last_sync
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        
        
        
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))
    
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))
    
        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
#        if len([s for s in batch.next_state if s is not None]) == 0 :
#            return
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]))
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
    
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.model(state_batch).gather(1, action_batch)
    
        #modify for 2 player
#        state_action_values = self.model(state_batch).gather(1, action_batch).squeeze()
    
        with torch.no_grad():
            # Compute V(s_{t+1}) for all next states.
            next_state_values = Variable(torch.zeros(self.BATCH_SIZE,1).type(Tensor))
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].unsqueeze(1)
#           next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
            # Now, we don't want to mess up the loss with a volatile flag, so let's
            # clear it. After this, we'll just end up with a Variable that has
            # requires_grad=False
#            next_state_values.volatile = False
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

    
#        # Compute Huber loss
#        loss = arguments.loss_F(state_action_values, expected_state_action_values)

#        if self.steps_done % 300 == 0:
#            print(state_action_values[0:2])

        loss = arguments.loss(state_action_values, expected_state_action_values)

        # current_params = parameters_to_vector(self.model.parameters())
        # if any(np.less(current_params.data.cpu().numpy(),0)):
        #     print("RELU optimization diverged. Rolling back update...")
#        print(len(loss.data))
     
#        self.error_acc.append(loss.data.sum())
#        self.current_sum = (self.steps_done / (self.steps_done + 1.0)) * self.current_sum + loss.data[0]/(self.steps_done + 1)
#        print(self.steps_done)
#        print(self.current_sum)
        self.current_sum = loss.data.item()
#        print(self.current_sum)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()
    
