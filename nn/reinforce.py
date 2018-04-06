
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from collections import namedtuple

import Settings.arguments as arguments

reinf_tran = namedtuple('reinf_tran',['state','action','reward'])

class Policy(nn.Module):
    def __init__(self, n_agents=3):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(70, 64)
        self.fc1_bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 64)
        self.fc3_bn = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, 5)
        self.softmax = nn.Softmax()

        self.saved_log_probs = [[] for _ in range(n_agents)]

    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc1_bn(x))
        #        x = F.relu(x)
        x = self.fc2(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc2_bn(x))
        #        x = F.relu(x)
        x = self.fc3(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc3_bn(x))
        #        x = F.relu(x)
        output = self.output(x)
        output = self.softmax(output)
        return output


class ReinforceOptim(object):
    def __init__(self, lr=0.0001):
        self.model = Policy()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if arguments.gpu:
            self.model.cuda()

        self.steps_done = 0
        self.viz = None
        self.current_sum = 0

    def select_action(self, state, i_agnet):
        self.model.eval()
        state = state
        probs = self.model(Variable(state))
        m = Categorical(probs)
        action = m.sample()
        self.model.saved_log_probs[i_agnet].append(m.log_prob(action))
        return action.data[0]

    def finish_episode(self, env_memory):
        self.model.train()
        self.steps_done += 1

        policy_loss = []

        for i_agnet in range(len(env_memory)):
            if len(env_memory[i_agnet]) == 0: continue
            env_reward = reinf_tran(*zip(*env_memory[i_agnet])).reward
            rewards = []
            R = 0
            for r in env_reward:
                R = r + arguments.gamma * R
                rewards.insert(0, R)
            rewards = arguments.Tensor(rewards)
            # rewards = (rewards - rewards.mean()) / (rewards.std().item() + np.finfo(np.float32).eps)
            rewards = rewards / arguments.stack
            for log_prob, reward in zip(self.model.saved_log_probs[i_agnet], rewards):
                policy_loss.append(-log_prob * reward)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(0,1)
        self.optimizer.step()
        for i_policy, _ in enumerate(self.model.saved_log_probs):
            self.model.saved_log_probs[i_policy] = []


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