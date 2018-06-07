# -*- coding: utf-8 -*-
import torch
from torch import nn

class ActorCritic(nn.Module):
  def __init__(self, observation_dim, action_dim, hidden_size):
    super(ActorCritic, self).__init__()
    self.state_size = observation_dim
    self.action_size = action_dim

    self.relu = nn.ReLU(inplace=True)
    self.softmax = nn.Softmax(dim=1)

    self.fc1 = nn.Linear(self.state_size, hidden_size)
    self.lstm = nn.LSTMCell(hidden_size, hidden_size)
    self.fc_actor = nn.Linear(hidden_size, self.action_size)
    self.fc_critic = nn.Linear(hidden_size, self.action_size)

  def forward(self, x, h):
    x = self.relu(self.fc1(x))
    h = self.lstm(x, h)  # h is (hidden state, cell state)
    x = h[0]
    policy = self.softmax(self.fc_actor(x)).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    Q = self.fc_critic(x)
    V = (Q * policy).sum(1, keepdim=True)  # V is expectation of Q under π
    return policy, Q, V, h

class MCritic(nn.Module):
  def __init__(self, n_agent, state_dim, action_dim, hidden_size, output_size=1):
    super(MCritic, self).__init__()
    self.n_agent = n_agent
    self.state_size = state_dim * n_agent
    self.action_size = action_dim * n_agent

    self.relu = nn.ReLU(inplace=True)

    self.fc1 = nn.Linear(self.state_size, hidden_size)
    self.lstm = nn.LSTMCell(hidden_size + self.action_size, hidden_size)
    # self.fc_actor = nn.Linear(hidden_size, self.action_size)
    self.fc_critic = nn.Linear(hidden_size, output_size)

  def forward(self, x, a, h):
    x = self.relu(self.fc1(x))
    combined = torch.cat([x, a], 1)
    h = self.lstm(combined, h)  # h is (hidden state, cell state)
    x = h[0]
    V = self.fc_critic(x)
    return V, h

class Actor(nn.Module):
  def __init__(self, observation_dim, action_dim, hidden_size, output_act):
    super(Actor, self).__init__()
    self.state_size = observation_dim
    self.action_size = action_dim

    self.relu = nn.ReLU(inplace=True)
    self.output_act = output_act

    self.fc1 = nn.Linear(self.state_size, hidden_size)
    self.lstm = nn.LSTMCell(hidden_size, hidden_size)
    self.fc_actor = nn.Linear(hidden_size, self.action_size)

  def forward(self, x, h):
    x = self.relu(self.fc1(x))
    h = self.lstm(x, h)  # h is (hidden state, cell state)
    x = h[0]
    policy = self.output_act(self.fc_actor(x), dim=1)  # Prevent 1s and hence NaNs
    return policy, h

