import torch as th
from torch import nn


class Actor(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, hidden_size, output_size, output_act):
        super(Actor, self).__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # activation function for the output
        self.output_act = output_act

    def forward(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.output_act(out, dim=0)
        return out

class Critic(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim,hidden_size, output_size=1):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size , hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        out = nn.functional.relu(self.fc1(state))
        # out = th.cat([out, action], 1)
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class MCritic(nn.Module):
    """
    A network for critic
    """
    def __init__(self, n_agent, state_dim, action_dim, hidden_size, output_size=1):
        super(MCritic, self).__init__()
        aciton_size = action_dim * n_agent
        state_size = state_dim * n_agent
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size + aciton_size , hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, states, actions):
        state_out = nn.functional.relu(self.fc1(states))
        # out = th.cat([out, action], 1)
        combined = th.cat([state_out, actions], 1)
        out = nn.functional.relu(self.fc2(combined))
        out = self.fc3(out)
        return out

class ActorCritic(nn.Module):
    """
    An actor-critic network that shared lower-layer representations but
    have distinct output layers
    """
    def __init__(self, state_dim, action_dim, hidden_size,
                 actor_output_act, critic_output_size=1):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear = nn.Linear(hidden_size, action_dim)
        self.critic_linear = nn.Linear(hidden_size, critic_output_size)
        self.actor_output_act = actor_output_act

    def forward(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        act = self.actor_output_act(self.actor_linear(out))
        val = self.critic_linear(out)
        return act, val