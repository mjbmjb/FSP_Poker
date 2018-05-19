
import numpy as np
from itertools import count

from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.optim as optim
import random
import math
import Settings.arguments as arguments
import Settings.game_settings as game_settings

from copy import deepcopy
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.init import normal, calculate_gain, kaiming_normal
from collections import namedtuple
from nn.misc import hard_update, gumbel_softmax, onehot_from_logits

Experience = namedtuple('Experience',
                        ('states', 'actions', 'next_states', 'rewards'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def weights_init(m):
    if isinstance(m, nn.Linear):
        normal(m.weight.data,mean=1e-3, std=1e-3)
        normal(m.bias.data,mean=1e-3, std=1e-3)


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action, dim_same=True):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action

        if dim_same:
            obs_dim = dim_observation * n_agent
            act_dim = self.dim_action * n_agent
        else:
            obs_dim = dim_observation
            act_dim = dim_action

        self.FC1 = nn.Linear(obs_dim, 128)
        self.FC2 = nn.Linear(128+act_dim, 128)
        self.FC3 = nn.Linear(128, 128)
        self.FC4 = nn.Linear(128, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = torch.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 128)
        self.FC2 = nn.Linear(128, 128)
        self.FC3 = nn.Linear(128, dim_action)
        self.logsoftmax = nn.LogSoftmax(1)

    # action output between -2 and 2
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        # result = F.tanh(self.FC3(result))
        result = self.FC3(result)
        # return self.logsoftmax(result)
        return result

    def forward_fc(self, obs):
        result = F.relu(self.FC1(obs))
        result = self.FC2(result)
        return result

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


def one_hot(batch_size, action_dim, action, n_agents=game_settings.player_count):
    y_onehot = arguments.Tensor(batch_size, n_agents, action_dim).zero_()
    y_onehot.scatter_(1, action, 1)

class MADDPG:
    def __init__(self, n_agents=game_settings.player_count,
                 dim_obs=arguments.dim_obs,
                 dim_act=game_settings.actions_count,
                 batch_size=512,
                 capacity=100000,
                 explor_down=1000,
                 episodes_before_train=3000):
        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.obs_vec = None
        self.act_vec = None
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = arguments.gpu
        self.episodes_before_train = episodes_before_train

        self.GAMMA = arguments.gamma
        self.tau = 0.01
        self.explor_down = explor_down

        self.var = [1.0 for i in range(n_agents)]
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 10000

        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs,
                               dim_act) for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.005) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=1e-3) for x in self.actors]

        self.loss = nn.MSELoss

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.viz = None
        self.steps_done = 0
        self.episode_done = 0
        self.current_loss_actor = 0
        self.current_loss_critic = 0

    @classmethod
    def init_from_env(cls, env,episodes_before_train=3000):
        actors = []
        critics = []
        act_vec = []
        obs_vec = []

        n_agent = env.n
        for acsp, obsp in zip(env.action_space, env.observation_space):
            # get action space
            if isinstance(acsp, Discrete):
                get_shape = lambda x: x.n
            else:
                raise NotImplementedError
            # get obs dim
            obs_dim = obsp.shape[0]
            # get critic dim
            critic_obs_dim = 0
            critic_ac_dim = 0
            for acsp_, obsp_ in zip(env.action_space, env.observation_space):
                critic_obs_dim += obsp_.shape[0]
                critic_ac_dim += get_shape(acsp_)
            # actor and critic
            # actors.append(Actor(obs_dim, get_shape(acsp)).apply(weights_init))
            actors.append(Actor(obs_dim, get_shape(acsp)))
            critics.append(Critic(n_agent,critic_obs_dim,critic_ac_dim,dim_same=False))

        instance = cls(episodes_before_train=episodes_before_train)
        instance.n_agents = n_agent
        instance.actors = actors
        instance.critics = critics

        instance.actors_target = deepcopy(instance.actors)
        instance.critics_target = deepcopy(instance.critics)

        instance.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.0001) for x in instance.critics]
        instance.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.00001) for x in instance.actors]

        if instance.use_cuda:
            for x in instance.actors:
                x.cuda()
            for x in instance.critics:
                x.cuda()
            for x in instance.actors_target:
                x.cuda()
            for x in instance.critics_target:
                x.cuda()

        return instance

    def plot_error_vis(self, step):
        if self.episode_done == 0:
            return
        if not self.viz:
            import visdom
            self.viz = visdom.Visdom()
            self.actor_win = self.viz.line(X=np.array([self.episode_done]),
                                     Y=np.array([self.current_loss_actor]))
            self.critic_win = self.viz.line(X=np.array([self.episode_done]),
                                     Y=np.array([self.current_loss_critic]))

        self.viz.line(
                 X=np.array([self.episode_done]),
                 Y=np.array([self.current_loss_actor]),
                 win=self.actor_win,
                 update='append')
        self.viz.line(
                 X=np.array([self.episode_done]),
                 Y=np.array([self.current_loss_critic]),
                 win=self.critic_win,
                 update='append')


    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = Variable(torch.stack(batch.states).type(FloatTensor))
            action_batch = Variable(torch.stack(batch.actions))
            reward_batch = Variable(torch.stack(batch.rewards).type(FloatTensor))
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = Variable(torch.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor))

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            # current_player_action = whole_action[:,agent * 7:(agent+1) * 7]
            # current_Q = (self.critics[agent](whole_state, whole_action) * action_batch[:,agent,:]).sum(1)
            current_Q = self.critics[agent](whole_state, whole_action).squeeze()
            # current_playe action whole_action[:,agent * 7:(agent+1) * 7]
            non_final_next_actions = [
                self.select_action(i,
                                   non_final_next_states[:,i,:],
                                   is_target=True)[1] for i in range(self.n_agents)]

            non_final_next_actions = torch.stack(non_final_next_actions)
#           non_final_next_actions = Variable(non_final_next_actions)
#             non_final_next_actions = (
#                 non_final_next_actions.transpose(0,
#                                                  1).contiguous())

            target_Q = Variable(torch.zeros(
                self.batch_size).type(FloatTensor))
            # target_Q[non_final_mask] = (self.critics_target[agent](
            #     non_final_next_states.view(-1, self.n_agents * self.n_states),
            #     non_final_next_actions.view((-1,
            #                                 self.n_agents * self.n_actions))) * non_final_next_actions[agent]).sum(1)
            #  TODO 之前这里因为输入不是one-hot，所以考虑只输出当前动作的Q值
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view((-1,
                                            self.n_agents * self.n_actions))).squeeze()

            # scale_reward: to scale reward in Q functions
            target_Q = (target_Q * self.GAMMA) + reward_batch[:, agent].squeeze()

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            # TODO HuBer Loss
            # loss_Q = F.smooth_l1_loss(current_Q, target_Q.detach())

            loss_Q.backward()
            for param in self.critics[agent].parameters():
                param.grad.data.clamp_(-1, 1)
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            # get Gumbel-Softmax sample
            gum_action = gumbel_softmax(action_i, temperature=(np.exp(np.log(10)-self.steps_done/self.explor_down) + 1), hard=True)
            # gum_action = gumbel_softmax(action_i,
            #                             temperature=1,
            #                             hard=True)

            # softmax_action = F.softmax(action_i)
            # m = Categorical(softmax_action)
            # act = m.sample().view((-1, 1))
            # # produce ont hot
            # one_hot = torch.eye(softmax_action.size(1)).cuda()[act].squeeze(1)
            # log_pro = m.log_prob(act.squeeze())

            # log_pro = torch.log(gumbel_softmax(action_i, temperature=(1 / np.sqrt(self.steps_done*1e-1))))
            ac = action_batch.clone()
            # ac[:, agent, :] = one_hot
            ac[:, agent, :] = gum_action
            whole_action = ac.view(self.batch_size, -1)

            actor_loss = -self.critics[agent](whole_state, whole_action)
            # actor_loss = -self.critics[agent](whole_state, whole_action.detach()) * log_pro.unsqueeze(1)
            # actor_loss = actor_loss.sum()
            # 尝试和reinforce一样的做法
            actor_loss = actor_loss.mean()
            # actor_loss += (action_i ** 2).mean() * 1e-3
            actor_loss.backward()
            # 降一下梯度吧.....
            for param in self.actors[agent].parameters():
                param.grad.data.clamp_(-1, 1)

            # torch.nn.utils.clip_grad_norm(self.actors[agent].parameters(), 0.5)

            self.actor_optimizer[agent].step()
            # c_loss.append(loss_Q.data[0].sum())
            # a_loss.append(actor_loss.data[0].sum())
            self.current_loss_critic = loss_Q.data.sum()
            self.current_loss_actor = actor_loss.data.sum()

        self.steps_done += 1
        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, i_agent, state_batch, is_target=False, sto=False):
        temperature = np.exp(np.log(10)-self.steps_done/self.explor_down) + 1
        # return batch X action_dim
        # state_batch: n_agents x state_dim
        if is_target:
            policy = self.actors_target[i_agent](state_batch).data
        else:
            policy = self.actors[i_agent](state_batch).data
        # policy 可能为nan
        if np.any(np.isnan(policy.cpu().numpy())):
            assert (False)
        policy = F.softmax(policy / temperature)
        # assert((policy >= 0).sum() == 7)
        if sto: # stochastic process choose action based on action distribution
            m = Categorical(policy)
            act = m.sample().view((-1, 1))
            # produce ont hot
            one_hot = torch.eye(policy.size(1)).cuda()[act].squeeze(1)
        else:   # determinisitc process choose the max softmax action
            # TODO act remains None
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                       math.exp(-1. * self.steps_done / self.EPS_DECAY)
            one_hot = onehot_from_logits(policy,eps=eps_threshold).cuda()
            act = np.argmax(one_hot, 1).view((-1, 1)).cuda()

        return act, one_hot

    def save(self, path):
        for i in range(self.n_agents):
            save_path = path + '_' + str(i) + '.'
            torch.save(self.actors[i].state_dict(), save_path+'actor')
            torch.save(self.critics[i].state_dict(), save_path + 'critic')

    def load(self, path):
        for i in range(self.n_agents):
            save_path = path + '_' + str(i) + '.'
            self.actors[i].load_state_dict(torch.load(save_path+'actor'))
            self.critics[i].load_state_dict(torch.load(save_path+'critic'))

