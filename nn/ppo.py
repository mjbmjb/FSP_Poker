
import torch as th
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np
from copy import deepcopy

# from common.Agent import Agent
from nn.mlpac import Actor, Critic
from common.utils import to_tensor, index_to_one_hot, logpro2entropy
from nn.misc import soft_update
import Settings.arguments as arguments

import random
from torch.distributions import Categorical
from collections import  namedtuple
Experience = namedtuple("Experience",
                        ("states", "actions", "rewards", "next_states", "dones"))


class ReplayMemory(object):
    """
    Replay memory buffer
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, action, reward, next_state=None, done=None):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, rewards, next_states=None, dones=None):
        if isinstance(states, list):
            if next_states is not None and len(next_states) > 0:
                for s,a,r, n_s,d in zip(states, actions, rewards, next_states, dones):
                    self._push_one(s, a, r, n_s, d)
            else:
                for s,a,r, in zip(states, actions, rewards):
                    self._push_one(s, a, r)
        else:
            self._push_one(states, actions, rewards, next_states, dones)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)


class PPO:
    """
    An agent learned with PPO using Advantage Actor-Critic framework
    - Actor takes state as input
    - Critic takes both state and action as input
    - agent interact with environment to collect experience
    - agent training with experience to update policy
    - adam seems better than rmsprop for ppo
    """
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=100000, max_steps=None,
                 roll_out_n_steps=1, target_tau=1.,
                 target_update_steps=5, clip_param=0.2,
                 reward_gamma=0.99, reward_scale=1., tau=0.95 ,done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="adam", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=128, episodes_before_train=500,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_state = self.env.reset()
        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.tau = tau
        self.done_penalty = done_penalty

        self.roll_out_n_steps = roll_out_n_steps
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps
        self.clip_param = clip_param

        self.actor_output_act = actor_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train


        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.actor = Actor(self.state_dim, actor_hidden_size,
                                  self.action_dim, actor_output_act)
        self.critic = Critic(self.state_dim, critic_hidden_size, 1)
        # to ensure target network and learning network has the same weights
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        self.memory = ReplayMemory(memory_capacity)

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()

        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.roll_out_n_steps = roll_out_n_steps

        self.viz = None
        self.current_loss_actor = 0
        self.current_loss_critic = 0

    # predict softmax action based on state
    def _softmax_action(self, state):
        state_var = to_tensor([state], self.use_cuda)
        softmax_action_var = th.exp(self.actor(state_var))
        if self.use_cuda:
            softmax_action = softmax_action_var.data.cpu().numpy()[0]
        else:
            softmax_action = softmax_action_var.data.numpy()[0]
        return softmax_action

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state):
        softmax_action = self._softmax_action(state)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                  np.exp(-1. * self.n_steps / self.epsilon_decay)
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = np.argmax(softmax_action)
        return action

    # choose an action based on state for execution
    def action(self, state):
        softmax_action = self._softmax_action(state)
        action = np.argmax(softmax_action)
        return action

    # TODO add entrophy
    def dist_explortation_action(self, state):
        softmax_action = self._softmax_action(state)
        m = Categorical(to_tensor(softmax_action))
        action = m.sample().item()
        return action


    def dist_action(self, state):
        softmax_action = self._softmax_action(state)
        m = Categorical(to_tensor(softmax_action))
        action = m.sample().item()
        return action

    # evaluate value for a state-action pair
    def value(self, state):
        state_var = to_tensor([state], self.use_cuda)
        # action = index_to_one_hot(action, self.action_dim)
        # action_var = to_tensor([action], self.use_cuda)
        value_var = self.critic(state_var)
        if self.use_cuda:
            value = value_var.data.cpu().numpy()[0]
        else:
            value = value_var.data.numpy()[0]
        return value

    # discount roll out rewards
    def _discount_reward(self, rewards, values):
        advantage, deltas = np.zeros_like(rewards), np.zeros_like(rewards)

        prev_value = 0
        prev_advantage = 0
        for t in reversed(range(0, len(rewards))):
            deltas[t] = prev_value * self.reward_gamma + rewards[t] - values[t]
            advantage[t] = deltas[t] + self.reward_gamma * self.tau * prev_advantage

            prev_value = values[t]
            prev_advantage = advantage[t]
        returns = values + advantage
        return returns


    def interact(self):
        if (self.max_steps is not None) and (self.n_steps % self.max_steps == 0):
            self.env_state = self.env.reset()
        states = []
        actions = []
        rewards = []
        values = []
        steps = 0
        # take n steps
        # Todo modify to multi
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state[0])
            action = self.dist_explortation_action(self.env_state[0])
            one_hot_action = index_to_one_hot(action, dim=self.action_dim)
            next_state, reward, done, _ = self.env.step([one_hot_action])
            actions.append(action)
            if all(done) and self.done_penalty is not None:
                reward = self.done_penalty
            rewards.append(reward[0])
            # for advantage
            values.append(self.value(self.env_state[0])[0])

            final_state = next_state
            self.env_state = next_state
            if all(done):
                self.env_state = self.env.reset()
                break

        # discount reward
        if all(done):
            assert (False)
            final_value = 0.0
            self.n_episodes += 1
            self.episode_done = True
        else:
            self.episode_done = True
            # FIXME episodes
            self.n_episodes += 1
            # final_action = self.action(final_state)
            final_value = self.value(final_state[0])
            # 这里final_value是用来做TD(K)的最后一项
        # TODO 是否需要去掉discount
        dis_rewards = self._discount_reward(rewards, values)
        # print(self.n_steps)
        self.n_steps += 1
        self.memory.push(states, actions, dis_rewards)



    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor(batch.states, self.use_cuda).view(-1, self.state_dim)
        one_hot_actions = index_to_one_hot(batch.actions, self.action_dim)
        actions_var = to_tensor(one_hot_actions, self.use_cuda).view(-1, self.action_dim)
        returns_var = to_tensor(batch.rewards, self.use_cuda).view(-1, 1)
        # rewards_var = to_tensor(batch.disrewards, self.use_cuda).view(-1, 1)
        actions = to_tensor(batch.actions, self.use_cuda,dtype="long").view(-1, 1)

        # update actor network
        self.actor_optimizer.zero_grad()
        values = self.critic_target(states_var).detach()
        advantages = returns_var - values
        # # normalizing advantages seems not working correctly here
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        log_probs = self.actor(states_var)
        action_log_probs = th.gather(log_probs,dim=1, index=actions)
        dist_entropy = logpro2entropy(log_probs)
        old_action_log_probs = th.gather(self.actor_target(states_var).detach(),dim=1, index=actions)
        ratio = th.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * advantages
        surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        # PPO's pessimistic surrogate (L^CLIP)
        actor_loss = -th.mean(th.min(surr1, surr2)) - dist_entropy * self.entropy_reg
        actor_loss.backward()
        self.current_loss_actor = actor_loss.data
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic network
        self.critic_optimizer.zero_grad()
        target_values = returns_var
        values = self.critic(states_var)
        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(values, target_values)
        else:
            critic_loss = nn.MSELoss()(values, target_values)
        critic_loss.backward()
        self.current_loss_critic = critic_loss.data
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # update actor target network and critic target network
        if self.n_steps % self.target_update_steps == 0 and self.n_steps > 0:
            soft_update(self.actor_target, self.actor, self.target_tau)
            soft_update(self.critic_target, self.critic, self.target_tau)

    # evaluation the learned agent
    def evaluation(self, env, eval_steps = 50, eval_episodes=10):
        rewards = []
        infos = []
        for i in range(eval_episodes):
            steps = 0
            rewards_i = []
            infos_i = []
            state = env.reset()
            action = self.dist_action(state[0])
            one_hot_action = index_to_one_hot(action, self.action_dim)
            state, reward, done, info = env.step([one_hot_action])
            done = done[0] if isinstance(done, list) else done
            rewards_i.append(reward[0])
            infos_i.append(info)
            while not done and steps < eval_steps:
                action = self.dist_action(state[0])
                one_hot_action = index_to_one_hot(action, self.action_dim)
                state, reward, done, info = env.step([one_hot_action])
                done = done[0] if isinstance(done, list) else done
                rewards_i.append(reward[0])
                infos_i.append(info)
                steps += 1
                if arguments.load_model:
                    env.render()
            rewards.append(rewards_i)
            infos.append(infos_i)
        return rewards, infos

    def plot_error_vis(self):
        if self.n_episodes == 0:
            return
        if not self.viz:
            import visdom
            self.viz = visdom.Visdom()
            self.actor_win = self.viz.line(X=np.array([self.n_episodes]),
                                     Y=np.array([self.current_loss_actor]))
            self.critic_win = self.viz.line(X=np.array([self.n_episodes]),
                                     Y=np.array([self.current_loss_critic]))

        self.viz.line(
                 X=np.array([self.n_episodes]),
                 Y=np.array([self.current_loss_actor]),
                 win=self.actor_win,
                 update='append')
        self.viz.line(
                 X=np.array([self.n_episodes]),
                 Y=np.array([self.current_loss_critic]),
                 win=self.critic_win,
                 update='append')