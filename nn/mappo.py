
import torch as th
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np
from copy import deepcopy

# from common.Agent import Agent
from nn.mlpac import Actor, MCritic
from nn.ppo import PPO
from common.utils import to_tensor, index_to_one_hot, logpro2entropy
from nn.misc import soft_update
import Settings.arguments as arguments

import random
from torch.distributions import Categorical
from collections import  namedtuple
Experience = namedtuple("Experience",
                        ("states", "actions", "rewards", "advantages", "next_states", "dones"))


class ReplayMemory(object):
    """
    Replay memory buffer
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, action, reward, advantage, next_state=None, done=None):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, reward, advantage, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, rewards, advantages, next_states=None, dones=None):
        if isinstance(states, list):
            if next_states is not None and len(next_states) > 0:
                for s,a,r, ad, n_s,d in zip(states, actions, rewards, advantages, next_states, dones):
                    self._push_one(s, a, r, ad, n_s, d)
            else:
                for s,a,r, ad in zip(states, actions, rewards, advantages):
                    self._push_one(s, a, r, ad)
        else:
            self._push_one(states, actions, rewards, advantages, next_states, dones)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)


class MAPPO(PPO):
    """
    An agent learned with PPO using Advantage Actor-Critic framework
    - Actor takes state as input
    - Critic takes both state and action as input
    - agent interact with environment to collect experience
    - agent training with experience to update policy
    - adam seems better than rmsprop for ppo
    """
    def __init__(self,n_agent, env, state_dim, action_dim,
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

        super(MAPPO, self).__init__(env, state_dim, action_dim,
                 memory_capacity, max_steps,
                 roll_out_n_steps, target_tau,
                 target_update_steps, clip_param,
                 reward_gamma, reward_scale, tau ,done_penalty,
                 actor_hidden_size, critic_hidden_size,
                 actor_output_act, critic_loss,
                 actor_lr, critic_lr,
                 optimizer_type, entropy_reg,
                 max_grad_norm, batch_size, episodes_before_train,
                 epsilon_start, epsilon_end, epsilon_decay,
                 use_cuda)

        self.n_agent = n_agent

        self.actors = [Actor(state_dim, actor_hidden_size,
                                  action_dim, actor_output_act) for _ in range(n_agent)]
        self.critics = [MCritic(n_agent, state_dim, action_dim,
                              critic_hidden_size, 1) for _ in range(n_agent)]
        # to ensure target network and learning network has the same weights
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        self.memory = ReplayMemory(memory_capacity)

        if self.optimizer_type == "adam":
            self.actor_optimizer = [Adam(x.parameters(), lr=self.actor_lr) \
                                    for x in self.actors]
            self.critic_optimizer = [Adam(x.parameters(), lr=self.critic_lr) \
                                     for x in self.critics]
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = [RMSprop(x.parameters(), lr=self.actor_lr) \
                                    for x in self.actors]
            self.critic_optimizer = [RMSprop(x.parameters(), lr=self.critic_lr) \
                                     for x in self.critics]

        self.use_cuda = use_cuda
        if self.use_cuda:
            for i in range(n_agent):
                self.actors[i].cuda()
                self.critics[i].cuda()
                self.actors_target[i].cuda()
                self.critics_target[i].cuda()

        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.roll_out_n_steps = roll_out_n_steps

        self.viz = None
        self.current_loss_actor = 0
        self.current_loss_critic = 0

    # predict softmax action based on state
    def _softmax_action(self, agent, state):
        state_var = to_tensor(state[agent], self.use_cuda)
        softmax_action_var = th.exp(self.actors[agent](state_var))
        if self.use_cuda:
            softmax_action = softmax_action_var.data.cpu()
        else:
            softmax_action = softmax_action_var.data

        return softmax_action

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, agent, state):
        softmax_action = self._softmax_action(agent, state)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                  np.exp(-1. * self.n_steps / self.epsilon_decay)
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = np.argmax(softmax_action)
        return action

    # choose an action based on state for execution
    def action(self, agent, state):
        softmax_action = self._softmax_action(agent, state)
        action = np.argmax(softmax_action)
        return action

    # TODO add entrophy
    def dist_exploration_action(self, agent, state):
        softmax_action = self._softmax_action(agent, state)
        assert((softmax_action >= 0).sum().item() == 5)
        m = Categorical(to_tensor(softmax_action))
        action = m.sample()
        return action


    def dist_action(self,agent, state):
        softmax_action = self._softmax_action(agent, state)
        m = Categorical(to_tensor(softmax_action))
        action = m.sample()
        return action

    # evaluate value for a state-action pair
    def value(self, agent, states, actions):
        state_var_list = [to_tensor(states[i], self.use_cuda) for i in range(self.n_agent)]
        state_var = th.cat(state_var_list, 0).unsqueeze(0)
        action_var = th.cat(actions, 0).unsqueeze(0)
        # action = index_to_one_hot(action, self.action_dim)
        # action_var = to_tensor([action], self.use_cuda)
        value_var = self.critics[agent](state_var, action_var)
        if self.use_cuda:
            value = value_var.data.cpu()
        else:
            value = value_var.data
        return value

    # discount roll out rewards
    def _discount_reward(self, next_value, rewards, values):
        advantage, deltas = np.zeros_like(rewards), np.zeros_like(rewards)
        rewards, values = np.array(rewards), np.array(values)
        for a in range(self.n_agent):
            prev_value = next_value
            # TODO determine the value of pre_advantage
            prev_advantage = 0
            for t in reversed(range(0, len(rewards))):
                deltas[t,a] = prev_value * self.reward_gamma + rewards[t,a] - values[t,a]
                advantage[t,a] = deltas[t,a] + self.reward_gamma * self.tau * prev_advantage

                prev_value = values[t,a]
                prev_advantage = advantage[t,a]
        returns = values + advantage
        # advantage = (advantage - advantage.mean()) / advantage.std()
        return returns , advantage


    def interact(self):
        if (self.max_steps is not None) and (self.n_steps % self.max_steps == 0):
            self.env_state = self.env.reset()
        states = []
        actions = []
        rewards = []
        values = []
        hidden = []
        steps = 0
        # take n steps
        # Todo modify to multi
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action = [self.dist_exploration_action(agent, self.env_state)\
                      for agent in range(self.n_agent)]
            one_hot_action = index_to_one_hot(action, dim=self.action_dim)
            next_state, reward, done, _ = self.env.step(one_hot_action)
            actions.append(action)
            if all(done) and self.done_penalty is not None:
                reward = self.done_penalty
            rewards.append(reward)
            # for advantage
            value = [self.value(agent, self.env_state, one_hot_action) \
                     for agent in range(self.n_agent)]
            values.append(value)

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
            final_action = [self.action(agent, final_state) for agent in range(self.n_agent)]
            final_value = [self.value(agent,final_state,final_action) for agent in range(self.n_agent)]
            # 这里final_value是用来做TD(K)的最后一项
        # TODO 是否需要去掉discount
        dis_rewards, advantages = self._discount_reward(final_value, rewards, values)
        # print(self.n_steps)
        self.n_steps += 1
        self.memory.push(states, actions, dis_rewards, advantages)



    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        # batch x agent x state
        states_var = to_tensor(batch.states, self.use_cuda)
        # batch x [agent_n * state_dim]
        whole_states_var = states_var.view(self.batch_size,-1)
        # batch x agent
        action_var = arguments.LongTensor(batch.actions)
        # batch x agent x action_dim
        one_hot_actions = index_to_one_hot(action_var, self.action_dim)
        whole_ont_hot_actions = one_hot_actions.view(self.batch_size, -1)
        # batch x agent
        returns_var = to_tensor(batch.rewards, self.use_cuda)
        advantages_var = to_tensor(batch.advantages, self.use_cuda)
        # rewards_var = to_tensor(batch.disrewards, self.use_cuda).view(-1, 1)

        for agent in range(self.n_agent):

            # update critic network
            self.critic_optimizer[agent].zero_grad()
            target_values = returns_var[:,agent]
            values = self.critics[agent](whole_states_var,whole_ont_hot_actions).squeeze()
            if self.critic_loss == "huber":
                critic_loss = nn.functional.smooth_l1_loss(values, target_values)
            else:
                critic_loss = nn.MSELoss()(values, target_values)
            critic_loss.backward()
            self.current_loss_critic = critic_loss.data
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.critics[agent].parameters(), self.max_grad_norm)
            self.critic_optimizer[agent].step()

            # update actor network
            self.actor_optimizer[agent].zero_grad()
            # values = self.critics_target[agent](whole_states_var, whole_ont_hot_actions).squeeze().detach()
            # advantages = returns_var[:,agent] - values
            # # normalizing advantages seems not working correctly here
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            advantages = advantages_var[:,agent]
            log_probs = self.actors[agent](states_var[:,agent,:])
            action_log_probs = th.gather(log_probs,dim=1, index=action_var[:,agent].unsqueeze(1))
            # dist_entropy = logpro2entropy(log_probs)
            old_action_log_probs = th.gather(self.actors_target[agent](states_var[:,agent,:]).detach(),
                                             dim=1, index=action_var[:,agent].unsqueeze(1))
            ratio = th.exp(action_log_probs - old_action_log_probs).squeeze()
            surr1 = ratio * advantages
            surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            # PPO's pessimistic surrogate (L^CLIP)
            # actor_loss = -th.mean(th.min(surr1, surr2)) - dist_entropy * self.entropy_reg
            actor_loss = -th.mean(th.min(surr1, surr2))
            actor_loss.backward()
            self.current_loss_actor = actor_loss.data
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.actors[agent].parameters(), self.max_grad_norm)
            self.actor_optimizer[agent].step()

            # update actor target network and critic target network
            if self.n_steps % self.target_update_steps == 0 and self.n_steps > 0:
                soft_update(self.actors_target[agent], self.actors[agent], self.target_tau)
                soft_update(self.critics_target[agent], self.critics[agent], self.target_tau)

    # evaluation the learned agent
    # @return rewards: list(eval_episodes,eval_steps,agent)
    def evaluation(self, env, eval_steps = 50, eval_episodes=10):
        rewards = []
        infos = []
        for i in range(eval_episodes):
            steps = 0
            rewards_i = []
            infos_i = []
            state = env.reset()
            action = [self.dist_action(agent, state)\
                      for agent in range(self.n_agent)]
            one_hot_action = index_to_one_hot(action, self.action_dim)
            state, reward, done, info = env.step(one_hot_action)
            done = done[0] if isinstance(done, list) else done
            rewards_i.append(reward)
            infos_i.append(info)
            while not done and steps < eval_steps:
                action = [self.dist_action(agent, state)\
                          for agent in range(self.n_agent)]
                one_hot_action = index_to_one_hot(action, self.action_dim)
                state, reward, done, info = env.step(one_hot_action)
                done = done[0] if isinstance(done, list) else done
                rewards_i.append(reward)
                infos_i.append(info)
                if arguments.render:
                   env.render()
                steps += 1
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


    def save(self, path):
        for i in range(self.n_agent):
            save_path = path + '_' + str(i) + '.'
            th.save(self.actors[i].state_dict(), save_path+'actor')
            th.save(self.critics[i].state_dict(), save_path + 'critic')

    def load(self, path):
        for i in range(self.n_agent):
            save_path = path + '_' + str(i) + '.'
            self.actors[i].load_state_dict(th.load(save_path+'actor'))
            self.critics[i].load_state_dict(th.load(save_path+'critic'))