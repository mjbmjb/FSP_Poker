
import torch as th
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np
from copy import deepcopy

# from common.Agent import Agent
from nn.mlpac import Actor, MCritic
from nn.ppo import PPO
from common.utils import to_tensor, index_to_one_hot, logpro2entropy, padobs
from nn.misc import soft_update, gumbel_softmax
import Settings.arguments as arguments

import random
from torch.distributions import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from collections import  namedtuple
Experience = namedtuple("Experience",
                        ("states", "actions","log_probs", "rewards", "advantages", "next_states", "dones"))


class ReplayMemory(object):
    """
    Replay memory buffer
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, action, log_prob, reward, advantage, next_state=None, done=None):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, log_prob, reward, advantage, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, log_prob, rewards, advantages, next_states=None, dones=None):
        if isinstance(states, list):
            if next_states is not None and len(next_states) > 0:
                for s,a, lp, r, ad, n_s,d in zip(states, actions, log_prob, rewards, advantages, next_states, dones):
                    self._push_one(s, a, lp, r, ad, n_s, d)
            else:
                for s,a, lp, r, ad in zip(states, actions ,log_prob, rewards, advantages):
                    self._push_one(s, a, lp, r, ad)
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
                 memory_capacity=100000, max_steps=None, ppo_epco = 4,
                 roll_out_n_steps=1, target_tau=1.,
                 target_update_steps=5, clip_param=0.20,
                 reward_gamma=0.99, reward_scale=1., tau=0.95 ,done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.0004, critic_lr=0.0004,
                 optimizer_type="adam", entropy_reg=0.05,
                 max_grad_norm=0.5, batch_size=128, episodes_before_train=500,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True, obspad_dim=None, is_dist = True, device = arguments.device):

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

        self.obspad_dim = obspad_dim

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
        self.device = device

        for i in range(n_agent):
            self.actors[i].to(self.device)
            self.critics[i].to(self.device)
            self.actors_target[i].to(self.device)
            self.critics_target[i].to(self.device)

        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.roll_out_n_steps = roll_out_n_steps
        self.ppo_epco = ppo_epco
        self.is_dist = is_dist

        self.viz = None
        self.current_loss_actor = 0
        self.current_loss_critic = 0

    # predict softmax action based on state
    def _logsoftmax_action(self, agent, state):
        state_var = to_tensor(state[agent]).unsqueeze(0)
        log_policy = self.actors[agent](state_var)
        # print(log_policy)
        # softmax_action_var = th.exp(log_policy)

        return log_policy.detach()

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, agent, state):
        softmax_action = self._logsoftmax_action(agent, state)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                  np.exp(-1. * self.n_steps / self.epsilon_decay)
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = np.argmax(softmax_action)
        action_var = th.LongTensor([action], device=self.device)
        log_prob = softmax_action.log()[:,action_var]
        return action_var, log_prob

    # choose an action based on state for execution
    def action(self, agent, state):
        softmax_action = self._softmax_action(agent, state)
        action = np.argmax(softmax_action)
        action_var = th.LongTensor([action], device=self.device)
        return action_var

    # TODO add entrophy
    def dist_exploration_action(self, agent, state):
        logsoftmax_action = self._logsoftmax_action(agent, state)
        if (logsoftmax_action >= 0).sum().item() < 3:
            self._logsoftmax_action(agent, state)
        m = Categorical(logits = logsoftmax_action)

        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.n_steps / self.epsilon_decay)
        # epsilon = 2
        if np.random.rand() < epsilon:
            action = m.sample()
        else:
            action = th.LongTensor([np.random.choice(self.action_dim)],
                                    device = arguments.device)
        log_prob = m.logits[:,action]
        # if log_prob.item()  == float('inf') or  log_prob.item()  == float('-inf'):
        #     mjb = 1
        return action, log_prob


    def dist_action(self,agent, state):
        logsoftmax_action = self._logsoftmax_action(agent, state)
        m = Categorical(logits = logsoftmax_action)
        action = m.sample()
        return action

    # evaluate value for a state-action pair
    def value(self, agent, states, actions, is_target = False):
        if isinstance(states, list):
            state_var_list = [to_tensor(states[i], self.device) for i in range(self.n_agent)]
            state_var = th.cat(state_var_list, 0).unsqueeze(0)
            action_var = th.cat(actions, 0).unsqueeze(0)
        else:
            state_var = states
            action_var = actions

        mask = th.Tensor(action_var.shape).to(self.device).fill_(1)
        mask[:, agent * self.action_dim:(agent + 1) * self.action_dim].fill_(0)
        # action = index_to_one_hot(action, self.action_dim)
        # action_var = to_tensor([action], self.device)
        action_var = action_var * mask

        acting_critic = self.critics_target if is_target else self.critics
        value_var = acting_critic[agent](state_var, action_var).to(self.device)

        return value_var

    # discount roll out rewards
    def _discount_reward(self, rewards, values, final_value):
        advantage, deltas = np.zeros_like(rewards), np.zeros_like(rewards)
        rewards, values = np.array(rewards), np.array(values)
        for a in range(self.n_agent):
            prev_value = final_value[a]
            # TODO determine the value of pre_advantage
            prev_advantage = 0
            for t in reversed(range(0, len(rewards))):
                deltas[t,a] = prev_value * self.reward_gamma + rewards[t,a] - values[t,a]
                advantage[t,a] = deltas[t,a] + self.reward_gamma * self.tau * prev_advantage

                prev_value = values[t,a]
                prev_advantage = advantage[t,a]
        returns = values + advantage
        advantage = (advantage - advantage.mean()) / advantage.std()
        return returns , advantage

    def interact(self):
        if (self.max_steps is not None) and (self.n_steps % self.max_steps == 0):
            self.env_state = padobs(self.env.reset(), self.obspad_dim)
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        hidden = []
        steps = 0
        # take n steps
        # Todo modify to multi
        for i in range(self.roll_out_n_steps):

            states.append(self.env_state)
            if self.is_dist:
                action, log_prob = zip(*[self.dist_exploration_action(agent, self.env_state)\
                      for agent in range(self.n_agent)])
            else:
                action, log_prob = zip(*[self.exploration_action(agent, self.env_state) \
                                         for agent in range(self.n_agent)])

            one_hot_action = index_to_one_hot(action, dim=self.action_dim)
            next_state, reward, done, _ = self.env.step(one_hot_action)
            next_state = padobs(next_state, self.obspad_dim)
            actions.append(action)
            log_probs.append(log_prob)
            if all(done) and self.done_penalty is not None:
                reward = self.done_penalty
            rewards.append(reward)
            # for advantage
            value = [self.value(agent, self.env_state, one_hot_action, is_target=True).item() \
                     for agent in range(self.n_agent)]
            values.append(value)

            final_state = next_state
            self.env_state = next_state
            if all(done):
                self.env_state = padobs(self.env.reset(), self.obspad_dim)
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
            if self.is_dist:
                final_action, _ = zip(*[self.dist_exploration_action(agent, final_state)\
                      for agent in range(self.n_agent)])
            else:
                final_action, _ = zip(*[self.exploration_action(agent, final_state)\
                      for agent in range(self.n_agent)])
            one_hot_final_action = index_to_one_hot(final_action, dim=self.action_dim)
            final_value = [self.value(agent, final_state, one_hot_final_action ,is_target=True).item()\
                           for agent in range(self.n_agent)]
            # 这里final_value是用来做TD(K)的最后一项
        # TODO 是否需要去掉discount
        dis_rewards, advantages = self._discount_reward(rewards, values, final_value)
        # print(self.n_steps)
        self.n_steps += 1
        self.memory.push(states, actions, log_probs, dis_rewards, advantages)



    # train on a roll out batch
    def train(self):
        for _ in range(self.ppo_epco):
            if self.n_episodes <= self.episodes_before_train:
                pass

            batch = self.memory.sample(self.batch_size)
            # batch x agent x state
            states_var = to_tensor(batch.states, self.device)
            # batch x [agent_n * state_dim]
            whole_states_var = states_var.view(self.batch_size,-1)
            # batch x agent
            action_var = arguments.LongTensor(batch.actions)
            log_probs_var = to_tensor(batch.log_probs)

            # batch x agent x action_dim
            one_hot_actions = index_to_one_hot(action_var, self.action_dim)
            whole_ont_hot_actions = one_hot_actions.view(self.batch_size, -1)
            # batch x agent
            returns_var = to_tensor(batch.rewards, self.device)
            advantages_var = to_tensor(batch.advantages, self.device)
            # rewards_var = to_tensor(batch.disrewards, self.device).view(-1, 1)

            for agent in range(self.n_agent):
                # print(self.actors[agent].fc1.weight)

                # update critic network
                self.critic_optimizer[agent].zero_grad()
                target_values = returns_var[:,agent]
                values = self.value(agent, whole_states_var, whole_ont_hot_actions, is_target=False).squeeze()
                if self.critic_loss == "huber":
                    critic_loss = nn.functional.smooth_l1_loss(values, target_values)
                else:
                    critic_loss = nn.MSELoss()(values, target_values)
                critic_loss.backward()
                self.current_loss_critic = critic_loss.data
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm(self.critics[agent].parameters(), self.max_grad_norm)
                self.critic_optimizer[agent].step()


                # values = self.critics_target[agent](whole_states_var, whole_ont_hot_actions).squeeze().detach()
                # advantages = returns_var[:,agent] - values
                # # normalizing advantages seems not working correctly here
                # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

                # TODO advantage is not correct when using Qnet, so there use Qnet to replace advantage
                # update actor network
                self.actor_optimizer[agent].zero_grad()
                advantages = advantages_var[:,agent]
                log_probs = self.actors[agent](states_var[:,agent,:])
                action_log_probs = th.gather(log_probs,dim=1, index=action_var[:,agent].unsqueeze(1)).squeeze()
                dist_entropy = logpro2entropy(log_probs)
                # old_action_log_probs = th.gather(self.actors_target[agent](states_var[:,agent,:]).detach(),
                #                                  dim=1, index=action_var[:,agent].unsqueeze(1))

                # # using Catrgorical to produce log_prob seems not working
                # action_probs = th.exp(self.actors[agent](states_var[:, agent, :]))
                # m = Categorical(action_probs)
                # action_log_probs = m.log_prob(action_var[:, agent])
                # dist_entropy = m.entropy()

                old_action_log_probs = log_probs_var[:, agent]
                ratio = th.exp(action_log_probs - old_action_log_probs).squeeze()
                surr1 = ratio * advantages
                surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                # PPO's pessimistic surrogate (L^CLIP)
                actor_loss = -th.mean(th.min(surr1, surr2)) - dist_entropy * self.entropy_reg
                # actor_loss = -th.mean(th.min(surr1, surr2))
                if actor_loss.data == float("inf") or actor_loss.data == float("-inf"):
                    print(actor_loss.data)
                actor_loss.backward()
                self.current_loss_actor = actor_loss.data

                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm(self.actors[agent].parameters(), self.max_grad_norm)
                self.actor_optimizer[agent].step()

                if agent == 3:
                    mjb = 1
                # update actor target network and critic target network
                if self.n_steps % self.target_update_steps == 0:
                    # print('Update target network')
                    # soft_update(self.actors_target[agent], self.actors[agent], self.target_tau)
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
            state = padobs(env.reset(), self.obspad_dim)
            if self.is_dist:
                action = [self.dist_action(agent, state)\
                      for agent in range(self.n_agent)]
            else:
                action = [self.action(agent, state) \
                          for agent in range(self.n_agent)]
            one_hot_action = index_to_one_hot(action, self.action_dim)
            state, reward, done, info = env.step(one_hot_action)
            state = padobs(state, self.obspad_dim)
            done = done[0] if isinstance(done, list) else done
            rewards_i.append(reward)
            infos_i.append(info)
            while not done and steps < eval_steps:
                if self.is_dist:
                    action = [self.dist_action(agent, state)\
                          for agent in range(self.n_agent)]
                else:
                    action = [self.action(agent, state) \
                              for agent in range(self.n_agent)]
                one_hot_action = index_to_one_hot(action, self.action_dim)
                state, reward, done, info = env.step(one_hot_action)
                state = padobs(state, self.obspad_dim)
                done = done[0] if isinstance(done, list) else done
                rewards_i.append(reward)
                infos_i.append(info)
                if arguments.evalation:
                   env.render('human')
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

            self.critics_target[i].load_state_dict(self.critics[i].state_dict())
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())