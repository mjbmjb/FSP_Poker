import math
import torch
import random
import time
import numpy as np
import visdom
from datetime import datetime

from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from collections import deque, namedtuple

import Settings.arguments as arguments
from nn.lstmac import MCritic, Actor
from nn.SharedRMSprop import SharedRMSprop
from nn.make_env import make_env

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'policy'))

def state_to_tensor(states):
  return [torch.from_numpy(state).float().unsqueeze(0) for state in states]

class EpisodicReplayMemory():
  def __init__(self, capacity, max_episode_length):
    # Max number of transitions possible will be the memory capacity, could be much less
    self.num_episodes = capacity // max_episode_length
    self.memory = deque(maxlen=self.num_episodes)
    self.memory.append([])  # List for first episode
    self.position = 0

  def append(self, state, action, reward, policy):
    self.memory[self.position].append(Transition(state, action, reward, policy))  # Save s_i, a_i, r_i+1, µ(·|s_i)
    # Terminal states are saved with actions as None, so switch to next episode
    if action is None:
      self.memory.append([])
      self.position = min(self.position + 1, self.num_episodes - 1)

  # Samples random trajectory
  def sample(self, maxlen=0):
    while True:
      e = random.randrange(len(self.memory))
      mem = self.memory[e]
      T = len(mem)
      if T > 0:
        # Take a random subset of trajectory if maxlen specified, otherwise return full trajectory
        if maxlen > 0 and T > maxlen + 1:
          t = random.randrange(T - maxlen - 1)  # Include next state after final "maxlen" state
          return mem[t:t + maxlen + 1]
        else:
          return mem

  # Samples batch of trajectories, truncating them to the same length
  def sample_batch(self, batch_size, maxlen=0):
    batch = [self.sample(maxlen=maxlen) for _ in range(batch_size)]
    minimum_size = min(len(trajectory) for trajectory in batch)
    batch = [trajectory[:minimum_size] for trajectory in batch]  # Truncate trajectories
    return list(map(list, zip(*batch)))  # Transpose so that timesteps are packed together

  def __len__(self):
    return sum(len(episode) for episode in self.memory)

class MAcerOptim():
    def __init__(self, observation_dim, action_dim, n_agent):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.n_agent = n_agent

        self.T_max = 10000000 # Number of training steps
        self.t_max = 500 # Max number of forward steps for A3C before update

        self.memory_capacity = 1000000
        self.num_processes = arguments.num_process
        self.max_episode_length = 100 # Maximum episode length (used to determine length of memory)

        self.evaluate = False
        self.evaluation_episodes = 500
        self.evaluation_interval = 2500 # Number of training steps between evaluations (roughly)

        self.on_policy = False
        self.hidden_size = 64
        self.discount = 0.99 # γ
        self.replay_ratio = 4
        self.replay_start = 20000
        self.trace_decay = 1 # Eligibility trace decay factor: λ
        self.trace_max = 10 # Importance weight truncation (max) value: c
        self.trust_region = True
        self.trust_region_decay = 0.99 # Discount factor: γ
        self.trust_region_threshold = 1 # Trust region threshold value: δ
        self.lr = 0.0007
        self.lr_decay = True # Linearly decay learning rate to 0
        self.rmsprop_decay = 0.99 # RMSprop decay factor: α
        self.batch_size = 256
        self.entropy_weight = 0.0001 # Entropy regularisation weight: β
        self.max_gradient_norm = 40  # Gradient L2 normalisation

        self.shared_critic = []
        self.shared_actor = []
        self.shared_a_critic = []
        self.shared_a_actor = []
        self.optimisers_actor = []
        self.optimisers_critic = []
        for i in range(n_agent):
            sc = MCritic(observation_dim, action_dim, self.hidden_size)
            sc.share_memory()
            sac = MCritic(observation_dim, action_dim, self.hidden_size)
            sac.load_state_dict(sc.state_dict())
            sac.share_memory()
            for param in sac.parameters():
                param.requires_grad = False

            sa = Actor(observation_dim, action_dim, self.hidden_size)
            sa.share_memory()
            saa = Actor(observation_dim, action_dim, self.hidden_size)
            saa.load_state_dict(sa.state_dict())
            saa.share_memory()
            for param in saa.parameters():
                param.requires_grad = False

            optc = SharedRMSprop(sc.parameters(), lr=self.lr, alpha=self.rmsprop_decay)
            optc.share_memory()
            opta = SharedRMSprop(sa.parameters(), lr=self.lr, alpha=self.rmsprop_decay)
            opta.share_memory()

            self.shared_critic.append(sc)
            self.shared_actor.append(sa)
            self.shared_a_critic.append(sac)
            self.shared_a_actor.append(saa)
            self.optimisers_critic.append(optc)
            self.optimisers_actor.append(opta)



    # Knuth's algorithm for generating Poisson samples
    def _poisson(self, lmbd):
        L, k, p = math.exp(-lmbd), 0, 1
        while p > L:
            k += 1
            p *= random.uniform(0, 1)
        return max(k - 1, 0)

    # Transfers gradients from thread-specific model to shared model
    def _transfer_grads_to_shared_model(self, model, shared_model):
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    # Adjusts learning rate
    def _adjust_learning_rate(self, optimiser, lr):
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr

    # Updates networks
    def _update_networks(self, T, model, loss):
        # Zero shared and local grads
        self.optimiser.zero_grad()
        """
        Calculate gradients for gradient descent on loss functions
        Note that math comments follow the paper, which is formulated for gradient ascent
        """
        loss.backward()
        # Gradient L2 normalisation
        nn.utils.clip_grad_norm(model.parameters(), self.max_gradient_norm)

        # Transfer gradients to shared model and update
        self._transfer_grads_to_shared_model(model, self.shared_model)
        self.optimiser.step()
        if self.lr_decay:
            # Linearly decay learning rate
            self._adjust_learning_rate(self.optimiser, max(self.lr * (self.T_max - T.value()) / self.T_max, 1e-32))

        # Update shared_average_model
        for shared_param, shared_average_param in zip(self.shared_model.parameters(), self.shared_average_model.parameters()):
            shared_average_param = self.trust_region_decay * shared_average_param + (
                        1 - self.trust_region_decay) * shared_param

    # Computes a trust region loss based on an existing loss and two distributions
    def _trust_region_loss(self, model, distribution, ref_distribution, loss, threshold):
        # Compute gradients from original loss
        model.zero_grad()
        loss.backward(retain_graph=True)
        # Gradients should be treated as constants (not using detach as volatility can creep in when double backprop is not implemented)
        g = [Variable(param.grad.data.clone()) for param in model.parameters() if param.grad is not None]
        model.zero_grad()

        # KL divergence k ← ∇θ0∙DKL[π(∙|s_i; θ_a) || π(∙|s_i; θ)]
        kl = F.kl_div(distribution.log(), ref_distribution, size_average=False)
        # Compute gradients from (negative) KL loss (increases KL divergence)
        (-kl).backward(retain_graph=True)
        k = [Variable(param.grad.data.clone()) for param in model.parameters() if param.grad is not None]
        model.zero_grad()

        # Compute dot products of gradients
        k_dot_g = sum(torch.sum(k_p * g_p) for k_p, g_p in zip(k, g))
        k_dot_k = sum(torch.sum(k_p ** 2) for k_p in k)
        # Compute trust region update
        if k_dot_k.data.item() > 0:
            trust_factor = ((k_dot_g - threshold) / k_dot_k).clamp(min=0)
        else:
            trust_factor = Variable(torch.zeros(1))
        # z* = g - max(0, (k^T∙g - δ) / ||k||^2_2)∙k
        z_star = [g_p - trust_factor.expand_as(k_p) * k_p for g_p, k_p in zip(g, k)]
        trust_loss = 0
        for param, z_star_p in zip(model.parameters(), z_star):
            trust_loss += (param * z_star_p).sum()
        return trust_loss

    # Trains model
    def _train(self, T, model, policies, Qs, Vs, actions, rewards, Qret,
               average_policies, old_policies=None):
        off_policy = old_policies is not None
        # policy is batch
        action_size = policies[0].size(1)
        policy_loss, value_loss = 0, 0

        # Calculate n-step returns in forward view, stepping backwards from the last state
        t = len(rewards)
        for i in reversed(range(t)):
            # Importance sampling weights ρ ← π(∙|s_i) / µ(∙|s_i); 1 for on-policy
            if off_policy:
                rho = policies[i].detach() / old_policies[i]
            else:
                rho = Variable(torch.ones(1, action_size))

            # Qret ← r_i + γQret
            Qret = rewards[i] + self.discount * Qret
            # Advantage A ← Qret - V(s_i; θ)
            A = Qret - Vs[i]

            # Log policy log(π(a_i|s_i; θ))
            log_prob = policies[i].gather(1, actions[i]).log()
            # g ← min(c, ρ_a_i)∙∇θ∙log(π(a_i|s_i; θ))∙A    |  log_prob.requires_grad == True
            single_step_policy_loss = -(
                        rho.gather(1, actions[i]).clamp(max=self.trace_max) * log_prob * A.detach()).mean(
                0)  # Average over batch
            # Off-policy bias correction
            if off_policy:
                # g ← g + Σ_a [1 - c/ρ_a]_+∙π(a|s_i; θ)∙∇θ∙log(π(a|s_i; θ))∙(Q(s_i, a; θ) - V(s_i; θ)
                bias_weight = (1 - self.trace_max / rho).clamp(min=0) * policies[i]
                single_step_policy_loss -= (
                            bias_weight * policies[i].log() * (Qs[i].detach() - Vs[i].expand_as(Qs[i]).detach())).sum(
                    1).mean(0)
            if self.trust_region:
                # Policy update dθ ← dθ + ∂θ/∂θ∙z*
                policy_loss += self._trust_region_loss(model, policies[i], average_policies[i], single_step_policy_loss,
                                                  self.trust_region_threshold)
            else:
                # Policy update dθ ← dθ + ∂θ/∂θ∙g
                policy_loss += single_step_policy_loss

            # Entropy regularisation dθ ← dθ + β∙∇θH(π(s_i; θ))
            policy_loss -= self.entropy_weight * -(policies[i].log() * policies[i]).sum(1).mean(
                0)  # Sum over probabilities, average over batch

            # Value update dθ ← dθ - ∇θ∙1/2∙(Qret - Q(s_i, a_i; θ))^2
            Q = Qs[i].gather(1, actions[i])
            value_loss += ((Qret - Q) ** 2 / 2).mean(0)  # Least squares loss

            # Truncated importance weight ρ¯_a_i = min(1, ρ_a_i)
            truncated_rho = rho.gather(1, actions[i]).clamp(max=1)
            # Qret ← ρ¯_a_i∙(Qret - Q(s_i, a_i; θ)) + V(s_i; θ)
            Qret = truncated_rho * (Qret - Q.detach()) + Vs[i].detach()

        # Update networks
        self._update_networks(T, model, policy_loss + value_loss)

# Acts and trains model
def train(rank, model_opti, T):
    # every train process has distribute model
    n_agent = model_opti.n_agent
    critics = [MCritic(model_opti.observation_dim, model_opti.action_dim, model_opti.hidden_size) for i in range(n_agent)]
    actors = [Actor(model_opti.observation_dim, model_opti.action_dim, model_opti.hidden_size) for i in range(n_agent)]
    for critic, actor in zip(critics,actors):
        critic.train()
        actor.train()

    env = make_env('simple')

    if not model_opti.on_policy:
        # Normalise memory capacity by number of tr aining processes
        memory = EpisodicReplayMemory(model_opti.memory_capacity // model_opti.num_processes, model_opti.max_episode_length)

    t = 1  # Thread step counter
    done = True  # Start new episode

    while T.value() <= model_opti.T_max:
        # On-policy episode loop
        while True:
            # Sync with shared model at least every t_max steps
            # [model[i].load_state_dict(model_opti[i].shared_model.state_dict()) for i in range(model_opti.n_agent)]

            # Get starting timestep
            t_start = t

            # Reset or pass on hidden state
            if done:
                hx, avg_hx = torch.zeros(1, model_opti.hidden_size), torch.zeros(1, model_opti.hidden_size)
                cx, avg_cx = torch.zeros(1, model_opti.hidden_size), torch.zeros(1, model_opti.hidden_size)
                # Reset environment and done flag
                state = state_to_tensor(env.reset())
                done, episode_length = False, 0
            else:
                # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
                hx = hx.detach()
                cx = cx.detach()

            hxs = [hx.copy() for _ in range(n_agent)]
            cxs = [cx.copy() for _ in range(n_agent)]
            avg_hxs = [avg_hx.copy() for _ in range(n_agent)]
            avg_cxs = [avg_cx.copy() for _ in range(n_agent)]


            # Lists of outputs for training
            policies, Qs, Vs, actions, rewards, average_policies = [], [], [], [], [], []

            while not done and t - t_start < model_opti.t_max:
                # Calculate policy and values
                action = []
                one_hot_action = []
                policy = []
                for i in range(n_agent):
                    Q, (hxs[i], cxs[i]) = critics[i](state, (hxs[i], cxs[i]))
                    p, (hxs[i], cxs[i]) = actors[i](state[i], (hxs[i], cxs[i]))
                    average_p, (avg_hx, avg_cx) = model_opti.shared_average_model[i](state[i], (avg_hxs[i], avg_cxs[i]))

                    # Sample action
                    m = Categorical(p)
                    a = m.sample()
                    one_hot_a = torch.eye(p.size(1))[a].squeeze(0)
                    a = a.data.item()
                    # Graph broken as loss for stochastic action calculated manually
                    action.append(a)
                    one_hot_action.append(one_hot_a)
                    policy.append(p)

                # Step
                next_state, reward, done, _ = env.step(one_hot_action)
                next_state = state_to_tensor(next_state)
                # reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
                # TODO modify reward to mulplayer

                done = all(done) or episode_length >= model_opti.max_episode_length  # Stop episodes at a max length
                episode_length += 1  # Increase episode counter

                if not model_opti.on_policy:
                    # Save (beginning part of) transition for offline training
                    memory.append(state, action, reward, [p.data for p in policy])  # Save just tensors
                # Save outputs for online training
                [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies),
                                                   (policy, Q, V, action),
                                                    torch.Tensor(reward), average_policy))]

                # Increment counters
                t += 1
                T.increment()

                # Update state
                state = next_state

            # Break graph for last values calculated (used for targets, not directly as model outputs)
            if done:
                # Qret = 0 for terminal s
                Qret = Variable(torch.zeros(1, 1))

                if not model_opti.on_policy:
                    # Save terminal state for offline training
                    memory.append(state, None, None, None)
            else:
                # Qret = V(s_i; θ) for non-terminal s
                _, _, Qret, _ = model(Variable(state), (hx, cx))
                Qret = Qret.detach()

            # Train the network on-policy
            model_opti._train(T, model, policies, Qs, Vs, actions,
                   rewards, Qret, average_policies)

            # Finish on-policy episode
            if done:
                break

        # Train the network off-policy when enough experience has been collected
        if not model_opti.on_policy and len(memory) >= model_opti.replay_start:
            # Sample a number of off-policy episodes based on the replay ratio
            for _ in range(model_opti._poisson(model_opti.replay_ratio)):
                # Act and train off-policy for a batch of (truncated) episode
                trajectories = memory.sample_batch(model_opti.batch_size, maxlen=model_opti.t_max)

                # Reset hidden state
                hx, avg_hx = Variable(torch.zeros(model_opti.batch_size, model_opti.hidden_size)), Variable(
                    torch.zeros(model_opti.batch_size, model_opti.hidden_size))
                cx, avg_cx = Variable(torch.zeros(model_opti.batch_size, model_opti.hidden_size)), Variable(
                    torch.zeros(model_opti.batch_size, model_opti.hidden_size))

                # Lists of outputs for training
                policies, Qs, Vs, actions, rewards, old_policies, average_policies = [], [], [], [], [], [], []

                # Loop over trajectories (bar last timestep)
                for i in range(len(trajectories) - 1):
                    # Unpack first half of transition
                    state = torch.cat([trajectory.state for trajectory in trajectories[i]], 0)
                    action = Variable(
                        torch.LongTensor([trajectory.action for trajectory in trajectories[i]])).unsqueeze(1)
                    reward = Variable(
                        torch.Tensor([trajectory.reward for trajectory in trajectories[i]])).unsqueeze(1)
                    old_policy = Variable(torch.cat([trajectory.policy for trajectory in trajectories[i]], 0))

                    # Calculate policy and values
                    policy, Q, V, (hx, cx) = model(Variable(state), (hx, cx))
                    average_policy, _, _, (avg_hx, avg_cx) = model_opti.shared_average_model(Variable(state), (avg_hx, avg_cx))

                    # Save outputs for offline training
                    [arr.append(el) for arr, el in
                     zip((policies, Qs, Vs, actions, rewards, average_policies, old_policies),
                         (policy, Q, V, action, reward, average_policy, old_policy))]

                    # Unpack second half of transition
                    next_state = torch.cat([trajectory.state for trajectory in trajectories[i + 1]], 0)
                    done = Variable(
                        torch.Tensor([trajectory.action is None for trajectory in trajectories[i + 1]]).unsqueeze(
                            1))

                # Do forward pass for all transitions
                _, _, Qret, _ = model(Variable(next_state), (hx, cx))
                # Qret = 0 for terminal s, V(s_i; θ) otherwise
                Qret = ((1 - done) * Qret).detach()

                # Train the network off-policy
                model_opti._train(T, model, policies, Qs, Vs,
                       actions, rewards, Qret, average_policies, old_policies=old_policies)
        done = True

    env.close()

def test(rank, model_opti, T):
    torch.manual_seed(1234 + rank)
    env = make_env('simple')

    model = ActorCritic(model_opti.observation_dim, model_opti.action_dim, model_opti.hidden_size)
    model.eval()

    viz = visdom.Visdom()
    win = viz.line(X=np.array([0]),
                   Y=np.array([0]))

    can_test = True  # Test flag
    t_start = 1  # Test step counter to check against global counter
    rewards, steps = [], []  # Rewards and steps for plotting
    l = str(len(str(model_opti.T_max)))  # Max num. of digits for logging steps
    done = True  # Start new episode

    while T.value() <= model_opti.T_max:
        if can_test:
            t_start = T.value()  # Reset counter

            # Evaluate over several episodes and average results
            avg_rewards, avg_episode_lengths = [], []
            for _ in range(model_opti.evaluation_episodes):
                while True:
                    # Reset or pass on hidden state
                    if done:
                        # Sync with shared model every episode
                        model.load_state_dict(model_opti.shared_model.state_dict())
                        hx = Variable(torch.zeros(1, model_opti.hidden_size), volatile=True)
                        cx = Variable(torch.zeros(1, model_opti.hidden_size), volatile=True)
                        # Reset environment and done flag
                        state = state_to_tensor(env.reset()[0])
                        done, episode_length = False, 0
                        reward_sum = 0

                    # Optionally render validation states
                    if arguments.load_model:
                        env.render()

                    # Calculate policy
                    with torch.no_grad():
                        policy, _, _, (hx, cx) = model(state,
                                                   (hx.detach(), cx.detach()))  # Break graph for memory efficiency

                    # Choose action greedily
                    action = policy.max(1)[1].item()
                    one_hot_action = torch.eye(policy.size(1))[action].squeeze(0)
                    # Graph broken as loss for stocha

                    # Step
                    state, reward, done, _ = env.step([one_hot_action] * 3)
                    state = state_to_tensor(state[0])
                    reward_sum += reward[0]
                    done = all(done) or episode_length >= model_opti.max_episode_length  # Stop episodes at a max length
                    episode_length += 1  # Increase episode counter

                    # Log and reset statistics at the end of every episode
                    if done:
                        avg_rewards.append(reward_sum)
                        avg_episode_lengths.append(episode_length)
                        break

            print(('[{}] Step: {:<' + l + '} Avg. Reward: {:<8} Avg. Episode Length: {:<8}').format(
                datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
                t_start,
                sum(avg_rewards) / model_opti.evaluation_episodes,
                sum(avg_episode_lengths) / model_opti.evaluation_episodes))

            if model_opti.evaluate:
                return

            # rewards.append(avg_rewards)  # Keep all evaluations
            # steps.append(t_start)
            # plot_line(steps, rewards)  # Plot rewards
            plot_vis(t_start, avg_rewards, viz, win)
            torch.save(model.state_dict(), 'model.pth')  # Save model params
            can_test = False  # Finish testing
        else:
            if T.value() - t_start >= model_opti.evaluation_interval:
                can_test = True

        time.sleep(0.001)  # Check if available to test every millisecond

    env.close()

def plot_vis(step, rewards, viz, win):
    viz.line(
        X=np.array([step]),
        Y=np.array([rewards]),
        win=win,
        update='append')


