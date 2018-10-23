    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 05:15:26 2017

@author: mjb
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append('/home/carc/mjb/deepStack/')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import cProfilev

from nn.sim_env import SimEnv
from nn.make_env import make_env

from nn.dqn import DQNOptim
from nn.reinforce import ReinforceOptim
from nn.maddpg import MADDPG
from nn.net_sl import SLOptim
from nn.dqn import Transition as RL_TRAN
from nn.net_sl import Transition as SL_TRAN
from nn.maddpg import Experience as MADDPG_TRAN
from nn.ppo import PPO
from nn.mappo import MAPPO
from nn.lstmmappo import MAPPO as LSTMMAPPO
from Tree.game_state import GameState

import random
import math
import numpy as np
import torch
from torch import multiprocessing as mp

import Settings.arguments as arguments
import Settings.constants as constants
import Settings.game_settings as game_settings
from torch.autograd import Variable
from itertools import count
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

from collections import namedtuple
import datetime

num_episodes = 10
env = SimEnv(True)

Agent = namedtuple('Agent', ['rl', 'sl', 'ID'])
Reward = [0] * game_settings.player_count

rl_model = {'dqn': DQNOptim, 'reinforce': ReinforceOptim, 'maddpg': MADDPG}
sl_model = {'mlr': SLOptim}

Agents = []
num_agent = game_settings.player_count if arguments.multi_agent else 1
for i in range(num_agent):
    Agents.append(Agent(rl=rl_model[arguments.rl_model](), sl=sl_model[arguments.sl_model](), ID=i))


def load_model(iter_time):
    iter_str = str(iter_time)
    #    # load rl model (only the net)
    for i in range(game_settings.player_count):
        Agents[i].rl.model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '_' + str(i) + '_' + '.rl'))
        Agents[i].rl.target_net.load_state_dict(Agents[i].rl.model.state_dict())

        Agents[i].sl.model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '_' + str(i) + '_' + '.sl'))


def save_model(episod):
    path = arguments.WORK_PATH + "/Data/Model/Iter:" + str(episod)

    for agent in Agents:
        # save sl strategy
        #    torch.save(table_sl.strategy, sl_name)
        sl_model = agent.sl.model.state_dict()
        rl_model = agent.rl.model.state_dict()
        if arguments.cpu_store:
            sl_model = agent.sl.model.cpu().state_dict()
            rl_model = agent.rl.model.cpu().state_dict()

        torch.save(sl_model, path + '_' + str(agent.ID) + '_' + '.sl')
        # save rl strategy
        # 1.0 save the prarmeter
        torch.save(rl_model, path + '_' + str(agent.ID) + '_' + '.rl')
        # TODO add rl memory save
        # 2.0 save the memory of sl
        # save_memory(path + '_')

        if arguments.gpu:
            agent.sl.model.cuda()
            agent.rl.model.cuda()


def save_memory(path):
    for agent in Agents:
        # if len(agent.sl.memory.memory) > 0:
        #     sl_memory = SL_TRAN(*zip(*agent.sl.memory.memory))
        #     np.save(path + str(agent.ID) + '_slm_', sl_memory)

        # sl_state = np.vstack(sl_memory.state)
        # sl_action = np.vstack(sl_memory.policy)
        # sl_whole = np.append(sl_state, sl_action, axis=1)
        # np.save(path, sl_whole)
        # np.save(memory_name + '_' + str(agent.ID) + '_' + '.rl', np.array(agent.rl.memory.memory))
        if len(agent.rl.memory.memory) > 0:
            rl_memory = RL_TRAN(*zip(*agent.rl.memory.memory))
            np.save(path + str(agent.ID) + '_rlm_', rl_memory)
            # rl_state = np.vstack(rl_memory.state)
            # rl_reward = np.vstack(rl_memory.reward)
            # rl_action = np.vstack(rl_memory.action)
            # rl_whole = np.hstack((rl_state,rl_reward,rl_action))
            #
            # rl_nexts = np.array(rl_memory.next_state) # cause there are None value in next_state
            # np.save(path+'_ns_',rl_nexts)
            # np.save(path, rl_whole)


def save_table_csv(table):
    with open('../Data/table.csv', 'a') as fout:
        for i in range(table.size(0)):
            fout.write(str(table[i].sum()))
            fout.write(',')
        fout.write('\n')

street_list = []

def store_memory(env, agents):
    # store the memory
    for record_list in env.memory:
        for i in range(len(record_list)):
            state, action, reward = record_list[i]
            #            if reward[0] > 10 or reward[0] < -10:
            #                assert(False)
            #                print(reward[0])
            # the next state is the state when agents action next time
            if i + 1 >= len(record_list):
                next_state = None
            else:
                next_state, _, _ = record_list[i + 1]
            #            if reward[0] > 0:
            #                print(reward[0])
            # convert to tensor
            # street_list.append(state.street)
            state_tensor = env.state2tensor(state)
            # action_tensor = arguments.LongTensor(action)
            action_tensor = action
            next_state_tensor = env.state2tensor(next_state)
            #            reward.div_(arguments.stack*game_settings.player_count)

            #            agents[state.current_player].rl.memory.push(state_tensor, action_tensor, next_state_tensor, reward)
            #            print(reward[0])
            agents.rl.memory.push(state_tensor, action_tensor, next_state_tensor, reward.unsqueeze(1) / (1.0 * arguments.stack))


def get_reward(agents, env):
    iter_count = 0
    for record_list in env.memory:
        state, _, _, reward = record_list[-1]
        rewards[agents[state.current_player].ID] = rewards[agents[state.current_player].ID] + reward
        iter_count = iter_count + 1
    assert (iter_count == game_settings.player_count)


def print_dead(net):
    current_params = parameters_to_vector(net.parameters())
    dead_num = np.count_nonzero(np.less(current_params.data.cpu().numpy(), 0))
    if isinstance(net, DQNOptim):
        print('rl_dead: %d' % dead_num)
    else:
        print('sl_dead: %d' % dead_num)


######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` variable. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes.

time_start = 0


# @profile
def single_train():
    import time
    time_start = time.time()
    table_update_num = 0

    if arguments.load_model:
        load_model(arguments.load_model_num)
    Agent = Agents[0]

    for i_episode in range(arguments.epoch_count + 1):
        #        random.shuffle(Agents)

        # Initialize the environment and state
        env.reset()
        state = GameState()
        flag = 0 if random.random() > arguments.eta else 1
        #        import nn.Test.test_env as test
        #        test.test_five_card(state)

        for t in count():
            #            Agent = Agents[state.current_player]
            state_tensor = env.state2tensor(state)
            current_player = state.current_player
            # Select and perform an action
            #            print(state_tensor.size(1))
            assert (state_tensor.size(1) == arguments.dim_obs)

            if flag == 0:
                # sl
                #                action = Agents[current_player].sl.select_action(state_tensor)
                action = Agents[0].sl.select_action(state_tensor)[0]
            #                print("SL Action:"+str(action[0][0]))
            elif flag == 1:
                # rl
                #                action = Agents[current_player].rl.select_action(state_tensor)
                action = Agents[0].rl.select_action(state_tensor, state.current_player)[0]
            #                print("RL Action:"+str(action[0][0]))
            else:
                assert (False)

            is_rl = arguments.rl_model == 'dqn' or flag == 1 and arguments.rl_model == 'reinforce'
            next_state, done, action_taken,_ = env.step(state, action, is_rl)
            action[0][0] = action_taken

            if flag == 1:
                #                if Agents[0].rl.steps_done > Agents[0].rl.EPS_DECAY * 3:
                if Agent.rl.steps_done > arguments.sl_start:
                    # if choose sl store tuple(s,a) in supervised learning memory Msl
                    # state_tensor = env.state2tensor(state)
                    action_tensor = action[0]
                    Agent.sl.memory.push(state_tensor, action_tensor)
            #                print('Action:' + str(action[0][0]))

            if done:
                if arguments.rl_model == 'dqn':
                    store_memory(env, Agent)
                    #                for agent in Agents:
                    if (Agents[0].rl.steps_done - 1) % 200 == 0:  # remember that at first the two net shall be the same
                        Agents[0].rl.target_net.load_state_dict(Agents[0].rl.model.state_dict())
                        Agents[0].rl.steps_done += 1  # track always assign problem
                    if len(Agents[0].rl.memory.memory) >= Agents[
                        0].rl.memory.capacity / 10 and i_episode % arguments.rl_update == 0:
                        Agents[0].rl.optimize_model()
                elif arguments.rl_model == 'reinforce':
                    Agents[0].rl.finish_episode(env.memory)
                else:
                    raise (NotImplementedError)

                if len(Agent.sl.memory.memory) >= Agent.sl.memory.capacity / 10 and i_episode % arguments.sl_update == 0:
                    Agent.sl.optimize_model()
                if i_episode % 100 == 0 and i_episode > 0:
                    # Agent.sl.test(10)
                    Agent.rl.plot_error_vis(i_episode)
                    Agent.sl.plot_error_vis(i_episode)
                    # print dead relu
                    print_dead(Agent.rl.model)
                    print_dead(Agent.sl.model)
                    # print('episod: %d' % (i_episode))
                    print('episod: %d rl: %d,sl: %d' % (i_episode, \
                                                        len(Agents[0].rl.memory), \
                                                        len(Agent.sl.memory)))
                    # record the award
                #                    record_reward(Agents, env, Reward)
                if i_episode != 0 and i_episode % arguments.save_epoch == 0:
                    save_model(i_episode)
                break

            state = next_state

        #    dqn_optim.plot_error()
        #    global LOSS_ACC
        #    LOSS_ACC = dqn_optim.error_acc
        # save the model
        if arguments.load_model:
            i_episode = i_episode + arguments.load_model_num

    save_model(i_episode)
    print('Complete')
    print((time.time() - time_start))

def finish_episod(maddpg, record_a):
    # store the record to memory
    for i, record in enumerate(record_a):
        s,a,s1,r = record
        r = r.type(arguments.Tensor) / (arguments.stack*0.5)
        if i + 1 < len(record_a):
            s1 = record_a[i + 1].states
        maddpg.memory.push(s,a,s1,r)
    maddpg.episode_done += 1
    # c_loss, a_loss = maddpg.update_policy()
    # return c_loss, a_loss

def mul_train():
    import time
    time_start = time.time()
    table_update_num = 0
    maddpg = MADDPG()
    sl = SLOptim(state_dim=133)

    if arguments.load_model:
        maddpg.load("../Data/Model/Iter:" + str(arguments.load_model_num))

    for i_episode in range(arguments.epoch_count + 1):
        env.reset()
        state = GameState()
        record_a = []
        # print('episode:%d' % i_episode)
        for t in count():
            # Agent = Agents[state.current_player]
            next_state, done, state_a, action_a ,reward = env.step_r(state, maddpg, sl)

            state_a = torch.stack(state_a).squeeze(1)
            action_a = torch.stack(action_a).squeeze(1)
            reward = reward.unsqueeze(1)

            record_a.append(MADDPG_TRAN(state_a, action_a, None, reward))

            if done:
                finish_episod(maddpg, record_a)
                maddpg.episode_done += 1
                if i_episode % arguments.rl_update == 0:
                    maddpg.update_policy()
                if i_episode % arguments.sl_update == 0:
                    sl.optimize_model()

                if i_episode % 100 == 0:
                    print('episode:%d memory:%d' % (i_episode, len(maddpg.memory.memory)))
                    print_dead(maddpg.actors[0])

                    if maddpg.episode_done > maddpg.episodes_before_train:
                        maddpg.plot_error_vis(i_episode)
                        sl.plot_error_vis(i_episode)
                # record_reward(Agents, env, Reward)
                if i_episode != 0 and i_episode % arguments.save_epoch == 0:
                    maddpg.save("../Data/Model/Iter:" + str(i_episode))
                    torch.save(sl.model.state_dict(), "../Data/Model/Iter:" +
                                                             str(i_episode) + '_' + str(0) + '_' + '.sl')
                break

            state = next_state
#    dqn_optim.plt.ioff()
#    dqn_optim.plt.show()
# @profile
def pad2tensor(obs, pad_dim):
    return_obs = []
    for ob, dim in zip(obs, pad_dim):
        npob = np.pad(ob, (0,dim), 'constant', constant_values=0)
        return_obs.append(torch.from_numpy(npob).type(arguments.Tensor))
    return torch.stack(return_obs)
# @profile
def gym_maddpg_train():

    table_update_num = 0
    # simple | simple_adversary | simple_crypto | simple_push | simple_reference |
    # simple_speaker_listener | simple_spread | simple_tag | simple_world_comm
    env = make_env('simple_spread')
    n_agent = env.n
    obs_dim = np.array([shape.shape[0] for shape in env.observation_space])
    act_dim = env.action_space[0].n
    pad_dim = obs_dim.max() - obs_dim

    maddpg = MADDPG(n_agents=n_agent,
                    dim_obs=obs_dim.max(),
                    dim_act=act_dim,
                    episodes_before_train=50)
    sl = [SLOptim(state_dim=obs_dim.max()) for _ in range(n_agent)]

    import visdom
    vis = visdom.Visdom()
    win = None

    totalTime = 0

    max_steps = 300
    aver = np.zeros((n_agent,))

    if arguments.load_model:
        maddpg.load("../Data/Model/(gym)Iter:" + str(arguments.load_model_num))
        for i in range(n_agent):
            sl[i].model.load_state_dict(torch.load("../Data/Model/(gym)Iter:" +
                       str(arguments.load_model_num) + '_' + str(i) + '_' + '.sl'))
        # TODO remember to remove it when train
        # maddpg.steps_done = arguments.load_model_num

    for i_episode in range(arguments.epoch_count + 1):
        startTime = datetime.datetime.now()

        reward_record = []
        adversaries_reward_record = []
        agent_reward_record = []
        total_reward = 0.0
        adversaries_reward = 0.0
        agent_reward = 0.0
        rr = np.zeros((n_agent,))

        obs = env.reset()
        # convert obs to list(Tensor)
        obs = pad2tensor(obs, pad_dim)

        steps = 0
        # print('episode:%d' % i_episode)
        for t in count():

            is_rl = random.random() > arguments.eta  # 0 sl 1 rl
            has_sl = arguments.eta != 0 # 是否用了FSP，不用FSP就设置eta=0（全部rl）

            if is_rl:
                action = []
                for i in range(n_agent):
                    # a_n 是数值的action,sl需要
                    a_n, a = maddpg.select_action(i, Variable(obs[i]).unsqueeze(0))
                    action.append(a.squeeze())

                    # 不不用FSP就不push了
                    if has_sl:
                        for i in range(n_agent):
                            # 这里我需要数值的action
                            sl[i].memory.push(obs[i].unsqueeze(0),a_n)

            else:
                action = [sl[i].select_action(obs[i].unsqueeze(0))[1].squeeze() for i in range(n_agent)]

            np_action = [a.cpu().numpy() for a in action]
            obs_, reward, done, info = env.step(np_action)
            # convert obs to list(Tensor)
            obs_ = pad2tensor(obs_, pad_dim)
            action_a = torch.stack(action)
            reward_a = arguments.Tensor(reward)
            # print(reward)
            # 碰到就是最终回报
            # if reward_a[0:3].sum().item() > 1:
            #     # print('steps_____________%d'% steps)
            #     maddpg.memory.push(obs, action_a, None, reward_a)
            # else:
            maddpg.memory.push(obs, action_a, obs_, reward_a)

            reward = np.array(reward)
            # total_reward += reward.sum()
            adversaries_reward += reward[0:1].sum()
            # print(adversaries_reward)
            agent_reward = reward[1:].sum()
            rr += reward

            steps += 1

            # if steps % 4 == 0:
            #     maddpg.update_policy()

            if steps >= max_steps or all(done):
                # if i_episode % arguments.rl_update == 0:
                maddpg.update_policy()
                if has_sl :
                    for i in range(n_agent):
                        sl[i].optimize_model()
                break
            if i_episode % 100 == 0 and arguments.load_model: # show every 50 episode
                env.render()
            obs = obs_

            if arguments.evalation:
                env.render()

        # if i_episode % 100 == 0:
        print('episode:%d memory:%d' % (i_episode, len(maddpg.memory.memory)))
        # print_dead(maddpg.actors[0])

        if maddpg.episode_done > maddpg.episodes_before_train:
            maddpg.plot_error_vis(i_episode)
        # record_reward(Agents, env, Reward)
        if i_episode != 0 and i_episode % arguments.save_epoch == 0:
            maddpg.save("../Data/Model/(gym)Iter:" + str(i_episode))
            for i in range(n_agent):
                torch.save(sl[i].model.state_dict(), "../Data/Model/(gym)Iter:" +
                       str(i_episode) + '_' + str(i) + '_' + '.sl')
        maddpg.episode_done += 1

        endTime = datetime.datetime.now()
        runTime = (endTime - startTime).seconds
        totalTime = totalTime + runTime

        eps_threshold = maddpg.EPS_END + (maddpg.EPS_START - maddpg.EPS_END) * \
                        math.exp(-1. * maddpg.steps_done / maddpg.EPS_DECAY)
        # print('Episode:%d,reward = %f' % (i_episode, total_reward))
        print("EPS: %f" % eps_threshold)
        print('Episode:%d,adversaries_reward = %f' % (i_episode, adversaries_reward))
        print('Episode:%d,agent_reward = %f' % (i_episode, agent_reward))
        print('this episode run time:' + str(runTime))
        print('totalTime:' + str(totalTime))
        reward_record.append(total_reward)
        adversaries_reward_record.append(adversaries_reward)
        agent_reward_record.append(agent_reward)
        # aver = (i_episode / (i_episode + 1.0)) * aver + (1 / (i_episode + 1.0)) * rr

        if win is None:
            win = vis.line(X=np.arange(i_episode, i_episode + 1),
                           Y=np.array([rr]),
                           opts=dict(
                               ylabel='Reward',
                               xlabel='Episode',
                               title='MADDPG on MOE\n' +
                                     'agent=%d' % n_agent +
                                     ', sensor_range=0.2\n',
                               legend= ['Agent-%d' % i for i in range(n_agent)]))
        else:
            vis.line(X=np.array(
                [np.array(i_episode).repeat(n_agent)]),
                Y= np.array([rr]),
                win=win,
                update='append')

    env.close()
#    dqn_optim.plt.ioff()
#    dqn_optim.plt.show()

# Global counter
class Counter():
  def __init__(self):
    self.val = mp.Value('i', 0)
    self.lock = mp.Lock()

  def increment(self):
    with self.lock:
      self.val.value += 1

  def value(self):
    with self.lock:
      return self.val.value

from nn.acer import ActorCritic, AcerOptim, train as acer_train , test as acer_test

def gym_acer_train():
    mp.set_start_method('spawn', force=True)
    # # torch.manual_seed(1234)
    # if arguments.rl_model == 'dqn':
    #     single_train()
    # else:
    #     # mul_train()
    #     # gym_maddpg_train()
    #     gym_acer_tarin()


    # mp.set_start_method('spawn')
    # BLAS setup
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # Setup
    # mp.set_start_method(platform.python_version()[0] == '3' and 'spawn' or 'fork')  # Force true spawning (not forking) if available
    T = Counter()  # Global shared counter
    # Create shared network
    env = make_env('simple')

    n_agent = env.n
    obs_dim = np.array([shape.shape[0] for shape in env.observation_space])
    act_dim = env.action_space[0].n
    # pad_dim = obs_dim.max() - obs_dim
    # env.close()

    acer_optim = AcerOptim(obs_dim.max(), act_dim)

    if arguments.load_model:
        acer_optim.shared_model.load_state_dict(torch.load(arguments.WORK_PATH + '/nn/model.pth'))
        acer_optim.shared_average_model.load_state_dict(acer_optim.shared_model.state_dict())

    # acer_train(0, acer_optim, T)

    # Start validation agent
    processes = []
    p = mp.Process(target=acer_test, args=(0, acer_optim, T))
    p.start()
    processes.append(p)


    # Start training agents
    for rank in range(1, arguments.num_process + 1):
        p = mp.Process(target=acer_train, args=(rank, acer_optim, T))
        p.start()
        processes.append(p)

     # Clean up
    for p in processes:
        p.join()

def gym_ppo_train():
    import matplotlib.pyplot as plt
    from common.utils import agg_double_list

    # env_id = 'simple_tag'
    # env_id = 'simple_spread'
    env_id = 'simple_speaker_listener'
    env = make_env(env_id)
    env.seed(1234)
    env_eval = make_env(env_id)
    env_eval.seed(4321)
    state_dim = max([item.shape[0] for item in env.observation_space])
    action_dim = max([item.n for item in env.action_space])
    # for pad
    obs_dim = np.array([shape.shape[0] for shape in env.observation_space])
    obspad_dim = obs_dim.max() - obs_dim

    n_agent = env.n

    MAX_EPISODES = 200000
    MAX_STEPS = 2
    EPISODES_BEFORE_TRAIN = 10
    EVAL_EPISODES = 5
    EVAL_INTERVAL = 50

    # roll out n steps
    ROLL_OUT_N_STEPS = 50
    # only remember the latest ROLL_OUT_N_STEPS
    MEMORY_CAPACITY = ROLL_OUT_N_STEPS
    # MEMORY_CAPACITY = 50000
    # only use the latest ROLL_OUT_N_STEPS for training PPO
    BATCH_SIZE = ROLL_OUT_N_STEPS

    TARGET_UPDATE_STEPS = 20
    TARGET_TAU = 0.99

    REWARD_DISCOUNTED_GAMMA = 0.0
    TAU = 0.95
    ENTROPY_REG = 0.01
    #
    DONE_PENALTY = None

    HIDDEN_SIZE = 512
    CRITIC_LOSS = "huber"
    MAX_GRAD_NORM = None

    EPSILON_START = 0.50
    EPSILON_END = 0.05
    EPSILON_DECAY = 3000

    mappo = LSTMMAPPO(n_agent=n_agent, env=env, memory_capacity=MEMORY_CAPACITY,
              state_dim=state_dim, action_dim=action_dim,
              batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
              done_penalty=DONE_PENALTY, roll_out_n_steps=ROLL_OUT_N_STEPS, max_steps=MAX_STEPS,
              target_update_steps=TARGET_UPDATE_STEPS, target_tau=TARGET_TAU,
              actor_hidden_size=HIDDEN_SIZE, critic_hidden_size=HIDDEN_SIZE,
              reward_gamma=REWARD_DISCOUNTED_GAMMA, tau=TAU,
              epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
              epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
              episodes_before_train=EPISODES_BEFORE_TRAIN,
              critic_loss=CRITIC_LOSS,
              use_cuda=arguments.gpu, is_hidden=True, obspad_dim=obspad_dim)

    if arguments.load_model:
        mappo.load("../Data/Model/(mappo)Iter:%d" % (arguments.load_model_num))
        mappo.n_episodes = arguments.load_model_num

    episodes = []
    eval_rewards = []
    while mappo.n_episodes < MAX_EPISODES:
        mappo.interact()
        if arguments.evalation:
            rewards, _ = mappo.evaluation(env_eval, 100, EVAL_EPISODES)
            continue
        # print("Episode %d" % ppo.n_episodes )
        if mappo.n_episodes >= EPISODES_BEFORE_TRAIN:
            mappo.train()
        if mappo.episode_done and ((mappo.n_episodes + 1) % EVAL_INTERVAL == 0)\
                              and mappo.n_episodes >= EPISODES_BEFORE_TRAIN:
            mappo.save("../Data/Model/(mappo)Iter:" + str(mappo.n_episodes+1))
            rewards, _ = mappo.evaluation(env_eval, 100, EVAL_EPISODES)
            rewards_mu, rewards_std = agg_double_list(rewards)
            print("Episode %d, Average Reward %s" % (mappo.n_episodes + 1, str(rewards_mu)))
            episodes.append(mappo.n_episodes + 1)
            # TODO add multi agent reward

            # FIXME drop the first reward
            if rewards_mu[0]  > -500:
                eval_rewards.append(rewards_mu[0])
            else:
                eval_rewards.append(rewards_mu[0] / 100)
            mappo.plot_error_vis()

            episodesnp = np.array(episodes)
            eval_rewardsnp = np.array(eval_rewards)
            np.savetxt(arguments.WORK_PATH + "/Data/mappo/%s_ppo_episodes.txt" % env_id, episodesnp)
            np.savetxt(arguments.WORK_PATH + "/Data/mappo/%s_ppo_eval_rewards.txt" % env_id, eval_rewardsnp)

            plt.figure()
            plt.plot(episodesnp, eval_rewardsnp)
            plt.title("%s" % env_id)
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.legend(["PPO"])
            plt.savefig(arguments.WORK_PATH + "/Data/mappo/%s_ppo.png" % env_id)


if __name__ == "__main__":
    # gym_ppo_train()
    # gym_maddpg_train()
    single_train()