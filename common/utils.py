
import torch as th
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import Settings.arguments as arguments


def identity(x):
    return x


def entropy(p):
    return -th.sum(p * th.log(p), 1)


def kl_log_probs(log_p1, log_p2):
    return -th.sum(th.exp(log_p1)*(log_p2 - log_p1), 1)


def index_to_one_hot(index, dim):
    eye = th.eye(dim).to(arguments.device)
    one_hot = [eye[i.squeeze()] for i in index] if isinstance(index, list) or isinstance(index , tuple) \
               else eye[index]
    return one_hot

def cut_action(actions, action_space):
    return [action[0:item.n] for action, item in zip(actions,action_space)]

def to_tensor(x, device = arguments.device, dtype="float"):
    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return th.FloatTensor(x).to(device)
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return th.LongTensor(x).to(device)
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return th.ByteTensor(x).to(device)
    else:
        x = np.array(x, dtype=np.float64).tolist()
        return th.FloatTensor(x).to(device)


def agg_double_list(l):
    # l: [ [...], [...], [...] ]
    # l_i: result of each step in the i-th episode
    s = [np.sum(np.array(l_i), 0) for l_i in l]
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    return s_mu, s_std
# @params x: log_pro

def logpro2entropy(logpros):
    # x: log_pro
    probs = th.exp(logpros)
    dist_entropy = -(logpros * probs).sum(-1).mean()
    return dist_entropy

def padobs(obs, pad_dim):
    return_obs = []
    for ob, dim in zip(obs, pad_dim):
        npob = np.pad(ob, (0,dim), 'constant', constant_values=0)
        return_obs.append(npob)
    return return_obs
