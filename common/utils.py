
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
    eye = th.eye(dim)
    if arguments.gpu:
        eye = eye.cuda()
    one_hot = [eye[i] for i in index] if isinstance(index, list) else \
               eye[index]
    return one_hot


def to_tensor(x, use_cuda = arguments.gpu, dtype="float"):
    FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor
    LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor
    ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return FloatTensor(x)
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return LongTensor(x)
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return ByteTensor(x)
    else:
        x = np.array(x, dtype=np.float64).tolist()
        return FloatTensor(x)


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