import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn.make_env import make_env

import torch as th

env_id = 'simple_spread'
m = th.load("/home/carc/mjb/maddpg-pytorch/models/" + env_id + '/' + env_id + "/run4/model.pt")

path = "/home/carc/mjb/deepStack/Data/stored_Model/%s/mappo_Iter" % (env_id)
for i in range(len(m['agent_params'])):
    save_path = path + '_' + str(i) + '.'
    th.save(m['agent_params'][i]['policy'], save_path + 'actor')
    th.save(m['agent_params'][i]['critic'], save_path + 'critic')
print('Done')