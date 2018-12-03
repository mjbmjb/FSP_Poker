import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from nn.make_env import make_env

env_id = 'simple_tag'
env = make_env(env_id)
env.seed(1234)

env.reset()
a = env.render()