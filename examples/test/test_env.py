
import sys
import os
import numpy as np
sys.path.append('/mnt/data/quant/qlib/examples/')
from trade.env.env_jue import JueStockEnv
import yaml
config_path = "/mnt/data/quant/qlib/examples/trade/exp/example/OPDS/config_jue.yml"
with open(config_path, "r") as f:
    c = yaml.load(f)
print(c)

from trade.sampler.jue_sampler import Sampler
samp = Sampler(**c['io_conf']['train_sampler']['config'])
samp.reset()
samp.sample()
env = JueStockEnv(c['env_conf'])
# for _ in range(len(samp.index)):
    # print(_, samp.index[_])
sample = samp.sample()
# print(sample)
a0 = env.reset(samp.sample())
while not env.done:
    a = env.action_space.sample()
    a = np.random.choice([0]*20 + [1,2,3,4])
    print(a)
    a1 = env.step(a)
    print(a1[0])
    print(a1[1:])
