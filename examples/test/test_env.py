
import sys
import os
import shutil
from matplotlib import pyplot as plt
import numpy as np
sys.path.append('/mnt/data/quant/qlib/examples/')
from trade.env.env_jue import JueStockEnv
import yaml
config_path = "/mnt/data/quant/qlib/examples/trade/exp/example/OPDS/config_jue_ts_buy.yml"
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

tmp = '/mnt/data/quant/qlib/examples/test/tmp'
if os.path.exists(tmp):
    shutil.rmtree(tmp)

os.makedirs(tmp, exist_ok=True)
env.render(path=tmp)
while not env.done:
    a = env.action_space.sample()
    a = np.random.choice([2]*100 + [0, 1,3,4])
    print(a)
    a1 = env.step(a)
    env.render(path=tmp)
    # print(a1[0])
    # print(a1[1:])
# plot_action(env.traded_log, 'b.png')
