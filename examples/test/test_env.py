
import sys
import os
from matplotlib import pyplot as plt
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
def plot_action(df, path, title=''):
    df['action'] = df['deal_pos']
    df['time'] = df.index.strftime('%H%M')
    df['reward'] = df['reward'] * 0.1

    ax1 = df.plot(x='time', y='reward', figsize=(12,4), color='red')
    df.plot(x='time', y='change', figsize=(12,4), color='orange', ax=ax1, title=title)
    ax2 = ax1.twinx() 

    df.plot.scatter(x='time', y='action', figsize=(12,4), ylim=[-0.1, 0.5], ax=ax2)
    

    plt.savefig(path)
    plt.close()
    
a0 = env.reset(samp.sample())
while not env.done:
    a = env.action_space.sample()
    a = np.random.choice([1]*20 + [0,2])
    print(a)
    a1 = env.step(a)
    # print(a1[0])
    print(a1[1:])
plot_action(env.traded_log, 'b.png')
