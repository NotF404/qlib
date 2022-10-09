
import sys
import os
import numpy as np
sys.path.append('/mnt/data/quant/qlib/examples/')
from trade.env.env_jue import JueStockEnv, JueStockEnv_Acc
import yaml
config_path = "/mnt/data/quant/qlib/examples/trade/exp/example/OPDS/config_jue.yml"
with open(config_path, "r") as f:
    c = yaml.load(f)
print(c)

from trade.sampler.jue_sampler import JueSampler, JueTestSampler
samp = JueTestSampler(c)
samp.reset()
env = JueStockEnv_Acc(c['env_conf'])
for _ in range(len(samp.index)):
    print(_, samp.index[_])
    sample = samp.sample()
    (
        ins,
        date,
        feature_dfs,
        is_buy,
    ) = sample
    a0 = env.reset(samp.sample())
    a1 = env.step(0)
    a1 = env.step(0)
    a1 = env.step(0)
    a1 = env.step(0)
    a1 = env.step(0)
    a1 = env.step(1)
    a1 = env.step(3)
    a2 = env.step(0)
    a2 = env.step(2)
    a2 = env.step(2)
