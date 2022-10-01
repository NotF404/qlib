import sys
import os
sys.path.append('/mnt/data/quant/qlib/examples/')
from trade.sampler.jue_sampler import JueSampler
import yaml
config_path = "/mnt/data/quant/qlib/examples/trade/exp/example/OPDT/config_jue.yml"
with open(config_path, "r") as f:
    c = yaml.load(f)
print(c)
samp = JueSampler(c)
samp.reset()
print(samp.sample())