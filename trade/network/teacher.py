import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import sys

from tianshou.data import to_torch
from .model import NSTransformer

class Teacher_Extractor(nn.Module):
    def __init__(self, device="cuda:0", **kargs):
        super().__init__()
        self.device = device
        hidden_size = kargs["hidden_size"]

        self.rnn = nn.GRU(64, hidden_size, batch_first=True)
        self.rnn2 = nn.GRU(64, hidden_size, batch_first=True)
        self.dnn = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),)
        self.cnn = nn.Sequential(nn.Conv1d(2, 64, 7, padding=3), nn.ReLU(),)
        self.raw_fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 128), nn.LayerNorm(128),
        )

    # def _init_weights(self):

    def _to_device(self, data):
        return torch.from_numpy(data).float().to(self.device)

    def obs2arg(self, obs):
        return self._to_device(obs['dec_data']), obs['index']

    def forward(self, inp):

        x_dec, index = self.obs2arg(inp)
        raw_in = x_dec[:,:,:-2]
        dnn_in = x_dec[:,:,-2:]
        cnn_out = self.cnn(raw_in.transpose(2,1)).transpose(2, 1)
        rnn_in = self.raw_fc(cnn_out)
        rnn2_in = self.dnn(dnn_in)
        rnn2_out = self.rnn2(rnn2_in)[0]
        rnn_out = self.rnn(rnn_in)[0][:, -1, :]
        rnn2_out = rnn2_out[:, -1]
        # dnn_out = self.dnn(dnn_in)
        fc_in = torch.cat((rnn_out, rnn2_out), dim=-1)
        feature = self.fc(fc_in)
        return feature


class Teacher_Actor(nn.Module):
    def __init__(self, extractor, in_shape=128, out_shape=1, device=torch.device("cpu"), **kargs):
        super().__init__()
        self.extractor = extractor
        self.layer_out = nn.Sequential(nn.Linear(in_shape, out_shape), nn.Softmax(dim=-1))
        self.device = device

    def forward(self, obs, state=None, info={}):
        self.feature = self.extractor(obs)
        out = self.layer_out(self.feature)
        return out, state


class Teacher_Critic(nn.Module):
    def __init__(self, extractor, in_shape=128, out_shape=1, device=torch.device("cpu"), **kargs):
        super().__init__()
        self.extractor = extractor
        self.value_out = nn.Linear(in_shape, out_shape)
        self.device = device

    def forward(self, obs, state=None, info={}):
        self.feature = self.extractor(obs)
        return self.value_out(self.feature).squeeze(-1)
