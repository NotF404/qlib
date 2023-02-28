import pandas as pd
import numpy as np
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
from jue_data.data.feature_fn import timeseries_feature_fn

class JueObs():
    def __init__(self, config) -> None:
        feature_size = config['feature_size']
        self.perfect_info = config['perfect_info']
        self._observation_space = Tuple(
            (
                Box(-np.inf, np.inf, shape=(feature_size,), dtype=np.float32),
                Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
                Discrete(2),
            )
        )
    
    def get_space(self):
        return self._observation_space

    def __call__(self, *args, **kargs):
        return self.get_obs(*args, **kargs)

    def get_obs(
        self, sample, t, max_step, position, is_buy, *args, **kargs):
        if t == 0:
            timeseries_feature_fn(sample, is_buy=is_buy, aug=True)
            self.enc_data = sample['min_data_enc']
            self.enc_time = sample['min_data_enc_time']
            self.dec_data = sample['min_data_dec']
            self.dec_time = sample['min_data_dec_time']
            self.pred_start = sample['pred_start']
            self.t0_position = 1. if not is_buy else 0.

            dec_self_pos = np.array([[self.t0_position, 0.]]).repeat(self.pred_start, axis=0)
            dec_pred_pos = np.ones((241-self.pred_start, 2)) * -1
            self.sprivate_states_pos = np.concatenate([dec_self_pos, dec_pred_pos], axis=0)
            self.dec_mask = np.ones_like(self.sprivate_states_pos) * np.array([[0., -0.3]])

        index = t+self.pred_start
        self.sprivate_states_pos[index] = [position, t/max_step]
        self.dec_data[['position', 'time_step']] = self.sprivate_states_pos

        dec_data = self.dec_data.copy()
        if not self.perfect_info:
            dec_data['change'].iloc[index+1:] = self.dec_mask[index+1:, 0]
            dec_data['amount'].iloc[index+1:] = self.dec_mask[index+1:, 1]
        # assert not (
        #     np.isnan(list_private_state).any() | np.isinf(list_private_state).any()
        # ), f"{private_state}, {target}"
        # for k, p in self.public_state.items():
        #     assert not (np.isnan(p).any() | np.isinf(p).any()), f"{p}"
        return {"enc_data":self.enc_data[['change', 'amount']].values, 
                "enc_time":self.enc_time, 
                "dec_data":dec_data[['change', 'amount', 'position', 'time_step']].values.copy(), 
                "dec_time":self.dec_time, "index":index}