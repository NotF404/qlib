import pandas as pd
import numpy as np
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
from jue_data.data.feature_fn import timeseries_feature_fn, fix_len_timeseries_feature_fn, _df_2_feature

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


class JueTSObs():
    def __init__(self, config) -> None:
        feature_size = config['feature_size']
        self.ts_len = config['ts_len']
        self.is_inference = config['is_inference']
        self.initiated = False
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

    def get_inference_obs(self, sample, t, max_step, position, is_buy):
        df_pred = sample['min_dfs'].pop(-1)
        if not self.initiated:
            fix_len_timeseries_feature_fn(sample, is_buy=is_buy, aug=True)
            self.dec_data = sample['min_data_dec']

            self.pred_start = len(self.dec_data) if is_buy else len(self.dec_data) + 121
            self.t0_position = 1. if not is_buy else 0.
            self.dec_data['position'] = self.t0_position
            self.dec_data['time_step'] = 0.

            self.private_state = []
            self.t = -1
            self.initiated = True
        df_pred = _df_2_feature(df_pred, True)
        df_pred['amount'] = df_pred['amount'] * 3.
        df_pred = df_pred[['change', 'amount']]

        if self.t < t-1:
            print('self.t < t-1:', f"{self.t}_{ t-1}")
            self.private_state += [self.private_state[-1]] * (t - self.t - 1)
        elif self.t > t-1:
            print('self.t > t-1:', f"{self.t}_{ t-1}")
        self.private_state.append([position, t/max_step])
        df_pred[['position', 'time_step']] = self.private_state

        dec_data = self.dec_data.iloc[self.ts_len-len(df_pred):].append(df_pred, axis=0)
        self.t = t
        # assert not (
        #     np.isnan(list_private_state).any() | np.isinf(list_private_state).any()
        # ), f"{private_state}, {target}"
        # for k, p in self.public_state.items():
        #     assert not (np.isnan(p).any() | np.isinf(p).any()), f"{p}"
        return {
                "dec_data":dec_data[['change', 'amount', 'position', 'time_step']].values.copy(), 
                "index":t}

    def get_obs(
        self, sample, t, max_step, position, is_buy, *args, **kargs):
        if t == 0:
            fix_len_timeseries_feature_fn(sample, is_buy=is_buy, aug=True)

            self.dec_data = sample['min_data_dec']
            self.pred_start = len(self.dec_data) - 120 if is_buy else len(self.dec_data) - 241
            self.t0_position = 1. if not is_buy else 0.
            self.dec_data['position'] = self.t0_position
            self.dec_data['time_step'] = 0.

        self.index = t+self.pred_start
        df_index =  self.dec_data.index[self.index]
        self.dec_data.loc[df_index, 'position'] = position
        self.dec_data.loc[df_index, 'time_step'] = t/max_step

        dec_data = self.dec_data.iloc[self.index+1-self.ts_len:self.index+1].copy()

        # assert not (
        #     np.isnan(list_private_state).any() | np.isinf(list_private_state).any()
        # ), f"{private_state}, {target}"
        # for k, p in self.public_state.items():
        #     assert not (np.isnan(p).any() | np.isinf(p).any()), f"{p}"
        return {
                "dec_data":dec_data[['change', 'amount', 'position', 'time_step']].values.copy(), 
                "index":self.index}

    def render_obs(self):
        d = self.dec_data.iloc[self.index+1-self.ts_len:self.index+1]
        d['time'] = d.index.strftime('%H%M')
        print(d.describe())
        fig = d.plot(x='time', y=['change', 'amount', 'position', 'time_step'], subplots=True, kind='line', title=f"{self.index}_{d.index[-1]}")
        return fig