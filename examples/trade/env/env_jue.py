import os
import gym

gym.logger.set_level(40)
import numpy as np
import pandas as pd
import pickle as pkl

import sys

sys.path.append("..")
from trade.util import merge_dicts
from trade import reward
from trade import observation
from trade import action

ZERO = 1e-7


class JueStockEnv(gym.Env):
    """Single-assert environment"""

    def __init__(self, config):
        if "log" in config:
            self.log = config["log"]
        else:
            self.log = True
        # loader_conf = config['loader']['config']
        self.is_buy = config["is_buy"]
        self.target_df_start = config["target_df_start"]
        self.target_df_end = config["target_df_end"]
        obs_conf = config["obs"]["config"]
        self.obs = getattr(observation, config["obs"]["name"])(obs_conf)
        self.action_func = getattr(action, config["action"]["name"])(config["action"]["config"])
        self.reward_func_list = []
        self.reward_log_dict = {}
        self.reward_coef = []
        for name, conf in config["reward"].items():
            self.reward_coef.append(conf["coefficient"])
            self.reward_func_list.append(getattr(reward, name)(conf))
            self.reward_log_dict[name] = 0.0
        self.observation_space = self.obs.get_space()
        self.action_space = self.action_func.get_space()
        self.target_money = 5000000
        self.penalty_time = 0.01 / 120 / 120 # 所有step的惩罚加起来在0.5%

    def toggle_log(self, log):
        self.log = log

    def reset(self, sample):
        """

        :param sample:

        """

        for key in self.reward_log_dict.keys():
            self.reward_log_dict[key] = 0.0

        sample = sample[0]
        self.ins = sample['meta_info']['instrument']
        self.date = sample['meta_info']['datetime']
        # 去掉batch
        # close       avg  volume/100     amount/1000    change

        self.target_df = sample['min_dfs'][-1].iloc[self.target_df_start: self.target_df_end]

        self.sample = sample

        self.t = 0
        self.max_step_num = len(self.target_df)

        self.position = 0. if self.is_buy else 1.0

        self.state = self.obs(
            self.sample,
            self.t,
            self.max_step_num,
            self.position,
            self.is_buy,
        )
        self.traded_log = self.target_df.copy(deep=True)
        self.traded_log['deal_pos'] = 0.
        self.traded_log['reward'] = 0.
        self.day_vwap = np.nansum(self.target_df["close"] * self.target_df['amount']) / self.target_df['amount'].sum()
        self.day_twap = np.nanmean(self.target_df["close"].values)
        self.day_high = self.target_df.close.max()
        self.day_low = self.target_df.close.min()
        self.day_amp = (self.day_high - self.day_low) / self.day_low * 100
        try:
            assert not (np.isnan(self.day_vwap) or np.isinf(self.day_vwap))
        except:
            print(self.ins)
            print(self.day_vwap)

        self.done = False

        self.this_cash = 0

        self.total_reward = 0
        self.total_instant_rew = 0
        self.last_rew = 0
        return self.state

    def step(self, action):
        """

        :param action:

        """
        assert not self.done
        pos = self.action_func(
            action,
            self.position,
        )
        self.traded_log.loc[self.traded_log.index[self.t], 'deal_pos'] = pos

        reward = 0.0

        if self.t == self.max_step_num - 1:
            self.done = True
            if self.is_buy:
                pos = 1.0 - self.position
            else:
                pos = self.position
            # reward -= 0.2
            self.traded_log.loc[self.traded_log.index[self.t], 'deal_pos'] = pos
        # else: 
        reward += self.handle_pos(pos)

        if (self.position < ZERO and not self.is_buy) or (self.position >= 1.0 and self.is_buy):
            self.done = True

        self.traded_log.loc[self.traded_log.index[self.t], 'reward'] = reward

        self.t += 1
        if self.done:
            this_vwap = (self.traded_log['close'] * self.traded_log['deal_pos']).sum()
            # 卖出的平均价格/天平均价格， 不加权的计算方法
            this_vv_ratio = this_vwap / self.day_vwap
            this_tt_ratio = self.traded_log['close'][self.traded_log['deal_pos']>0].mean() / self.day_twap

            if self.is_buy:
                performance_raise = (1 - this_vv_ratio) * 10000
                PA = (1 - this_tt_ratio) * 10000
            else:
                performance_raise = (this_vv_ratio - 1) * 10000
                PA = (this_tt_ratio - 1) * 10000

            # for i, reward_func in enumerate(self.reward_func_list):
            #     if reward_func.isinstant():
            #         tmp_r = reward_func(PA, 1.)
            #         reward += tmp_r * self.reward_coef[i]
            #         self.reward_log_dict[type(reward_func).__name__] += tmp_r

            self.state = self.obs(
                self.sample, 
                self.t, 
                self.max_step_num, 
                self.position, 
                self.is_buy
                )
            info = {
                "PR": performance_raise,
                "PA": PA,
                "reward": reward,
                # 'vwap':self.day_vwap,
                # 'twap':self.day_twap
            }

            info = merge_dicts(info, self.reward_log_dict)
            if self.log:
                info["df"] = self.traded_log
                info['ins'] = self.ins
            del self.sample
            return self.state, reward, self.done, info

        else:
            self.state = self.obs(
                self.sample, 
                self.t, 
                self.max_step_num, 
                self.position, 
                self.is_buy
            )
            return self.state, reward, self.done, {}

    def handle_pos(self, pos):
        reward = 0.
        if pos == 0.: return reward
        
        close = self.target_df.iloc[self.t].close
        if self.is_buy:
            performance_raise = (1 - close / self.day_vwap) * 10000
            PA_t = (1 - close / self.day_twap) * 10000
        else:
            performance_raise = (close / self.day_vwap - 1) * 10000
            PA_t = (close / self.day_twap - 1) * 10000

        # reward = - self.t * self.penalty_time
        # if pos == 0.: return reward - PA_t * 0.0001

        if self.is_buy:
            self.position += pos
        else:
            self.position -= pos

        for i, reward_func in enumerate(self.reward_func_list):
            if reward_func.isinstant():
                tmp_r = reward_func(PA_t, pos, self.day_amp)
                reward += tmp_r * self.reward_coef[i] # 按涨幅做归一化
                self.reward_log_dict[type(reward_func).__name__] += tmp_r
        if abs(reward) > 3: self.log_info(f'reward>3, {reward}, {PA_t}, {close}, {performance_raise}, {pos}')
        return reward

    def log_info(self, msg):
        print(msg, ':\n' ,'day_high', self.day_high, 
                'day_low', self.day_low, 
                'day_vwap', self.day_vwap, 
                'day_twap', self.day_twap,
                'day_amp',self.day_amp)
        print('reward_log_dict:\n', self.reward_log_dict)

    def render(self, mode='human', path='.'):
        # import matplotlib.pyplot as plt
        df = self.traded_log
        df['time'] = df.index.strftime('%H%M')
        # df['reward'] = df['reward'] * 0.01

        if self.t != 0: 
            fig_action = df.plot(x='time', y=['reward', 'change'], figsize=(8,4), 
                        subplots=True, kind='line', title=f"{self.t-1}_{df.index[self.t-1]}")
            ax2 = fig_action[1].twinx() 
            df.plot.scatter(x='time', y='deal_pos', figsize=(8,4), ylim=[-0.1, 1.1], ax=ax2)
            fig_action[0].figure.savefig(os.path.join(path, f'action_{self.t-1}.png'))
        fig_obs = self.obs.render_obs()
        # for i, ax in enumerate(fig_obs):
        fig_obs[0].figure.savefig(os.path.join(path, f'obs_{self.t}.png'))