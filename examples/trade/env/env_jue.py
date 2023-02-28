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
        if self.is_buy:
            self.target_df = sample['next_day_min_data'].iloc[0:120]
        else:
            self.target_df = sample['min_dfs'][-1].iloc[121:241]

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
        self.day_vwap = self.target_df.iloc[-1].avg
        self.day_twap = np.nanmean(self.target_df["avg"].values)
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
        reward = 0.0
        reward += self.handle_pos(pos)

        if self.t == self.max_step_num - 1:
            self.done = True
            if self.is_buy:
                reward += self.handle_pos(1.0 - self.position)
            else:
                reward += self.handle_pos(self.position)

        if (self.position < ZERO and not self.is_buy) or (self.position >= 1.0 and self.is_buy):
            self.done = True

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

            for i, reward_func in enumerate(self.reward_func_list):
                if not reward_func.isinstant:
                    tmp_r = reward_func(performance_raise, 1.)
                    reward += tmp_r * self.reward_coef[i]
                    self.reward_log_dict[type(reward_func).__name__] += tmp_r

            self.state = self.obs(
                self.sample, 
                self.t, 
                self.max_step_num, 
                self.position, 
                self.is_buy
                )
            if self.log:
                res = pd.DataFrame(
                    {
                        "sell": not self.is_buy,
                        "vwap": this_vwap,
                        "this_vv_ratio": this_vv_ratio,
                        "reward": reward,
                        "PR": performance_raise,
                        'day_vwap': self.day_vwap,
                        'PA':PA

                    },
                    index=[[self.ins], [self.date]],
                )
            info = {
                "PR": performance_raise,
                "PA": PA,
            }

            info = merge_dicts(info, self.reward_log_dict)
            if self.log:
                info["df"] = self.traded_log
                info["res"] = res
                info['ins'] = self.ins
            del self.sample
            return self.state, reward, self.done, info

        else:
            self.t += 1
            self.state = self.obs(
                self.sample, 
                self.t, 
                self.max_step_num, 
                self.position, 
                self.is_buy
            )
            return self.state, reward, self.done, {}

    def handle_pos(self, pos):
        if pos == 0.: return 0
        if self.is_buy:
            self.position += pos
        else:
            self.position -= pos
        close = self.target_df.iloc[self.t].close

        self.traded_log.loc[self.traded_log.index[self.t], 'deal_pos'] = pos

        if self.is_buy:
            performance_raise = (1 - close / self.day_vwap) * 10000
            PA_t = (1 - close / self.day_twap) * 10000
        else:
            performance_raise = (close / self.day_vwap - 1) * 10000
            PA_t = (close / self.day_twap - 1) * 10000

        reward = 0
        for i, reward_func in enumerate(self.reward_func_list):
            if reward_func.isinstant:
                tmp_r = reward_func(performance_raise, pos)
                reward += tmp_r * self.reward_coef[i]
                self.reward_log_dict[type(reward_func).__name__] += tmp_r
        return reward

    def render(self, mode='human'):
        if not self.log: return
        # if self.background is None:
        self.traded_log['time'] = self.traded_log.index.strftime('%d%H%M')
        self.background = self.traded_log.plot(x='time', y='change', figsize=(20,4), ylim=[-0.12, 0.12])
        ax2 = self.background.twinx() 
        self.traded_log.plot(x='time', y=self.state['dec_data'][:,-2], color='orange', ax=ax2, title=f"{self.t}")