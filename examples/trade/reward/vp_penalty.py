import numpy as np
from .base import Instant_Reward


class VP_Penalty_small(Instant_Reward):
    """Reward: (Abs(vv_ratio_t - 1) * 10000 - v_t^2 * penalty) / 100"""

    def __init__(self, config):
        self.penalty = config["penalty"]

    def get_reward(self, performance_raise, v_t, target, *args):
        """

        :param performance_raise: Abs(vv_ratio_t - 1) * 10000.
        :param target: Target volume
        :param v_t: The traded volume
        """
        assert target > 0
        reward = performance_raise * v_t / target
        reward -= self.penalty * (v_t / target) ** 2
        assert not (np.isnan(reward) or np.isinf(reward)), f"{performance_raise}, {v_t}, {target}"
        return reward / 100


class VP_Penalty_small_vec(VP_Penalty_small):
    def get_reward(self, performance_raise, trade_pos, day_amp, *args):
        """

        :param performance_raise: Abs(vv_ratio_t - 1) * 10000.
        :param target: Target volume
        :param v_t: The traded volume
        """
        assert trade_pos > 0
        # reward = (performance_raise / 100. / (day_amp + 0.1)) * trade_pos  * 2
        reward = performance_raise / 100 * trade_pos
        if abs(reward) > 1.5:
            reward_value = 1.5 + (abs(reward) - 1.5) / (day_amp + 0.01)
            reward = reward_value if reward > 0 else - reward_value
        # reward -= self.penalty * ((v_t / target) ** 2).sum()# TODO: 不知道有无必要， 我想卖得越早越好(这个值很小， 影响还需要确认)
        assert not (np.isnan(reward) or np.isinf(reward)), f"{performance_raise}"
        return reward #归一化后买到最大值， 此时reward是0.5
