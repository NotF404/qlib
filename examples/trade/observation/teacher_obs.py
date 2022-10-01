import pandas as pd
import numpy as np
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
import math
import json

from .obs_rule import RuleObs


class TeacherObs(RuleObs):
    """
    The Observation used for OPD method.

    Consist of public state(raw feature), private state, seqlen

    """
            # second_feat = second_data_df[['price', 'Vol', 'yclose', 'pos']].values.astype(np.double)
            # mask = np.zeros((len(second_feat), 1))
            # second_feat, mask = padding2len(second_feat, n_days * 5100), padding2len(mask, n_days * 5100, constant_values=1)
    def get_obs(
        self, raw_df, feature_dfs, t, interval, position, target, is_buy, max_step_num, interval_num, *args, **kargs,
    ):
        if t == -1:
            self._last_position = position / target
            self.private_states = [[self._last_position, 0.]]
            self.public_state = self.get_feature_res(feature_dfs, t, interval, whole_day=True)
        else:
            last_point = len(self.private_states)
            step = (self._last_position - position / target) / (t - last_point + 2)
            private_state = [[self._last_position - step * (t0 - last_point), (t0 + 1) / max_step_num] for t0 in range(last_point, t+2)]
            self._last_position = position / target
            self.private_states.extend(private_state)
        # list_private_state = np.concatenate(self.private_states)
        list_private_state = np.concatenate(
            (self.private_states, [[0.0, 0.0]] * (240 - len(self.private_states)),)
        )
        seqlen = np.array([240])
        assert not (
            np.isnan(list_private_state).any() | np.isinf(list_private_state).any()
        ), f"{private_state}, {target}"
        for k, p in self.public_state.items():
            assert not (np.isnan(p).any() | np.isinf(p).any()), f"{p}"
        return {"pub_state":self.public_state, "pri_state":list_private_state, "seqlen":seqlen}


class RuleTeacher(RuleObs):
    """ """

    def get_obs(
        self, raw_df, feature_dfs, t, interval, position, target, is_buy, max_step_num, interval_num, *args, **kargs,
    ):
        if t == -1:
            self.private_states = []
        public_state = feature_dfs[0].reshape(-1)[: 6 * 240]
        private_state = np.array([position / target, (t + 1) / max_step_num])
        teacher_action = self.get_feature_res(feature_dfs, t, interval)[-self.features[1]["size"] :]
        self.private_states.append(private_state)
        list_private_state = np.concatenate(self.private_states)
        list_private_state = np.concatenate(
            (list_private_state, [0.0] * 2 * (interval_num + 1 - len(self.private_states)),)
        )
        seqlen = np.array([interval])
        return np.concatenate((teacher_action, public_state, list_private_state, seqlen))
