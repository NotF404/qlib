import pandas as pd
import numpy as np
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
import math
import json


class BaseObs(object):
    """ """

    def __init__(self, config):
        self._observation_space = None

    def get_space(self):
        """ """
        return self._observation_space

    def get_obs(self, t):
        pass


class RuleObs(BaseObs):
    """The observation for minute-level rule-based agents, which consists of prediction, private state and direction information."""

    def __init__(self, config):
        feature_size = 0
        self.features = config["features"]
        self.time_interval = config["time_interval"]
        self.max_step_num = config["max_step_num"]
        for feature in self.features:
            feature_size += feature["size"]

        self._observation_space = Tuple(
            (
                Box(-np.inf, np.inf, shape=(feature_size,), dtype=np.float32),
                Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
                Discrete(2),
            )
        )

    def __call__(self, *args, **kargs):
        return self.get_obs(*args, **kargs)

    def _min_data_2_feature(self, df: pd.DataFrame, last_close: float):
        """_summary_

        Args:
            df (pd.DataFrame): _description_
        """
        df['close'] = (df['close'] / last_close - 1.) * 10.
        df['amount'] = np.log1p(df['amount']+1e-4) / 10. - 0.45
        return df[['close', 'amount']].values[:240]


    def get_feature_res(self, df_list, time, interval, whole_day=False, interval_num=8):
        """
        This method would extract the needed feature from the feature dataframe based on the feature name
        and the description in feature config.

        :param df_list: The dataframes of features, the order is consistent with the feature list.
        :param time: The index of current minute of the day (starting from -1).
        :param interval: The index of interval or decition making.
        :param whole_day: if True, this method would return the concatenate of all dataframe.(Default value = False)

        """
        day_feat, day_label, min_df2, min_df1 = df_list
        min_df1 = self._min_data_2_feature(min_df1, last_close=day_label.close1)
        min_df2 = self._min_data_2_feature(min_df2, last_close=day_label.close2)
        return {"day_feat":day_feat, "min_df1":min_df1, "min_df2":min_df2}


    def get_obs(self, raw_df, feature_dfs, t, interval, position, target, is_buy, *args, **kargs):
        private_state = np.array([position, target, t, self.max_step_num])
        prediction_state = self.get_feature_res(feature_dfs, t, interval)
        return {
            "prediction": prediction_state,
            "private": private_state,
            "is_buy": int(is_buy),
        }


class RuleInterval(RuleObs):
    """
    The observation for interval_level rule based strategy.

    Consist of interval prediction, private state, direction


    """

    def get_obs(
        self,
        raw_df,
        feature_dfs,
        t,
        interval,
        position,
        target,
        is_buy,
        max_step_num,
        interval_num,
        action=1.0,
        *args,
        **kargs
    ):
        private_state = np.array([position, target, interval - 1, interval_num])
        prediction_state = self.get_feature_res(feature_dfs, t, interval)
        return {
            "prediction": prediction_state,
            "private": private_state,
            "is_buy": int(is_buy),
            "action": action,
        }
