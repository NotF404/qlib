import pickle 
import json
import time
import traceback
import pandas as pd
import numpy as np
from multiprocessing.context import Process
from multiprocessing import Queue

import os
import sys
from jue_data.data.history_kline_data import cache_day_feat_data
from jue_data.data.history_min_data import load_n_day_min_data
sys.path.append("..")


def toArray(data):
    if type(data) == np.ndarray:
        return data

    elif type(data) == list:
        data = np.array(data)
        return data

    elif type(data) == pd.DataFrame:
        share_index = toArray(data.index)
        share_value = toArray(data.values)
        share_colmns = toArray(data.columns)
        return share_index, share_value, share_colmns

    else:
        try:
            share_array = np.array(data)
            return share_array
        except:
            raise NotImplementedError

    
class JueSampler:
    """The sampler for training of single-assert RL."""

    def __init__(self, config):
        self.day_dir = cache_day_feat_data(
            config["train_start_date"],
            config["train_end_date"],
            config["valid_start_date"],
            config["valid_end_date"],
            config["test_start_date"],
            config["test_end_date"],
            config['instruments'],
            config["day_data_cache_dir"],
        )
        self.min_data_dir = config["min_data_dir"]
        self.n_days = config['n_days']
        self.stock_trade_dates = json.load(open(config['stock_trade_dates'], 'r'))
        self.day_df = pickle.load(open(os.path.join(self.day_dir, 'df_train.pkl'), 'rb')) 
        self.index = self.day_df.index.remove_unused_levels()

        # self.sample_list = [tuple(self.index[i]) for i in range(len(self.day_df))]
        self.queue = Queue(1000)
        self.child = None
        self.order_df = None

    @staticmethod
    def _worker(day_df, min_data_dir, index, n_days, stock_trade_dates, queue):

        while True:
            # print(ins)
            date, code = np.random.choice(index, 1)[0]
            min_data_dfs = load_n_day_min_data(code, date, min_data_dir, stock_trade_dates, n_days)

            day_feat = day_df.loc[(date, code), 'feature']
            day_label = day_df.loc[(date, code), 'label']
            # day_raw_df_index, day_raw_df_value, day_raw_df_column = toArray(second_data_dfs[-1])
            # 他需要用 raw_df 【vwap0， volume0】计算reward， volume0用来看最大能够卖出多少
            queue.put(
                (code, date, [day_feat, day_label] + min_data_dfs, False,),
                block=True,
            )

    def _sample_ins(self):
        """ """
        return np.random.choice(self.index, 1)[0][1]

    def reset(self):
        """ """
        if self.child is None:
            self.child = Process(
                target=self._worker,
                args=(self.day_df, self.min_data_dir, self.index, self.n_days, self.stock_trade_dates, self.queue,),
                daemon=True,
            )
            self.child.start()

    def sample(self):
        """ """
        sample = self.queue.get(block=True)
        return sample

    def stop(self):
        """ """
        try:
            self.child.terminate()
        except:
            for p in self.child:
                p.terminate()


class JueTestSampler(JueSampler):
    """The sampler for backtest of single-assert strategies."""

    def __init__(self, config):
        super().__init__(config)
        self.ins_index = -1
        self.day_df = pickle.load(open(os.path.join(self.day_dir, 'df_test.pkl'), 'rb')) 
        self.index = self.day_df.index.remove_unused_levels()

    @staticmethod
    def _worker(day_df, min_data_dir, index, n_days, stock_trade_dates, queue):

        for idx in index:
            # print(ins)
            date, code = idx
            min_data_dfs = load_n_day_min_data(code, date, min_data_dir, stock_trade_dates, n_days)

            day_feat = day_df.loc[(date, code), 'feature']
            day_label = day_df.loc[(date, code), 'label']

            queue.put(
                (code, date, [day_feat, day_label] + min_data_dfs, False,),
                block=True,
            )
        for _ in range(100):
            queue.put(None)

    def reset(self):
        """

        reset the sampler and change self.order_dir if order_dir is not None.

        """
        if not self.child is None:
            self.child.terminate()
            while not self.queue.empty():
                self.queue.get()
        self.child = Process(
            target=self._worker,
            args=(self.day_df, self.min_data_dir, self.index, self.n_days, self.stock_trade_dates, self.queue,),
            daemon=True,
        )
        self.child.start()

class JueInferSampler(JueSampler):
    """The sampler for backtest of single-assert strategies."""

    def __init__(self, config, dataset):
        super().__init__(config)
        self.ins_index = -1
        self.day_df = pickle.load(open(os.path.join(self.day_dir, f'df_{dataset}.pkl'), 'rb')) 
        self.index = self.day_df.index.remove_unused_levels()

    def sample(self, date, code):

        min_data_dfs = load_n_day_min_data(code, date, self.min_data_dir, self.stock_trade_dates, self.n_days)

        day_feat = self.day_df.loc[(date, code), 'feature']
        day_label = self.day_df.loc[(date, code), 'label']

        return code, date, [day_feat, day_label] + min_data_dfs, False


    def reset(self):
        pass
    
    def stop(self):
        pass

