from copyreg import pickle
import json
import traceback
import pandas as pd
import numpy as np
from multiprocessing.context import Process
from multiprocessing import Queue

import os
import sys
from jue_data.data.history_kline_data import cache_day_feat_data
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


def load_pickle_data(date, code, root):
    fn = os.path.join(root, code, str(date)+'.pkl')
    try:
        data = pickle.load(open(fn, 'rb'))
        # print(data.shape)
    except:
        with open('miss_data.txt', 'a') as f:
            s = f'{code}_{date}\n'
            f.write(s)
        print(s)
        traceback.print_exc()
    return data

def preprocess_order_book_data(df):
    df.drop(labels=['a5', 'a4', 'a3', 'a2', 'a1', 'b1', 'b2', 'b3', 'b4', 'b5'], axis=1, inplace=True)
    if 'code' in df.index.names: 
        df = df.droplevel(0)
    df['price'][df['price']<-10.] = 0.
    df['Vol'] = df['Vol']/3.75
    return df

def get_history_orderbook_data(code, date):
    df = load_pickle_data(date, code, '/mnt/stockdata/eastmoney_replay_data1/')
    df = preprocess_order_book_data(df)
    return df

def load_n_days_replay(code, date, stock_trade_dates, n=2):
    dfs = []
    date_idx = int(stock_trade_dates[code]['k2i'][date])
    for i, idx in enumerate(range(date_idx, date_idx-n, -1)):
        if idx < 0: 
            print(date, code, n, '没有那么多数据')
            break
        date = stock_trade_dates[code]['i2k'][str(idx)]
        # df = load_min_data(date, code, self.root_replay)
        df = get_history_orderbook_data(code, date)#, self.root_replay
        dfs.append(df)
    return dfs[::-1]

def padding2len(array, len, constant_values=0):
    data_len = array.shape[0]
    if data_len >= len:
        return array[:len, :]
    else:
        return np.pad(array, ((len-data_len, 0), (0, 0)), constant_values=constant_values)

def get_replay_data_batch(date, code, n_days):
    date = date.date().isoformat().replace('-', '')
    dfs = load_n_days_replay(code, date)
    return dfs

def debug_save_df(code, date, n_days):
    min_data = load_n_days_replay(code, date, n_days)
    min_data.to_csv(f'{code}_s_{date}_{n_days}days.csv')

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
        self.second_dir = config["second_dir"] + "/"
        self.n_days = config['n_days']
        self.second_dir = '/mnt/stockdata/eastmoney_replay_data1/'
        self.day_df = pickle.load(open(os.path.join(self.day_dir, 'df_train.pkl'), 'rb')) 
        self.index = self.day_df.index.remove_unused_levels()

        # self.sample_list = [tuple(self.index[i]) for i in range(len(self.day_df))]
        self.queue = Queue(1000)
        self.child = None
        self.order_df = None

    @staticmethod
    def _worker(day_df, second_dir, index, n_days, queue):

        while True:
            # print(ins)
            date, code = np.random.choice(index, 1)[0]
            second_data_dfs = get_replay_data_batch(date, code, n_days)

            day_feat = day_df.loc[(date, code), 'feature']
            day_label = day_df.loc[(date, code), 'label']
            # day_raw_df_index, day_raw_df_value, day_raw_df_column = toArray(second_data_dfs[-1])
            # 他需要用 raw_df 【vwap0， volume0】计算reward， volume0用来看最大能够卖出多少
            queue.put(
                (code, date, [day_feat, day_label] + second_data_dfs, False,),
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
                args=(self.day_df, self.second_dir, self.index, self.n_days, self.queue,),
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

    @staticmethod
    def _worker(day_df, second_dir, index, n_days, queue):

        for idx in index:
            # print(ins)
            date, code = idx
            second_data_dfs = get_replay_data_batch(date, code, n_days)

            day_feat = day_df.loc[(date, code), 'feature']

            day_label = day_df.loc[(date, code), 'label']

            queue.put(
                (code, date, [day_feat, day_label] + second_data_dfs, False,),
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
            args=(self.day_df, self.second_dir, self.index, self.n_days, self.queue,),
            daemon=True,
        )
        self.child.start()
