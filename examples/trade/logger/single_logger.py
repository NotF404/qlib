from datetime import datetime
import gc
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocessing import Queue, Process
import time

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def GLR(values):
    """

    Calculate -P(value | value > 0) / P(value | value < 0)

    """
    pos = []
    neg = []
    for i in values:
        if i > 0:
            pos.append(i)
        elif i < 0:
            neg.append(i)
    return -np.mean(pos) / np.mean(neg)


class DFLogger(object):
    """The logger for single-assert backtest.
    Would save .pkl and .log in log_dir


    """

    def __init__(self, log_dir, writer=None):
        self.log_dir = log_dir + "/"
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.queue = Queue(100000)
        self.raw_log_dir = self.log_dir

    @staticmethod
    def _worker(log_dir, queue):
        df_cache = {}
        stat_cache = {}
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        while True:
            info = queue.get(block=True)
            if info == "stop":
                summary = {}
                for k, v in stat_cache.items():
                    summary[k + "_std"] = np.nanstd(v)
                    summary[k + "_mean"] = np.nanmean(v)
                summary["GLR"] = GLR(stat_cache["PA"])

                # json.dump(stat_cache, 
                #         open(os.path.join(log_dir, f'infer_{datetime.now().isoformat()}.log'), 'w'), 
                #         indent=2, cls=NpEncoder)
                queue.put(summary)
                break
            elif len(info) == 0:
                continue
            else:
                df = info.pop("df")
                res = info.pop("res")
                ins = info.pop('ins')
                date = df.index[0].date().strftime('%Y%m%d')

                # print(os.path.join(log_dir, ins, str(date) + ".log"))
                if random.random() < 0.005:
                    os.makedirs(os.path.join(log_dir, ins), exist_ok=True)
                    plot_action(df, os.path.join(os.path.join(log_dir, ins, str(date) + ".png")))
                    # df.to_pickle(os.path.join(os.path.join(log_dir, ins, str(date) + ".pkl")))
                    # res.to_pickle(os.path.join(os.path.join(log_dir, ins, str(date) + ".log")))
                del df
                del res
                gc.collect()
                
                for k, v in info.items():
                    if k not in stat_cache:
                        stat_cache[k] = []
                    if hasattr(v, "__len__"):
                        stat_cache[k] += list(v)
                    else:
                        stat_cache[k].append(v)

    def reset(self):
        """ """
        while not self.queue.empty():
            self.queue.get()
        assert self.queue.empty()
        self.child = Process(target=self._worker, args=(self.log_dir, self.queue), daemon=True,)
        self.child.start()

    def set_step(self, step):

        self.log_dir = f"{self.raw_log_dir}{step}/"
        self.reset()

    def __call__(self, infos):
        for info in infos:
            if "env_id" in info:
                info.pop("env_id")
        self.update(infos)

    def update(self, infos):
        """store values in info into the logger"""
        for info in infos:
            self.queue.put(info, block=True)

    def summary(self):
        """:return: The mean and std of values in infos stored in logger"""
        summary = {}
        self.queue.put("stop", block=True)
        self.child.join()
        self.child.close()
        assert self.queue.qsize() == 1
        summary = self.queue.get()

        return summary


class InfoLogger(DFLogger):
    """ """

    def __init__(self, *args):
        self.stat_cache = {}
        self.queue = Queue(10000)
        self.child = Process(target=self._worker, args=(self.queue,), daemon=True)
        self.child.start()

    def _worker(logdir, queue):
        stat_cache = {}
        while True:
            info = queue.get(block=True)
            if info == "stop":
                summary = {}
                for k, v in stat_cache.items():
                    summary[k + "_std"] = np.nanstd(v)
                    summary[k + "_mean"] = np.nanmean(v)
                summary["GLR"] = GLR(stat_cache["PA"])
                queue.put(summary)
                stat_cache = {}
                time.sleep(5)
                continue
            if len(info) == 0:
                continue
            for k, v in info.items():
                if k == "res" or k == "df" or k == "ins":
                    continue
                if k not in stat_cache:
                    stat_cache[k] = []
                if hasattr(v, "__len__"):
                    stat_cache[k] += list(v)
                else:
                    stat_cache[k].append(v)

    def _update(self, info):
        if len(info) == 0:
            return
        for k, v in info.items():
            if k not in self.stat_cache:
                self.stat_cache[k] = []
            if hasattr(v, "__len__"):
                self.stat_cache[k] += list(v)
            else:
                self.stat_cache[k].append(v)

    def summary(self):
        """ """
        while not self.queue.empty():
            # print('not empty')
            # print(self.queue.qsize())
            time.sleep(1)
        self.queue.put("stop")
        # self.child.join()
        time.sleep(1)
        while not self.queue.qsize() == 1:
            # print(self.queue.qsize())
            time.sleep(1)
        assert self.queue.qsize() == 1
        summary = self.queue.get()

        return summary

    def set_step(self, step):
        return


def plot_action(df, path, title=''):
    df['action'] = df['deal_pos']
    df['time'] = df.index.strftime('%d%H%M')
    ax1 = df.plot.scatter(x='time', y='action', figsize=(20,4), ylim=[-0.1, 0.5])
    ax2 = ax1.twinx() 
    fig = df.plot(x='time', y='change', figsize=(20,4), color='orange', ax=ax2, title=title)
    plt.savefig(path)
    plt.close()

