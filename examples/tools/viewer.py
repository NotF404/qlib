import  seaborn as sb
import pandas as pd
import pickle
import argparse


def plot_fig(df_path, save_path, save_fig = True):
    df = pickle.load(open(df_path, 'rb'))
    print()