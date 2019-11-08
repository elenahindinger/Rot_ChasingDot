__author__ = 'Elena Maria Daniela Hindinger'

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import itertools as it
from scipy.io import loadmat
from Rot_ChasingDot_Functions import *

def plot_trajectory(df, xPos, yPos, new_filename):
    mpl.rcParams['agg.path.chunksize'] = 10000
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    axes.plot(df[xPos], df[yPos])
    axes.set(xlim=(0, 950), ylim=(0, 950))
    plt.tight_layout()
    fig.savefig(new_filename, bbox_inches='tight', format='tiff')
    plt.close('all')


def plot_time_spent_moving(exp, new_filename):
    exp_temp = exp.replace(-1, np.nan)
    tsmb = exp_temp.groupby('Trial')['boutCat'].count()
    tsmt = exp.groupby('Trial').size()
    tsm = tsmb.div(tsmt, axis=0) * 100

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    sns.despine()
    ax.plot(np.arange(41), tsm, color='k', marker='s')
    plt.xticks(np.arange(1, 41, 2), np.arange(1, 21))
    ax.set_xlabel('Trial', fontsize=20, labelpad=20)
    ax.set_ylabel('Time spent moving (%)', fontsize=20, labelpad=20)
    plt.tight_layout()
    fig.savefig(new_filename, bbox_inches='tight', format='tiff')
    plt.close('all')


def plot_distance(exp, new_filename):
    exp['Dist'] = distance(exp)
    temp = pd.concat([exp['StopWatchTime'], exp['Dist']], axis=1)
    temp['Time'] = pd.to_timedelta(temp.StopWatchTime, unit='s')
    dist = temp.drop(['StopWatchTime'], axis=1).resample('2s', on='Time').sum().reset_index()
    dist['Time2'] = dist['Time'].dt.total_seconds() / 60

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    sns.despine()
    ax.plot(dist.Time2, dist.Dist, color='k')
    ax.set_xlabel('Time (min)', fontsize=20, labelpad=20)
    ax.set_ylabel('Distance (mm)', fontsize=20, labelpad=20)
    plt.tight_layout()
    fig.savefig(new_filename, bbox_inches='tight', format='tiff')
    plt.close('all')


def plot_bouts_per_trials(bca, new_filename, colour_dict):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax = bca.plot(kind='bar', stacked=True,
                  color=[colour_dict.get(x, 'white') for x in bca.columns], ax=ax,  rot=0)
    plt.xticks(np.arange(1, 41, 2), np.arange(1, 21))
    ax.set_xlabel('Trial Number', fontsize=20, labelpad=20)
    ax.set_ylabel('Frequency of Bout Type', fontsize=20, labelpad=20)
    lg = ax.legend(loc=5, bbox_to_anchor=(1.15, 0.5), title='Bout Type', fontsize='x-large')
    lg.get_title().set_fontsize('18') #legend 'Title' fontsize
    plt.tight_layout()
    fig.savefig(new_filename, bbox_inches='tight', format='tiff')
    plt.close('all')


def plot_bouts_per_condition(bca, new_filename, colour_dict):
    hab = bca.iloc[0, :]
    trials = bca.iloc[1::2, :].sum(axis=0)
    breaks = bca.iloc[2::2, :].sum(axis=0)
    bg = pd.concat([hab, trials, breaks], axis=1)
    bg.columns = (['Habituation', 'Trials', 'ITI'])
    bg = bg.T

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax = bg.plot(kind='bar', stacked=True,
                 color=[colour_dict.get(x, 'white') for x in bg.columns], ax=ax, rot=0)
    plt.xticks(np.arange(3), ['Habituation', 'Trials', 'ITI'])
    ax.set_xlabel('Condition', fontsize=20, labelpad=20)
    ax.set_ylabel('Frequency of Bout Type', fontsize=20, labelpad=20)
    lg = ax.legend(loc=5, bbox_to_anchor=(1.15, 0.5), title='Bout Type', fontsize='x-large')
    lg.get_title().set_fontsize('18')  # legend 'Title' fontsize
    plt.tight_layout()
    fig.savefig(new_filename, bbox_inches='tight', format='tiff')
    plt.close('all')