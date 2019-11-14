__author__ = 'Elena Maria Daniela Hindinger'

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import itertools as it
from scipy.io import loadmat
from matplotlib.gridspec import GridSpec
from Rot_ChasingDot_Functions import *


''' Universal Bout Parameters '''
ordered_bouts = ['AS', 'Slow1', 'Slow2', 'ShortCS', 'LongCS', 'BS', 'J-Turn', 'HAT', 'RT', 'SAT', 'O-bend', 'LLC', 'SLC']
idx = [11, 7, 9, 1, 2, 3, 5, 13, 8, 12, 4, 10, 6]
numbers, unordered_bouts = (list(t) for t in zip(*sorted(zip(idx, ordered_bouts))))

colour_dict = {'AS': [0.4, 1, 1], 'Slow1': [0, 0.588235294, 1], 'Slow2': [0, 0, 0.784313725],
               'ShortCS': [0.392156863, 0.392156863, 0.392156863], 'LongCS': [0, 0, 0], 'BS': [1, 0.666666667, 0],
               'J-Turn': [0.980392157, 0.501960784, 0.447058824], 'HAT': [0.411764706, 1, 0.4], 'RT': [0, 0.6, 0],
               'SAT': [0.576470588, 0.439215686, 0.858823529], 'O-bend': [0.862745098, 0, 0.862745098],
               'LLC': [0.4, 1, 1], 'SLC': [1, 0, 0.196078431]}


def plot_trajectory(df, xPos, yPos, new_filename):
    print('Plotting trajectory...')
    mpl.rcParams['agg.path.chunksize'] = 10000
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    axes.plot(df[xPos], df[yPos])
    axes.set(xlim=(0, 950), ylim=(0, 950))
    plt.tight_layout()
    fig.savefig(new_filename, bbox_inches='tight', format='tiff')
    plt.close('all')


def plot_time_spent_moving(exp, new_filename):
    print('Plotting time spent moving...')
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
    print('Plotting distance moved...')
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


def plot_bouts_per_trials(bca, new_filename):
    print('Plotting number of bouts per trial...')
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


def plot_bouts_per_condition(bca, new_filename):
    print('Plotting number of bouts per condition...')
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


def plot_pie_bouttypes(bca, new_filename):
    print('Plotting pie chart of bout type percentages...')
    hab = bca.iloc[0, :]
    trials = bca.iloc[1::2, :].sum(axis=0)
    breaks = bca.iloc[2::2, :].sum(axis=0)
    dfs = [hab, trials, breaks]
    condition = ['Baseline', 'Trials', 'Intertrial Intervals']

    fig = plt.figure(figsize=(15, 10))
    cmp = JoaoColormap()
    gs = GridSpec(2, 3)  # 2 rows, 3 columns

    def my_autopct(pct):
        return ('%1.0f%%' % pct) if pct > 4 else ''

    for i in np.arange(3):
        ax = plt.subplot2grid((2, 3), (0, i))
        dfs[i].plot(kind='pie', colors=[colour_dict.get(x, 'white') for x in dfs[i].index], ax=ax,
                    autopct=my_autopct, pctdistance=1.1, labels=None)
        # ax.axis('equal')
        # ax.set_title(condition[i], fontsize=18)
        ax.set(ylabel='', title=condition[i], aspect='equal')
    ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    cat_sym_unique = np.arange(13)  # np.unique(cat_sym_tmp)
    for i in np.arange(13):
        tmp_str = ordered_bouts[i]
        ax4.scatter(i, 0.9, color=cmp(idx[i] - 1), s=200)
        ax4.text(i - 0.12, 0.8, tmp_str)
    ax4.set_ylim([0, 1])
    ax4.axis('off')
    gs.update(wspace=0.2, hspace=0.02)
    plt.tight_layout()
    fig.savefig(new_filename, bbox_inches='tight', format='tiff')
    plt.close('all')


def plot_IBI(dft, new_filename):
    dft['IBI'] = (dft.allboutstarts.shift(-1) - dft.allboutends) / 700
    hab = dft.groupby('Trial').get_group(0.0).replace(np.nan, 0.0)
    trials = dft.loc[dft['Trial'].isin(np.arange(1, 41, 2))].replace(np.nan, 0.0)
    breaks = dft.loc[dft['Trial'].isin(np.arange(2, 41, 2))].replace(np.nan, 0.0)

    fig, ax = plt.subplots(1, 1)
    ax = sns.distplot(hab.IBI, label='Habituation', ax=ax)
    ax = sns.distplot(trials.IBI, label='Trials', ax=ax)
    ax = sns.distplot(breaks.IBI, label='ITI', ax=ax)
    ax.set_xlabel('Length of Interbout Interval (s)', fontsize=20, labelpad=20)
    ax.set_ylabel('Frequency', fontsize=20, labelpad=20)
    lg = ax.legend(loc=1, title='Condition', fontsize='x-large')
    lg.get_title().set_fontsize('18')  # legend 'Title' fontsize
    plt.tight_layout()
    fig.savefig(new_filename, bbox_inches='tight', format='tiff')
    plt.close('all')


def plot_tailangle_time(TailAngles, trial_st, new_filename):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    sns.despine()
    colors = plt.cm.viridis(np.linspace(0,1,20))
    for i in np.arange(20):
        ax.plot(np.arange(42000), TailAngles.iloc[trial_st[i]-7000:trial_st[i]+35000,9]+i*6, c=colors[i], linewidth=0.75)
    plt.axvline(x=7000, color='r', linestyle='--')
    plt.axvline(x=28000, color='r', linestyle='--')
    plt.xticks(np.arange(0, 42000, 3500), np.arange(-10, 50, 5))
    plt.yticks(np.arange(0, 120, 6), ['Trial 1', 'Trial 2', 'Trial 3', 'Trial 4', 'Trial 5', 'Trial 6', 'Trial 7',
                                      'Trial 8', 'Trial 9', 'Trial 10', 'Trial 11', 'Trial 12', 'Trial 13', 'Trial 14',
                                      'Trial 15', 'Trial 16', 'Trial 17', 'Trial 18', 'Trial 19', 'Trial 20'])
    plt.tight_layout()
    fig.savefig(new_filename, bbox_inches='tight', format='tiff')
    plt.close('all')