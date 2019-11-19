__author__ = 'Elena Maria Daniela Hindinger'

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import itertools as it
from scipy.io import loadmat
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from Rot_ChasingDot_Functions import *


''' Universal Bout Parameters '''
ordered_bouts = ['AS', 'Slow1', 'Slow2', 'ShortCS', 'LongCS', 'BS', 'J-Turn', 'HAT', 'RT', 'SAT', 'O-bend', 'LLC', 'SLC']
idx = [11, 7, 9, 1, 2, 3, 5, 13, 8, 12, 4, 10, 6]
numbers, unordered_bouts = (list(t) for t in zip(*sorted(zip(idx, ordered_bouts))))
cmpW = JoaoColormap()
colour_dict = {'AS': [0.4, 1, 1], 'Slow1': [0, 0.588235294, 1], 'Slow2': [0, 0, 0.784313725],
               'ShortCS': [0.392156863, 0.392156863, 0.392156863], 'LongCS': [0, 0, 0], 'BS': [1, 0.666666667, 0],
               'J-Turn': [0.980392157, 0.501960784, 0.447058824], 'HAT': [0.411764706, 1, 0.4], 'RT': [0, 0.6, 0],
               'SAT': [0.576470588, 0.439215686, 0.858823529], 'O-bend': [0.862745098, 0, 0.862745098],
               'LLC': [1, 1, 0], 'SLC': [1, 0, 0.196078431]}


def plot_trajectory(df, xPos, yPos, new_filename):
    print('Plotting trajectory...')
    mpl.rcParams['agg.path.chunksize'] = 10000
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    axes.plot(df[xPos], df[yPos], color='k')
    axes.set(xlim=(0, 950), ylim=(0, 950))
    plt.tight_layout()
    fig.savefig(new_filename, bbox_inches='tight', format='tiff')
    plt.close('all')


def plot_trajectory_split(exp, new_filename):
    g = exp.groupby('Trial')
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    counter = 0
    custom_lines = [Line2D([0], [0], color='k', lw=3),
                    Line2D([0], [0], color='firebrick', lw=3),
                    Line2D([0], [0], color='darkorange', lw=3)]
    for name, group in g:
        if name == 0:
            axes.plot(group['xPosCart'], group['yPosCart'], c='k', label='Baseline')
        elif name % 2 != 0:
            axes.plot(group['xPosCart'], group['yPosCart'], c='firebrick', label='Trial')  # trials
            counter += 1
        else:
            axes.plot(group['xPosCart'], group['yPosCart'], c='darkorange', alpha=0.3, label='Intertrial Interval')  # intertrial intervals
            counter += 1
    axes.set(xlim=(0, 950), ylim=(0, 950))
    lg = axes.legend(custom_lines, ['Baseline', 'Trial', 'Intertrial Interval'], loc=1, fontsize='x-large')
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


def plot_IBI(df, new_filename):
    df['IBI'] = (df.allboutstarts.shift(-1) - df.allboutends) / 700
    hab = df.groupby('Trial').get_group(0.0).replace(np.nan, 0.0)
    trials = df.loc[df['Trial'].isin(np.arange(1, 41, 2))].replace(np.nan, 0.0)
    breaks = df.loc[df['Trial'].isin(np.arange(2, 41, 2))].replace(np.nan, 0.0)

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
    fig, ax = plt.subplots(22, 1, sharex=True, sharey=True, figsize=(15, 10))
    sns.despine(left=True)
    colors = plt.cm.viridis(np.linspace(0, 1, 20))
    # Baseline
    x = np.arange(42000)
    yb = TailAngles.iloc[int(trial_st[0]/2)-7000:int(trial_st[0]/2)+35000, 9].values
    ax[0].plot(x, yb, c='k', linewidth=0.75)
    ax[0].annotate(xy=(x[0], yb[0]), xytext=(-50.0, 0), textcoords='offset points', s=('Baseline'), va='center')
    ax[0].axis('off')
    # Empty second
    ax[1].axis('off')
    # Traces trials 1 - 20
    for i in np.arange(2, 22):
        y = TailAngles.iloc[trial_st[i-2]-7000:trial_st[i-2]+35000, 9].values
        ax[i].plot(x, y, c=colors[i-2], linewidth=0.75)
        ax[i].axvline(x=7000, color='r', linestyle='--')
        ax[i].axvline(x=28000, color='r', linestyle='--')
        ax[i].annotate(xy=(x[0], y[0]), xytext=(-50.0, 0), textcoords='offset points', s=('Trial %s' % str(i-1)),
                       va='center')
        if i != 21:
            ax[i].axis('off')
    ax[21].spines['bottom'].set_bounds(0, 42000)
    plt.xticks(np.arange(0, 42000, 3500), np.arange(-10, 50, 5))
    plt.yticks([])
    plt.xlabel('Time (seconds)', fontsize=14, labelpad=10)
    plt.subplots_adjust(hspace=0.02)
    fig.savefig(new_filename, bbox_inches='tight', format='tiff')
    plt.close('all')


def plot_boutmap(exp, trial_st, cmpW, new_filename):
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(22, 5)  # 22 rows, 2 columns
    # Baseline
    ax0 = plt.subplot2grid((22, 5), (0, 0), colspan=4)
    ax0.imshow(exp.iloc[int(trial_st[0]/2)-7000:int(trial_st[0]/2)+35000, 38].values.reshape((1, -1)), cmap=cmpW,
                 aspect='auto', vmin=1, vmax=13)
    sns.despine(bottom=True, ax=ax0)
    ax0.set_xticks([])
    ax0.set_yticks(np.arange(1))
    ax0.set_yticklabels(['Baseline'])
    # Empty second
    ax1 = plt.subplot2grid((22, 5), (1, 0), colspan=4)
    ax1.axis('off')
    # Traces trials 1 - 20
    for i in np.arange(2, 22):
        ax = plt.subplot2grid((22, 5), (i, 0), colspan=4)
        ax.imshow(exp.iloc[trial_st[i-2]-7000:trial_st[i-2]+35000, 38].values.reshape((1, -1)), cmap=cmpW,
                  aspect='auto', vmin=1, vmax=13)
        ax.axvline(x=7000, color='crimson', linestyle='--')
        ax.axvline(x=28000, color='crimson', linestyle='--')
        ax.axvspan(xmin=7000, xmax=28000, color='crimson', alpha=0.1)
        ax.set_xticks([])
        ax.set_yticks(np.arange(1))
        ax.set_yticklabels(['Trial %s' % int(i-1)])
        if i == 21:
            sns.despine(right=True, top=True, ax=ax)
            ax.spines['bottom'].set_bounds(0, 42000)
        else:
            sns.despine(bottom=True, top=True, right=True, ax=ax)
    plt.xticks(np.arange(0, 42000, 3500), np.arange(-10, 50, 5))
    plt.xlabel('Time (seconds)', fontsize=14, labelpad=10)
    # Legend
    axlg = plt.subplot2grid((22, 5), (0, 4), rowspan=22)
    for i in np.arange(13):
        tmp_str = ordered_bouts[i]
        axlg.scatter(0.2, i/13+0.04, color=cmpW(idx[i] - 1), s=200)
        axlg.text(0.3, i/13+0.05, tmp_str)
    axlg.set_xlim([0, 0.5])
    axlg.set_ylim([1, 0])
    axlg.axis('off')
    gs.update(wspace=0., hspace=0.2)
    # saving figure
    fig.savefig(new_filename, bbox_inches='tight', format='tiff')
    plt.close('all')

