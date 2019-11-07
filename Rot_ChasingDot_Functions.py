__author__ = 'Elena Maria Daniela Hindinger'

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import itertools as it
from scipy.io import loadmat


def find_files(exp_path):
    for file in os.listdir(exp_path):
        if file.endswith('000.mat'):
            cam_file = os.path.join(exp_path, file)
            filename = file[:-7]
        elif file.startswith('stimlog') and file.endswith('.mat'):
            stim_file = os.path.join(exp_path, file)
        elif file == 'tempBoutX2.mat':
            bout_file = os.path.join(exp_path, file)
        else:
            pass
    return cam_file, stim_file, bout_file, filename


def read_stimlog(stimfile, shortened=False):
    stimlog_dict = loadmat(stimfile)['StimLog']  # load mat file as numpy array
    n = []
    for i in np.arange(6):
        n.append(stimlog_dict[0][0][2][0][i][0][0][1].flatten())  # we need to find the arrays that correspond to the actual values, flatten them, add them to a list
    stimlog_df = pd.DataFrame(n).T  # creates dataframe from our list of values, transposes it
    stimlog_df.columns = np.concatenate(stimlog_dict[0][0][1][0]).tolist()  # take array of variable names and assign it to dataframe columns
    stimlog_df.rename(columns={'frameID' : 'Id', 'xPos' : 'xPosDot', 'yPos' : 'yPosDot'}, inplace=True)  # rename columns to prep for merging
    if shortened == True:
        stimlog_df = stimlog_df.drop(['iGlobalTime', 'iTimeDelta'], axis=1)  # we don't need these columns anymore
    return stimlog_df


def global_trimming(camlog, stimlog):
    ''' Get rid of all NaNs in datasets '''
    camlog = camlog.dropna()
    stimlog = stimlog.dropna(subset=['Id'])
    ''' Trim beginning to synchronise starts based on Id '''
    stim_st_id = stimlog.loc[0, 'Id']
    cam_st_id = camlog.loc[0, 'Id']
    if stim_st_id >= cam_st_id:
        # trim camlog
        camlog = camlog[(camlog.Id == stim_st_id).idxmax():].reset_index().drop('index', axis=1)
    else:
        # trim stimlog
        stimlog = stimlog[(stimlog.Id == cam_st_id).idxmax():].reset_index().drop('index', axis=1)
    ''' Trim end to synchronise ends based on Id of Stim '''
    cam_ed_id = round(camlog.iloc[-1, 0], -1)
    camlog = camlog[:(camlog.Id == cam_ed_id).idxmax()]
    stimlog = stimlog[:(stimlog.Id == cam_ed_id).idxmax()]
    return camlog, stimlog


def return_as_df(boutmat, keys):
    ''' Boutmat is the original dictionary, keys is a list of keys you want to have included in the new dataframe. '''
    dict2 = {x: boutmat[x] for x in keys}
    for key, val in dict2.items():
        dict2[key] = val.flatten()
    df = pd.DataFrame.from_dict(dict2)
    return df


def img_to_cart(dataframe, xPos_og, yPos_og, xPos_new, yPos_new):
    df = dataframe.copy()
    df[xPos_new] = df[yPos_og]  # transform X image to X cartesian
    df[yPos_new] = 948 - df[xPos_og]  # transform Y image to Y cartesian
    df = df.drop([xPos_og, yPos_og], axis=1)
    return df


def stim_shader_to_camera_space(dataframe, setup):
    df = dataframe.copy()
    if setup == 'atlas':
        M = np.array([489.8566, 0.5522, -0.5522, 489.8566]).reshape(2, 2)
        C = np.array([474.6911, 473.1050])
    elif setup == 'c3po':
        M = np.array([486.1282, 2.5560, -2.5560, 486.1282]).reshape(2, 2)
        C = np.array([469.7695, 468.0829])
    else:
        print('Please indicate which set-up this experiment was recorded on.')

    all_dot_pos =np.stack((df.xPosDot.values, df.yPosDot.values), axis=-1)
    r = np.dot(all_dot_pos, M) + C
    df['xPosDotCamSpace'] = r[:, 0::2].flatten()
    df['yPosDotCamSpace'] = 948.0-(r[:, 1::2].flatten())
    df = df.drop(['xPosDot', 'yPosDot'], axis=1)
    return df


def add_trial_number(df):
    df['diff'] = np.abs(df['xPosCartDot'].diff())  # find where dot position changes from -100 (not shown) to displayed
    df_filtered = df[df['diff'] > 10000.0]  # returns rows of condition changes
    trials = pd.DataFrame(np.concatenate((np.array([0]), df_filtered.index.values)), columns=['index'])  # creates temp dataframe with indices of condition changes
    trials['Trial'] = np.arange(len(trials))  # adds column with trial number
    trials.set_index('index', inplace=True)  # sets this as index to merge with df later
    df_trials = df.join(trials).drop(['diff'], axis=1)  # merges trial temp with dataframe
    df_trials.Trial = df_trials.Trial.fillna(method='pad')  # fills in missing conditions by forward fill
    return df_trials


def bouts_to_camlog(dft, camlog):
    ''' Returns an array containing bout types true with time that can then be added to camlog '''
    id_st = dft['allboutstarts'].values - int(camlog.iloc[0, 0])
    id_ed = dft['allboutends'].values - int(camlog.iloc[0, 0])
    tail_cat = np.zeros(camlog.shape[0]) - 1
    for i, val in enumerate(id_st):
        tail_cat[id_st[i]:id_ed[i]] = dft['boutCat'][i]
    return tail_cat


def plot_trajectory(df, xPos, yPos, new_filename):
    mpl.rcParams['agg.path.chunksize'] = 10000
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    axes.plot(df[xPos], df[yPos])
    axes.set(xlim=(0, 950), ylim=(0, 950))
    plt.tight_layout()
    fig.savefig(new_filename, bbox_inches='tight', format='tiff')
    plt.close('all')
