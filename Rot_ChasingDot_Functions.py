__author__ = 'Elena Maria Daniela Hindinger'

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import itertools as it
from scipy.io import loadmat
from math import sqrt


def JoaoColormap():
    clrs = np.array([
        [0.392156863, 0.392156863, 0.392156863],  #1
        [0, 0, 0], #2
        [1, 0.666666667, 0], #3
        [0.862745098, 0, 0.862745098], #4
        [0.980392157, 0.501960784, 0.447058824], #5
        [1, 0, 0.196078431], #6
        [0, 0.588235294, 1], #7
        [0, 0.6, 0], #8
        [0, 0, 0.784313725], #9
        [1, 1, 0], #10
        [0.4, 1, 1], #11
        [0.576470588, 0.439215686, 0.858823529], #12
        [0.411764706, 1, 0.4]]) #13

    return mpl.colors.ListedColormap(clrs)


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


def trim_start(camlog_og, stimlog_og):
    ''' Get rid of all NaNs in datasets '''
    camlog = camlog_og.dropna()
    stimlog = stimlog_og.dropna(subset=['Id'])
    ''' Trim beginning to synchronise starts based on Id '''
    st_id = np.max([stimlog.loc[0, 'Id'], camlog.loc[0, 'Id']])
    camlog2 = camlog[(camlog.Id == st_id).idxmax():].reset_index().drop('index', axis=1).copy()
    stimlog2 = stimlog[(stimlog.Id == st_id).idxmax():].reset_index().drop('index', axis=1).copy()
    return camlog2, stimlog2, (camlog.Id == st_id).idxmax()


# def trim_end(camlog_st, stimlog_st):
#     ''' OLD !!! Trim end to synchronise ends based on Id of Stim '''
#     stimlog_ed_temp = stimlog_st.iloc[:216000, :]
#     camlog_ed = camlog_st.iloc[:2519991, :]
#     camlog_ed_id = camlog_ed.iloc[-1, 0]+10
#     stimlog_ed = stimlog_ed_temp[:(stimlog_ed_temp.Id == camlog_ed_id).idxmax()]
#     return camlog_ed, stimlog_ed


def trim_end(camlog_st, stimlog_st):
    ''' Trim end to synchronise ends based on Id of Stim '''
    stimlog_ed_temp = stimlog_st.iloc[:216000, :].copy()
    stimlog_ed_temp['diff'] = np.abs(stimlog_ed_temp['Id'].diff())  # find where dot position changes from -100 (not shown) to displayed
    df_filtered = stimlog_ed_temp[stimlog_ed_temp['diff'] == 0.0]  # returns rows of condition changes
    stimlog_ed_id = df_filtered.Id.unique()[-1]
    stimlog_ed = stimlog_st[:(stimlog_st.Id == stimlog_ed_id).idxmax()].copy()
    camlog_ed = camlog_st[:(camlog_st.Id == stimlog_ed_id-9).idxmax()].copy()
    return camlog_ed, stimlog_ed, (camlog_st.Id == stimlog_ed_id-9).idxmax()


def return_as_df(boutmat, keys):
    ''' Boutmat is the original dictionary, keys is a list of keys you want to have included in the new dataframe. '''
    dict2 = {x: boutmat[x] for x in keys}
    for key, val in dict2.items():
        dict2[key] = val.flatten()
    df = pd.DataFrame.from_dict(dict2)
    return df


def return_as_df_multiple(boutmat, key):
    ''' Boutmat is the original dictionary, key contains multiple sub arrays. '''
    dict2 = {x: boutmat[x] for x in [key]}
    df = pd.DataFrame(dict2.get(key, ''))
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
    df_temp = df.copy()
    df_temp['diff'] = np.abs(df_temp['xPosCartDot'].diff())  # find where dot position changes from -100 (not shown) to displayed
    df_filtered = df_temp[df_temp['diff'] > 10000.0]  # returns rows of condition changes
    trials = pd.DataFrame(np.concatenate((np.array([0]), df_filtered.index.values)), columns=['index'])  # creates temp dataframe with indices of condition changes
    trials['Trial'] = np.arange(len(trials))  # adds column with trial number
    trials.set_index('index', inplace=True)  # sets this as index to merge with df later
    df_trials = df.join(trials)  # merges trial temp with dataframe
    df_trials.Trial = df_trials.Trial.fillna(method='pad')  # fills in missing conditions by forward fill
    return df_trials, trials


def bouts_to_camlog(df, camlog):
    ''' Returns an array containing bout types true with time that can then be added to camlog. '''
    id_st = df['allboutstarts'].values.astype(int) - int(camlog.loc[0, 'Id'])
    id_ed = df['allboutends'].values.astype(int) - int(camlog.loc[0, 'Id'])
    tail_cat = np.zeros(camlog.shape[0]) - 1
    for i, val in enumerate(id_st):
        tail_cat[id_st[i]:id_ed[i]] = df['boutCat'][i]
    return tail_cat


def calc_dist(x1, x2, y1, y2):
    dist = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
    return dist


def distance(df, px_per_mm=75):
    temp = pd.concat([df.xPosCart, df.xPosCart.shift(fill_value=df.loc[0, 'xPosCart']), df.yPosCart,
                     df.yPosCart.shift(fill_value=df.loc[0, 'yPosCart'])], axis=1)
    temp.columns = ['xPos', 'xPos2', 'yPos', 'yPos2']
    dist_px = temp.apply(lambda row: calc_dist(row[0], row[1], row[2], row[3]), axis=1)
    dist_mm = dist_px * px_per_mm / 1000
    return dist_mm


def my_autopct(pct):
    return ('%1.0f%%' % pct) if pct >= 3 else ''


def fix_BodyAngle(df):
    ''' Calculates body angle. '''
    # original values range from 0 to 3/2pi and 0 to -pi/2, the following line converts every angle from 0-2pi
    df['BodyAngleCont'] = df.BodyAngle.apply(lambda x: (2 * np.pi - np.abs(x)) if (-np.pi / 2) < x < 0 else x)
    # need to unwrap angles as we care about relative angular change, not absolute angle
    df['BodyAngleUnwrap'] = np.unwrap(df.BodyAngleCont, discont=0)
    return df['BodyAngleUnwrap'].values

