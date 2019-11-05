__author__ = 'Elena Maria Daniela Hindinger'

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.io import loadmat
from Rot_ChasingDot_Functions import *

ordered_bouts = ['AS', 'Slow1', 'Slow2', 'ShortCS', 'LongCS', 'BS', 'J-Turn', 'HAT', 'RT', 'SAT', 'O-bend', 'LLC', 'SLC']
unordered_bouts = ['ShortCS', 'LongCS', 'BS', 'O-bend', 'J-Turn', 'SLC', 'Slow1', 'RT', 'Slow2', 'LLC', 'AS', 'SAT', 'HAT']
# idx = [11  7  9  1  2  3  5 13  8 12  4 10 6];
colour_dict = {'AS': 'skyblue', 'Slow1': 'royalblue', 'Slow2': 'darkblue', 'ShortCS': 'grey', 'LongCS': 'k',
               'BS': 'orange', 'J-Turn': 'lightcoral', 'HAT': 'g', 'RT': 'darkgreen', 'SAT': 'mediumpurple',
               'O-bend': 'mediumvioletred', 'LLC': 'yellow', 'SLC': 'red'}

''' Reading files '''
exp_path = r'F:\Rotations\ExperimentalData\2019_10_24\20191024_experimental_Trial1_f2_Tu_4275ix_6dpf_Atlas_75_P2_chasingdot'  # CHANGE HERE TO LOOK AT A DIFFERENT FISH

print('Reading files...')
camfile, stimfile, boutfile, filename = find_files(exp_path=exp_path)  # finds correct file names based on experiment folder given above
camlog_og = pd.DataFrame(loadmat(camfile)['A'], columns=['Id', 'xPos', 'yPos', 'xBodyVector', 'yBodyVector', 'BodyAngle',
                                                      'MaxValue', 'BlobValue', 'midEyeX', 'midEyeY', 'hour', 'minute',
                                                      'StimuliIncr', 'CumSumTail', 'TailValues1', 'TailValues2',
                                                      'TailValues3', 'TailValues4', 'TailValues5', 'TailValues6',
                                                      'TailValues7', 'TailValues8', 'TailValues9', 'TailValues10',
                                                      'TailAngles1', 'TailAngles2', 'TailAngles3', 'TailAngles4',
                                                      'TailAngles5', 'TailAngles6', 'TailAngles7', 'TailAngles8',
                                                      'TailAngles9', 'TailAngles10', 'FirstorNot', 'TrackEye',
                                                      'TimerAcq', 'lag'])  # read camlog
stimlog_og = read_stimlog(stimfile, shortened=True)  # read stimlog
camlog, stimlog = global_trimming(camlog_og, stimlog_og)
boutmat = loadmat(boutfile)  # load bout mat file

''' Dealing with boutmat '''
Bouts = return_as_df(boutmat, keys=['allboutstarts', 'allboutends', 'rejectedBouts'])  # bout start, end, classified Y/N
RealEnds = return_as_df(boutmat, keys=['indRealEnds'])  # actual end of bout as frame Id
BoutCat = return_as_df(boutmat, keys=['boutCat'])  # bout category
DistToCenter = return_as_df(boutmat, keys=['distToCenter'])  # distance to centre of cloud
BodyAngles = return_as_df(boutmat, keys=['realBodyAngles'])  # body angles unwrapped
BoutSegMeth = return_as_df(boutmat, keys=['smootherTailCurveMeasure'])  # method of bout segmentation
# KinPar = return_as_df(boutmat, keys=['BoutKinematicParameters']) NOT DEALT WITH YET, NEXT VERSION
# NewScore = return_as_df(boutmat, keys=['newScore'])
# TailAngles = return_as_df(boutmat, keys=['smoothedCumsumInterpFixedSegmentAngles'])

# clean up Bouts to delete unclassified bouts
UnclassifiedBoutIndices = Bouts[Bouts['rejectedBouts'] == 1].index  # Get indices for which column rejected Bouts is 1
Bouts = Bouts.drop(UnclassifiedBoutIndices).reset_index().drop(['rejectedBouts', 'index'], axis=1)  # Delete these rows

''' Merge bout start, bout end, bout type with stimulus '''

BoutsWithCat = pd.concat([Bouts, BoutCat], axis=1)  # merge bouts with classification
BoutsWithCat['Id'] = BoutsWithCat.allboutstarts  # create a dummy column Id to merge with stim afterwards

df = pd.merge(BoutsWithCat, stimlog, on='Id')  # merge BoutsWithCat with stimlog

dft = add_trial_number(df)  # this function returns a new dataframe with trial number added, where 0 is habituation. You can do a lot of computation from this point onwards.

# ''' Interbout interval '''
# df_trials['IBI'] = df_trials.allboutstarts.shift(-1) - df_trials.allboutends
# bout_count_temp = df_trials.groupby('Trial')['IBI'].value_counts().unstack().fillna(0).rename_axis(index=None, columns=None)

''' Calculate number of bouts per category per stimulus condition '''

bout_count_temp = dft.groupby('Trial')['boutCat'].value_counts().unstack().fillna(0).rename_axis(index=None, columns=None)

bout_count_all = pd.DataFrame(np.zeros(shape=(41,13)), columns=np.arange(1, 14))  # create empty dataframe with all bout categories
bout_count_all.update(bout_count_temp)  # map actual bouts observed onto empty dataframe, this helps with plotting
bout_count_all.columns = unordered_bouts  # rename columns to bout type names
bout_count_all = bout_count_all[ordered_bouts]

bout_count_all.plot(kind='bar', stacked=True, color=[colour_dict.get(x, 'white') for x in bout_count_all.columns])

''' Mapping bouts onto camlog '''

# code from Adrien
# Create Vector same size as recording containing tail active and categories:
# STILL NEEDS IMPROVEMENT
id_st = dft['allboutstarts'].values
id_ed = dft['allboutends'].values
tail_st = np.zeros(camlog.shape[0])*np.nan
tail_ed = np.zeros(camlog.shape[0])*np.nan
tail_cat = np.zeros(camlog.shape[0])-1

for i, val in enumerate(id_st):
    tail_st[val] = i
    tail_cat[id_st[i]:id_ed[i]] = dft['boutCat'][i]
    tail_ed[val] = i

camlog['tail_st'] = tail_st
camlog['tail_ed'] = tail_ed
camlog['tail_cat'] = tail_cat
