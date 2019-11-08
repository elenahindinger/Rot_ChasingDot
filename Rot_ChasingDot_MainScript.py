__author__ = 'Elena Maria Daniela Hindinger'

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.io import loadmat
import itertools as it
from Rot_ChasingDot_Functions import *
from Rot_ChasingDot_Plots import *

exp_path = r'F:\Rotations\ExperimentalData\2019_10_24\20191024_experimental_Trial1_f2_Tu_4275ix_6dpf_Atlas_75_P2_chasingdot'  # CHANGE HERE TO LOOK AT A DIFFERENT FISH
save_path = r'F:\Rotations\Analysis'

#######################################################################################################################

''' Universal Bout Parameters '''
ordered_bouts = ['AS', 'Slow1', 'Slow2', 'ShortCS', 'LongCS', 'BS', 'J-Turn', 'HAT', 'RT', 'SAT', 'O-bend', 'LLC', 'SLC']
idx = [11, 7, 9, 1, 2, 3, 5, 13, 8, 12, 4, 10, 6]
numbers, unordered_bouts = (list(t) for t in zip(*sorted(zip(idx, ordered_bouts))))

colour_dict = {'AS': 'skyblue', 'Slow1': 'royalblue', 'Slow2': 'darkblue', 'ShortCS': 'grey', 'LongCS': 'k',
               'BS': 'orange', 'J-Turn': 'lightcoral', 'HAT': 'g', 'RT': 'darkgreen', 'SAT': 'mediumpurple',
               'O-bend': 'mediumvioletred', 'LLC': 'yellow', 'SLC': 'red'}

#######################################################################################################################

''' Reading files '''
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
camlog, stimlog = global_trimming(camlog_og, stimlog_og)  # trim before and after synch of camlog and stimlog
boutmat = loadmat(boutfile)  # load bout mat file

''' Some transformations in camlog and stimlog '''
setup = 'atlas' if 'atlas' in filename.lower() else 'c3po'
stimlog = stim_shader_to_camera_space(dataframe=stimlog, setup=setup)  # transform stimlog to camera space
stimlog = img_to_cart(dataframe=stimlog, xPos_og='xPosDotCamSpace', yPos_og='yPosDotCamSpace', xPos_new='xPosCartDot',
                      yPos_new='yPosCartDot')  # from camera coordinates to cartesian
camlog = img_to_cart(dataframe=camlog, xPos_og='xPos', yPos_og='yPos', xPos_new='xPosCart', yPos_new='yPosCart')

''' Dealing with boutmat '''
Bouts = return_as_df(boutmat, keys=['allboutstarts', 'allboutends', 'indRealEnds', 'rejectedBouts'])  # bout start, end, classified Y/N
BoutCat = return_as_df(boutmat, keys=['boutCat'])  # bout category
DistToCenter = return_as_df(boutmat, keys=['distToCenter'])  # distance to centre of cloud
BodyAngles = return_as_df(boutmat, keys=['realBodyAngles'])  # body angles unwrapped
BoutSegMeth = return_as_df(boutmat, keys=['smootherTailCurveMeasure'])  # method of bout segmentation
# KinPar = return_as_df(boutmat, keys=['BoutKinematicParameters']) NOT DEALT WITH YET, NEXT VERSION
# NewScore = return_as_df(boutmat, keys=['newScore']) NOT DEALT WITH YET, NEXT VERSION
# TailAngles = return_as_df(boutmat, keys=['smoothedCumsumInterpFixedSegmentAngles']) NOT DEALT WITH YET, NEXT VERSION

# clean up Bouts to delete unclassified bouts
UnclassifiedBoutIndices = Bouts[Bouts['rejectedBouts'] == 1].index  # Get indices for which column rejected Bouts is 1
Bouts = Bouts.drop(UnclassifiedBoutIndices).reset_index().drop(['rejectedBouts', 'index'], axis=1)  # Delete these rows


''' Merge bout start, bout end, bout type with stimulus '''
BoutsWithCat = pd.concat([Bouts.iloc[:, :2], BoutCat], axis=1)  # merge bouts with classification
BoutsWithCat['Id'] = BoutsWithCat.allboutstarts  # create a dummy column Id to merge with stim afterwards
df = pd.merge(BoutsWithCat, stimlog, on='Id')  # merge BoutsWithCat with stimlog
dft = add_trial_number(df)  # this function returns a new dataframe with trial number added, where 0 is habituation. You can do a lot of computation from this point onwards.


''' Mapping bouts onto camlog '''
tail_cat = bouts_to_camlog(dft, camlog)  # creates an array of bouts that is true for time and duration
camlog['boutCat'] = tail_cat  # adds this array to camera log
exp = camlog.merge(stimlog, on='Id', how='left')  # merge camera log and stimlog
exp[['xPosCartDot', 'yPosCartDot']] = exp[['xPosCartDot', 'yPosCartDot']].fillna(method='ffill')  # forward fill to pad out all rows with a dot position
exp['StopWatchTime'] = exp['StopWatchTime'].interpolate()  # interpolate time
exp = add_trial_number(exp)  # this function returns a new dataframe with trial number added.

#######################################################################################################################

''' PLOT 1 Trajectory '''
plot_trajectory(camlog, 'xPosCart', 'yPosCart', new_filename=os.path.join(save_path, (filename + '_trajectory.tiff')))

''' PLOT 2 + 3 Time Spent Moving and Distance Moved'''
plot_time_spent_moving(exp, new_filename=os.path.join(save_path, (filename + '_TimeSpentMoving.tiff')))
# tsm = tsm.to_frame()
# tsm.columns = ['TimeSpentMoving']
# tsm['TrialNum'] = np.concatenate([np.array([0]), np.repeat(np.arange(1, 21), 2)])
# tsm['Condition'] = ['Habituation'] + list(it.chain.from_iterable(zip( ['Trial']*20, ['ITI']*20)))
# sns.barplot(x='TrialNum', y='TimeSpentMoving', hue='Condition', data=tsm)

plot_distance(exp, new_filename=os.path.join(save_path, (filename + '_Distance.tiff')))


''' PLOT 4 Calculate number of bouts per category per stimulus condition '''
bout_count_temp = dft.groupby('Trial')['boutCat'].value_counts().unstack().fillna(0).rename_axis(index=None, columns=None)
bca = pd.DataFrame(np.zeros(shape=(41,13)), columns=np.arange(1, 14))  # create empty dataframe with all bout categories
bca.update(bout_count_temp)  # map actual bouts observed onto empty dataframe, this helps with plotting
bca.columns = unordered_bouts  # rename columns to bout type names
bca = bca[ordered_bouts]  # reorder dataframe to plot in correct order
# bca_norm = bca.div(bca.sum(axis=1), axis=0)*100

plot_bouts_per_trials(bca, new_filename=os.path.join(save_path, (filename + '_boutfreq.tiff')), colour_dict=colour_dict)


''' PLOT 5 '''
plot_bouts_per_condition(bca, new_filename=os.path.join(save_path, (filename + '_total_bout_count.tiff')), colour_dict=colour_dict)




''' Interbout interval '''

# May become interesting for simpler plots
dft['IBI'] = dft.allboutstarts.shift(-1) - dft.allboutends
ibi_stats = dft.groupby('Trial')['IBI'].describe()
ibi_stats_all = pd.DataFrame(np.zeros(shape=(41,8)), columns=ibi_stats.columns)  # create empty dataframe with all bout categories
ibi_stats_all.update(ibi_stats)  # map actual bouts observed onto empty dataframe, this helps with plotting
ibi_stats_all.iloc[:, 1:] = ibi_stats_all.iloc[:, 1:] / 700  # converts from frames to seconds

# normalisation
ibi_stats_all.iloc[0 , :] = ibi_stats_all.iloc[0, :] / 600.0
ibi_stats_all.iloc[1::2,:] = ibi_stats_all.iloc[1::2,:] / 30.0
ibi_stats_all.iloc[2::2,:] = ibi_stats_all.iloc[2::2,:] / 120.0


fig, ax = plt.subplots(1, 1)
ax = sns.barplot(x='index', y="mean", data=ibi_stats_all.reset_index(), ax=ax)
plt.xticks(np.arange(1, 41, 2), np.arange(1, 21))
ax.set_xlabel('Trial Number', fontsize=20, labelpad=20)
ax.set_ylabel('Mean Interbout Interval (seconds)', fontsize=20, labelpad=20)

''' IBI per hab, trial, break distribution plots '''
dft['IBI'] = dft.IBI / 700
hab = dft.groupby('Trial').get_group(0.0)
trials = dft.loc[dft['Trial'].isin(np.arange(1, 41, 2))]
breaks = dft.loc[dft['Trial'].isin(np.arange(2, 41, 2))]
breaks = breaks.replace(np.nan, 0.0)

sns.distplot(hab.IBI, bins=40, rug=True)
sns.distplot(trials.IBI, bins=40, rug=True)
sns.distplot(breaks.IBI, bins=40, rug=True)

plt.hist(hab.IBI, bins=40)






