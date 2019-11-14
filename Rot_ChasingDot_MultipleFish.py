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


''' Universal Bout Parameters '''
ordered_bouts = ['AS', 'Slow1', 'Slow2', 'ShortCS', 'LongCS', 'BS', 'J-Turn', 'HAT', 'RT', 'SAT', 'O-bend', 'LLC', 'SLC']
idx = [11, 7, 9, 1, 2, 3, 5, 13, 8, 12, 4, 10, 6]
numbers, unordered_bouts = (list(t) for t in zip(*sorted(zip(idx, ordered_bouts))))

colour_dict = {'AS': [0.4, 1, 1], 'Slow1': [0, 0.588235294, 1], 'Slow2': [0, 0, 0.784313725],
               'ShortCS': [0.392156863, 0.392156863, 0.392156863], 'LongCS': [0, 0, 0], 'BS': [1, 0.666666667, 0],
               'J-Turn': [0.980392157, 0.501960784, 0.447058824], 'HAT': [0.411764706, 1, 0.4], 'RT': [0, 0.6, 0],
               'SAT': [0.576470588, 0.439215686, 0.858823529], 'O-bend': [0.862745098, 0, 0.862745098],
               'LLC': [0.4, 1, 1], 'SLC': [1, 0, 0.196078431]}


#######################################################################################################################
big_folder = r'F:\Rotations\ExperimentalData\toanalyse'
save_path = r'F:\Rotations\Analysis'

for date_folder in os.listdir(big_folder):
    date_folder_path = os.path.join(big_folder, date_folder)
    for exp_subfolder in os.listdir(date_folder_path):
        exp_path = os.path.join(date_folder_path, exp_subfolder)
        print('Analysing %s' % exp_subfolder)
        # exp_path = r'F:\Rotations\ExperimentalData\2019_10_24\20191024_experimental_Trial1_f2_Tu_4275ix_6dpf_Atlas_75_P2_chasingdot'  # CHANGE HERE TO LOOK AT A DIFFERENT FISH

        #######################################################################################################################

        ''' Reading files '''
        print('Reading files...')
        camfile, stimfile, boutfile, filename = find_files(exp_path=exp_path)  # finds correct file names based on experiment folder given above
        camlog_og = pd.DataFrame(loadmat(camfile)['A'],
                                 columns=['Id', 'xPos', 'yPos', 'xBodyVector', 'yBodyVector', 'BodyAngle',
                                          'MaxValue', 'BlobValue', 'midEyeX', 'midEyeY', 'hour', 'minute',
                                          'StimuliIncr', 'CumSumTail', 'TailValues1', 'TailValues2',
                                          'TailValues3', 'TailValues4', 'TailValues5', 'TailValues6',
                                          'TailValues7', 'TailValues8', 'TailValues9', 'TailValues10',
                                          'TailAngles1', 'TailAngles2', 'TailAngles3', 'TailAngles4',
                                          'TailAngles5', 'TailAngles6', 'TailAngles7', 'TailAngles8',
                                          'TailAngles9', 'TailAngles10', 'FirstorNot', 'TrackEye',
                                          'TimerAcq', 'lag'])  # read camlog
        stimlog_og = read_stimlog(stimfile, shortened=True)  # read stimlog
        boutmat = loadmat(boutfile)  # load bout mat file

        ''' Some transformations in camlog and stimlog '''
        camlog_st, stimlog_st, cam_st_id = trim_start(camlog_og, stimlog_og)  # trim before and after synch of camlog and stimlog
        camlog_ed, stimlog_ed, cam_ed_id = trim_end(camlog_st, stimlog_st)
        setup = 'atlas' if 'atlas' in filename.lower() else 'c3po'
        stimlog_camspace = stim_shader_to_camera_space(dataframe=stimlog_ed,
                                                       setup=setup)  # transform stimlog to camera space
        stimlog = img_to_cart(dataframe=stimlog_camspace, xPos_og='xPosDotCamSpace', yPos_og='yPosDotCamSpace',
                              xPos_new='xPosCartDot',
                              yPos_new='yPosCartDot')  # from camera coordinates to cartesian
        camlog = img_to_cart(dataframe=camlog_ed, xPos_og='xPos', yPos_og='yPos', xPos_new='xPosCart',
                             yPos_new='yPosCart')

        ''' Dealing with boutmat '''
        Bouts = return_as_df(boutmat, keys=['allboutstarts', 'allboutends', 'indRealEnds',
                                            'rejectedBouts'])  # bout start, end, classified Y/N
        BoutCat = return_as_df(boutmat, keys=['boutCat'])  # bout category
        DistToCenter = return_as_df(boutmat, keys=['distToCenter'])  # distance to centre of cloud
        BodyAngles = return_as_df(boutmat, keys=['realBodyAngles'])  # body angles unwrapped
        BoutSegMeth = return_as_df(boutmat, keys=['smootherTailCurveMeasure'])  # method of bout segmentation
        KinPar = return_as_df_multiple(boutmat, key='BoutKinematicParameters')
        NewScore = return_as_df_multiple(boutmat, key='newScore')
        TailAngles = return_as_df_multiple(boutmat, key='smoothedCumsumInterpFixedSegmentAngles')
        TailAnglesTrim = TailAngles.iloc[cam_st_id:cam_ed_id + cam_st_id, :].copy()

        # clean up Bouts to delete unclassified bouts
        UnclassifiedBoutIndices = Bouts[Bouts['rejectedBouts'] == 1].index  # Get indices for which column rejected Bouts is 1
        Bouts = Bouts.drop(UnclassifiedBoutIndices).reset_index().drop(['rejectedBouts', 'index'],
                                                                       axis=1)  # Delete these rows

        ''' Merge bout start, bout end, bout type with stimulus '''
        BoutsWithCat = pd.concat([Bouts.iloc[:, :2], BoutCat], axis=1)  # merge bouts with classification
        BoutsWithCat['Id'] = BoutsWithCat.allboutstarts  # create a dummy column Id to merge with stim afterwards
        df = pd.merge(BoutsWithCat, stimlog, on='Id')  # merge BoutsWithCat with stimlog
        dft, trials_dft = add_trial_number(df)  # this function returns a new dataframe with trial number added, where 0 is habituation. You can do a lot of computation from this point onwards.

        ''' Mapping bouts onto camlog '''
        tail_cat = bouts_to_camlog(dft, camlog)  # creates an array of bouts that is true for time and duration
        camlog['boutCat'] = tail_cat  # adds this array to camera log
        exp = camlog.merge(stimlog, on='Id', how='left')  # merge camera log and stimlog
        exp[['xPosCartDot', 'yPosCartDot']] = exp[['xPosCartDot', 'yPosCartDot']].fillna(
            method='ffill')  # forward fill to pad out all rows with a dot position
        exp['StopWatchTime'] = exp['StopWatchTime'].interpolate()  # interpolate time
        exp, trials = add_trial_number(exp)  # this function returns a new dataframe with trial number added.

        #######################################################################################################################

        ''' PLOT 1 Trajectory '''
        plot_trajectory(camlog, 'xPosCart', 'yPosCart',
                        new_filename=os.path.join(save_path, (filename + '_1_trajectory.tiff')))

        ''' PLOT 2 + 3 Time Spent Moving and Distance Moved'''
        plot_time_spent_moving(exp, new_filename=os.path.join(save_path, (filename + '_2_TimeSpentMoving.tiff')))
        plot_distance(exp, new_filename=os.path.join(save_path, (filename + '_3_Distance.tiff')))

        ''' PLOT 4 Calculate number of bouts per category per stimulus condition '''
        bout_count_temp = dft.groupby('Trial')['boutCat'].value_counts().unstack().fillna(0).rename_axis(index=None,
                                                                                                         columns=None)
        bca = pd.DataFrame(np.zeros(shape=(41, 13)),
                           columns=np.arange(1, 14))  # create empty dataframe with all bout categories
        bca.update(bout_count_temp)  # map actual bouts observed onto empty dataframe, this helps with plotting
        bca.columns = unordered_bouts  # rename columns to bout type names
        bca = bca[ordered_bouts]  # reorder dataframe to plot in correct order

        plot_bouts_per_trials(bca, new_filename=os.path.join(save_path, (filename + '_4_boutfreq.tiff')))
        plot_bouts_per_condition(bca, new_filename=os.path.join(save_path, (filename + '_5_total_bout_count.tiff')))
        plot_pie_bouttypes(bca, new_filename=os.path.join(save_path, (filename + '_6_pie_bouttypes.tiff')))

        ''' IBI per hab, trial, break distribution plots '''
        plot_IBI(dft, new_filename=os.path.join(save_path, (filename + '_7_IBI.tiff')))

        ''' End Tail Angle over time per fish '''
        trial_st = trials.index.to_numpy()[1::2]

        plot_tailangle_time(TailAnglesTrim, trial_st, new_filename=os.path.join(save_path, (filename + '_8_TailAngles.tiff')))