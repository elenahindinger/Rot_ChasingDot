__author__ = 'Elena Maria Daniela Hindinger'

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.io import loadmat
from Rot_ChasingDot_Functions import *


exp_path = r'F:\Rotations\ExperimentalData\2019_10_24\20191024_experimental_Trial1_f1_Tu_4275ix_6dpf_Atlas_75_P2_chasingdot'

camfile, stimfile, boutfile, filename = find_files(exp_path=exp_path)

camlog = pd.DataFrame(loadmat(camfile)['A'], columns=['Id', 'xPos', 'yPos', 'xBodyVector', 'yBodyVector', 'BodyAngle',
                                                      'MaxValue', 'BlobValue', 'midEyeX', 'midEyeY', 'hour', 'minute',
                                                      'StimuliIncr', 'CumSumTail', 'TailValues1', 'TailValues2',
                                                      'TailValues3', 'TailValues4', 'TailValues5', 'TailValues6',
                                                      'TailValues7', 'TailValues8', 'TailValues9', 'TailValues10',
                                                      'TailAngles1', 'TailAngles2', 'TailAngles3', 'TailAngles4',
                                                      'TailAngles5', 'TailAngles6', 'TailAngles7', 'TailAngles8',
                                                      'TailAngles9', 'TailAngles10', 'FirstorNot', 'TrackEye',
                                                      'TimerAcq', 'lag'])  # read camlog

stimlog = read_stimlog(stimfile)  # read stimlog

boutmat = loadmat(boutfile)  # load bout mat file

Bouts = return_as_df(boutmat, keys=['allboutstarts', 'allboutends', 'rejectedBouts', 'indRealEnds'])
BoutCat = return_as_df(boutmat, keys=['boutCat'])
DistToCenter = return_as_df(boutmat, keys=['distToCenter'])
BodyAngles = return_as_df(boutmat, keys=['realBodyAngles'])
BoutSegMeth = return_as_df(boutmat, keys=['smootherTailCurveMeasure'])

# KinPar = return_as_df(boutmat, keys=['BoutKinematicParameters'])
# NewScore = return_as_df(boutmat, keys=['newScore'])
# TailAngles = return_as_df(boutmat, keys=['smoothedCumsumInterpFixedSegmentAngles'])

indexNames = Bouts[Bouts['rejectedBouts'] == 1].index  # Get names of indexes for which column rejected Bouts is True
Bouts = Bouts.drop(indexNames).reset_index().drop(['rejectedBouts', 'index'], axis=1) # Delete these row indexes from dataFrame
df = pd.concat([Bouts.iloc[:,:2], BoutCat], axis=1)

# array with st1, end1, st2, end2
# map boutCat onto array, pad
#
