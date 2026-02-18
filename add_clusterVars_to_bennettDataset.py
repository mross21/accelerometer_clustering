"""
@author: Mindy Ross
python version 3.7.4
pandas version: 1.3.5
numpy version: 1.19.2
"""
# Create new variables for Bennett dataset using new accelerometer clusters

import pandas as pd
import os
import re
import numpy as np
from pyarrow import parquet
import datetime

# paths
pathInKeypress = '*'
pathInAccel = '*'
fileUserIDs = '*' # parquet file
pathOut = '*'

# functions
# match kp and accel files
def find_pair(user_num):
    ls_accel_names      = os.listdir(pathInAccel)
    ls_kp_names         = os.listdir(pathInKeypress)

    kp_file = "fake"
    for user_strKP in ls_kp_names:
        extension = user_strKP.split('.')[1]
        if extension != "parquet":
            print("NOT A PARQUET")
            continue
        num = int(user_strKP.split('_')[1])
        if num == user_num:
            kp_file = user_strKP

    accel_file = "fake"
    for user_strAcc in ls_accel_names:
        extension = user_strAcc.split('.')[1]
        if extension != "parquet":
            print("NOT A PARQUET")
            continue
        num = int(user_strAcc.split('_')[1])
        if num == user_num:
            accel_file = user_strAcc
    if (accel_file != "fake") & (kp_file != "fake"):
        return (kp_file, accel_file)

# sort files by numerical order
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

def accel_filter(xyz):
    x = pd.to_numeric(xyz['x'])
    y = pd.to_numeric(xyz['y'])
    z = pd.to_numeric(xyz['z'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    dfOut = xyz.loc[(xyz['r'] >= 0.95) & (xyz['r'] <= 1.05)]
    return(dfOut)

def check_timestamps(dataframe):
    matchingTimestamps = []
    dfBySess = dataframe.groupby('recordId')
    for sess, group in dfBySess:
        kpT1 = group['keypressTimestampLocal'].iloc[0]
        kpT2 = group['keypressTimestampLocal'].iloc[-1]
        try:
            sessDur = datetime.timedelta(seconds=float(group['session_duration'].iloc[0]))
        except ValueError:
            continue
        Tdiff = kpT2 - kpT1
        delayT = sessDur - Tdiff
        if (delayT < pd.to_timedelta(7, unit='seconds')) or (delayT > pd.to_timedelta(0, unit='seconds')): 
            # print('=======================================')
            # print('kp time: ' + str(kpT1))
            # print('session time: ' + str(group.sessionTimestampLocal.iloc[0]))
            # print('=======================================')
            kpT = int(float(group.keypress_timestamp.iloc[0]))
            # print('kpT ' + str(kpT))
            sessT = int(group.session_timestamp.iloc[0][:-3])
            # print('sessT ' + str(sessT))
            if kpT == sessT:
                matchingTimestamps.append(sess)
            else:
                # print('timestamps dont match')
                # print('-- -- -- -- -- -- -- -- --')
                pass
        else:
            pass
            # print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
            # print('delay after last keypress not within 0 - 7 sec')
            # print('delay: ' + str(delayT.total_seconds()))
            # print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return(matchingTimestamps)

# new accel variables
def get_number_cluster_transitions(cluster):
    transition_flag = [np.where((cluster.iloc[i] - cluster.iloc[i-1]) == 0, 0, 1) for i in range(1, len(cluster))]
    n_transitions = sum(transition_flag)
    return(n_transitions)

def accelClusterFeatures(dataframe):
    n_clusters = int(max(dataframe['cluster_center']))
    total_dist = dataframe.groupby('sessionNumber', as_index=False).agg(
                                    {'dist_from_prev_cluster': 'first'}).sum()
    avg_n_cluster_perSess = dataframe.groupby('sessionNumber', as_index=False).agg(
                                            {'n_clusters_per_session': lambda x:x.value_counts().index[0]}).mean()
    avg_n_transitions_perSess = dataframe.groupby('sessionNumber', as_index=False).agg(
                                        {'n_cluster_transitions_per_session': lambda x:x.value_counts().index[0]}).mean()
    n_transitions = get_number_cluster_transitions(dataframe.groupby('sessionNumber', 
                            as_index=False).agg({'session_cluster': lambda x:x.value_counts().index[0]})['session_cluster'])
    return(n_clusters,total_dist,avg_n_cluster_perSess,avg_n_transitions_perSess,n_transitions)          

def accelMetrics_v2(dataframe):
    medX = np.nanmedian(dataframe.groupby('sessionNumber', as_index=False).agg(
                                    {'cluster_center_x': 'first'})['cluster_center_x'])
    medY = np.nanmedian(dataframe.groupby('sessionNumber', as_index=False).agg(
                                    {'cluster_center_y': 'first'})['cluster_center_y'])
    medZ = np.nanmedian(dataframe.groupby('sessionNumber', as_index=False).agg(
                                    {'cluster_center_z': 'first'})['cluster_center_z'])   
    return(medX,medY,medZ)   

def accMotion_v2(dataframe, col_name):
    l = dataframe.groupby('sessionNumber', as_index=False).agg(
                                    {col_name: 'first'})[col_name]
    sum_mot = sum((l - l.shift(1)).dropna())
    sd_mot = np.std((l - l.shift(1)).dropna())
    return(sum_mot, sd_mot)

def rotMotion_v2(dataframe):
    xList = dataframe.groupby('sessionNumber', as_index=False).agg(
                                    {'cluster_center_x': 'first'})['cluster_center_x']
    yList = dataframe.groupby('sessionNumber', as_index=False).agg(
                                    {'cluster_center_y': 'first'})['cluster_center_y']
    zList = dataframe.groupby('sessionNumber', as_index=False).agg(
                                    {'cluster_center_z': 'first'})['cluster_center_z']
    chord_new = np.sqrt(((xList - xList.shift(1))**2) + ((yList - yList.shift(1))**2) + 
                            ((zList - zList.shift(1))**2))
    chord_sum_new = sum(chord_new.dropna())
    arc_new = 2*np.arcsin(chord_new/2)
    arc_sum_new = sum(arc_new.dropna())
    return(chord_sum_new,arc_sum_new)

# to append data
listOut = []

# read in file for userIDs
dfUserIDs = pd.read_parquet(fileUserIDs, engine='pyarrow')
# find max user number
max_user = dfUserIDs['userID'].max()

# iterate through all files
for i in range(0, max_user):
    try:
        kp_file = find_pair(i)[0]
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(kp_file)
        accel_file = find_pair(i)[1]
        print(accel_file)
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
    except:
        continue

    dfKP = pd.read_parquet(pathInKeypress + kp_file, engine='pyarrow')
    dfAccel = pd.read_parquet(pathInAccel + accel_file, engine='pyarrow')
    
    # skip android data
    if 'Android' in dfAccel['phoneInfo'].iloc[0]:
        print('user ' + str(dfAccel['userID'].iloc[0]) + ' uses android phone => skip')
        continue

    # sort to chronological order based on keypress timestamp
    dfAccel['sessionTimestampLocal'] = pd.to_datetime(dfAccel['sessionTimestampLocal'])
    dfAccel.sort_values(by=['sessionTimestampLocal'], ascending=True, inplace=True, ignore_index=True)

    # filter accelerometer values to only ones with 0.95 < r < 1.05
    dfAccel = accel_filter(dfAccel)

    # format date and hour
    dfKP['date'] = pd.to_datetime(dfKP.date)
    dfKP['hour'] = pd.to_datetime(dfKP['keypressTimestampLocal']).dt.hour

    dfAccel['date'] = pd.to_datetime(dfAccel.date)
    dfAccel['hour'] = pd.to_datetime(dfAccel['sessionTimestampLocal']).dt.hour

    dfAccel['x'] = dfAccel['x'].apply(lambda x: float(x))
    dfAccel['y'] = dfAccel['y'].apply(lambda x: float(x))
    dfAccel['z'] = dfAccel['z'].apply(lambda x: float(x))

    # remove NaN dates 
    dfKP = dfKP.dropna(subset = ['date'])
    dfAccel = dfAccel.dropna(subset = ['date'])

    # restrict dates to original analysis
    dfKP = dfKP.loc[dfKP['date'] <= datetime.datetime(2022, 1, 31)]
    dfAccel = dfAccel.loc[dfAccel['date'] <= datetime.datetime(2022, 1, 31)]

    # find min date
    minDate =  dfKP.date.min() 
    # find max date
    maxD = dfKP.date.max() 
    study_duration = maxD - minDate
    if study_duration < pd.to_timedelta(7, unit = 'days'):
        maxDate = minDate + pd.to_timedelta(7, unit = 'days')
    else:
        maxDate = maxD 
       
    wkDate = minDate
    # loop through each week for the user
    for wk in pd.date_range(minDate, maxDate, freq = 'W'):
        wkDate = wkDate
        wkEnd = wkDate + pd.to_timedelta(6, unit = 'days')
        # week of keypresses to create variables
        group = dfKP.loc[(dfKP['date'] >= wkDate) & (dfKP['date'] <= wkEnd)]
        groupAcc = dfAccel.loc[(dfAccel['date'] >= wkDate) & (dfAccel['date'] <= wkEnd)]

        # healthCode, age, gender, date
        hc = dfKP['healthCode'].iloc[1]

# new accel variables by week
        # accel cluster features
        if len(groupAcc) < 2:
            continue
        n_clust = accelClusterFeatures(groupAcc)[0]
        tot_clust_dist = accelClusterFeatures(groupAcc)[1][1]
        avg_num_clust_perSess = accelClusterFeatures(groupAcc)[2][1]
        avg_num_trans_perSess = accelClusterFeatures(groupAcc)[3][1]
        n_clust_trans = accelClusterFeatures(groupAcc)[4]

        medX_new = accelMetrics_v2(groupAcc)[0]
        medY_new = accelMetrics_v2(groupAcc)[1]
        medZ_new = accelMetrics_v2(groupAcc)[2]

        # amount of motion
        Xmot_sum_new = accMotion_v2(groupAcc, 'cluster_center_x')[0]
        Xmot_sd_new = accMotion_v2(groupAcc, 'cluster_center_x')[1]
        Ymot_sum_new = accMotion_v2(groupAcc, 'cluster_center_y')[0]
        Ymot_sd_new = accMotion_v2(groupAcc, 'cluster_center_y')[1]
        Zmot_sum_new = accMotion_v2(groupAcc, 'cluster_center_z')[0]
        Zmot_sd_new = accMotion_v2(groupAcc, 'cluster_center_z')[1]     

        # amount of rotational motion
        chord_sum_new = rotMotion_v2(groupAcc)[0]
        arc_sum_new = rotMotion_v2(groupAcc)[1]

# add values to dfOut
        listOut.append((hc, wkDate, n_clust, tot_clust_dist, avg_num_clust_perSess, avg_num_trans_perSess, 
                        n_clust_trans, medX_new, medY_new, medZ_new, Xmot_sum_new, Xmot_sd_new, Ymot_sum_new,
                        Ymot_sd_new, Zmot_sum_new, Zmot_sd_new, chord_sum_new, arc_sum_new))

# start date of next week
        wkDate = wkDate + pd.to_timedelta(7, unit = 'days')
    
# start output dataframe
dfOut = pd.DataFrame(listOut, columns = ['healthCode', 'Date', 'n_clusters', 'total_distance_between_clusters', 'avg_n_clusters_perSession',
                                        'avg_n_transitions_perSession', 'n_cluster_transitions',
                                        'medianX_new', 'medianY_new', 'medianZ_new', 'Xmotion_sum_new', 'Xmotion_sd_new',
                                        'Ymotion_sum_new', 'Ymotion_sd_new', 'Zmotion_sum_new', 'Zmotion_sd_new', 
                                        'chord_sum_new', 'arc_sum_new'])

# save to csv
dfOut.to_csv(pathOut + '*.csv', index=False)

print('finish')
