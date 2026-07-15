import datetime
import os

import numpy as np
import pandas as pd

from accelerometer_utils import accel_filter
from config import PATH_ACCELEROMETER_PARQUET, PATH_KEYPRESS, PATH_OUTPUT_DATA, PATH_USER_IDS
from io_utils import out_path


def find_pair(user_num, pathInAccel, pathInKeypress):
    ls_accel_names = os.listdir(pathInAccel)
    ls_kp_names = os.listdir(pathInKeypress)

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
    if (accel_file != "fake") and (kp_file != "fake"):
        return (kp_file, accel_file)


def get_number_cluster_transitions(cluster):
    transition_flag = [np.where((cluster.iloc[i] - cluster.iloc[i - 1]) == 0, 0, 1) for i in range(1, len(cluster))]
    return sum(transition_flag)


def accelClusterFeatures(dataframe):
    n_clusters = int(max(dataframe['cluster_center']))
    total_dist = dataframe.groupby('sessionNumber', as_index=False).agg({'dist_from_prev_cluster': 'first'}).sum()
    avg_n_cluster_perSess = dataframe.groupby('sessionNumber', as_index=False).agg({'n_clusters_per_session': lambda x: x.value_counts().index[0]}).mean()
    avg_n_transitions_perSess = dataframe.groupby('sessionNumber', as_index=False).agg({'n_cluster_transitions_per_session': lambda x: x.value_counts().index[0]}).mean()
    n_transitions = get_number_cluster_transitions(dataframe.groupby('sessionNumber', as_index=False).agg({'session_cluster': lambda x: x.value_counts().index[0]})['session_cluster'])
    return (n_clusters, total_dist, avg_n_cluster_perSess, avg_n_transitions_perSess, n_transitions)


def accelMetrics_v2(dataframe):
    medX = np.nanmedian(dataframe.groupby('sessionNumber', as_index=False).agg({'cluster_center_x': 'first'})['cluster_center_x'])
    medY = np.nanmedian(dataframe.groupby('sessionNumber', as_index=False).agg({'cluster_center_y': 'first'})['cluster_center_y'])
    medZ = np.nanmedian(dataframe.groupby('sessionNumber', as_index=False).agg({'cluster_center_z': 'first'})['cluster_center_z'])
    return (medX, medY, medZ)


def accMotion_v2(dataframe, col_name):
    l = dataframe.groupby('sessionNumber', as_index=False).agg({col_name: 'first'})[col_name]
    sum_mot = sum((l - l.shift(1)).dropna())
    sd_mot = np.std((l - l.shift(1)).dropna())
    return (sum_mot, sd_mot)


def rotMotion_v2(dataframe):
    xList = dataframe.groupby('sessionNumber', as_index=False).agg({'cluster_center_x': 'first'})['cluster_center_x']
    yList = dataframe.groupby('sessionNumber', as_index=False).agg({'cluster_center_y': 'first'})['cluster_center_y']
    zList = dataframe.groupby('sessionNumber', as_index=False).agg({'cluster_center_z': 'first'})['cluster_center_z']
    chord_new = np.sqrt(((xList - xList.shift(1)) ** 2) + ((yList - yList.shift(1)) ** 2) + ((zList - zList.shift(1)) ** 2))
    chord_sum_new = sum(chord_new.dropna())
    arc_new = 2 * np.arcsin(chord_new / 2)
    arc_sum_new = sum(arc_new.dropna())
    return (chord_sum_new, arc_sum_new)


def main():
    listOut = []
    max_user = pd.read_parquet(PATH_USER_IDS, engine='pyarrow')['userID'].max()

    for i in range(0, max_user):
        try:
            pair = find_pair(i, PATH_ACCELEROMETER_PARQUET, PATH_KEYPRESS)
            print('++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(pair[0])
            print(pair[1])
            print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        except:
            continue

        dfKP = pd.read_parquet(PATH_KEYPRESS + pair[0], engine='pyarrow')
        dfAccel = pd.read_parquet(PATH_ACCELEROMETER_PARQUET + pair[1], engine='pyarrow')

        if 'Android' in dfAccel['phoneInfo'].iloc[0]:
            print('user ' + str(dfAccel['userID'].iloc[0]) + ' uses android phone => skip')
            continue

        dfAccel['sessionTimestampLocal'] = pd.to_datetime(dfAccel['sessionTimestampLocal'])
        dfAccel.sort_values(by=['sessionTimestampLocal'], ascending=True, inplace=True, ignore_index=True)
        dfAccel = accel_filter(dfAccel)

        dfKP['date'] = pd.to_datetime(dfKP.date)
        dfKP['hour'] = pd.to_datetime(dfKP['keypressTimestampLocal']).dt.hour
        dfAccel['date'] = pd.to_datetime(dfAccel.date)
        dfAccel['hour'] = pd.to_datetime(dfAccel['sessionTimestampLocal']).dt.hour
        dfAccel['x'] = dfAccel['x'].apply(lambda x: float(x))
        dfAccel['y'] = dfAccel['y'].apply(lambda x: float(x))
        dfAccel['z'] = dfAccel['z'].apply(lambda x: float(x))
        dfKP = dfKP.dropna(subset=['date'])
        dfAccel = dfAccel.dropna(subset=['date'])
        dfKP = dfKP.loc[dfKP['date'] <= datetime.datetime(2022, 1, 31)]
        dfAccel = dfAccel.loc[dfAccel['date'] <= datetime.datetime(2022, 1, 31)]

        minDate = dfKP.date.min()
        maxD = dfKP.date.max()
        study_duration = maxD - minDate
        if study_duration < pd.to_timedelta(7, unit='days'):
            maxDate = minDate + pd.to_timedelta(7, unit='days')
        else:
            maxDate = maxD

        wkDate = minDate
        for wk in pd.date_range(minDate, maxDate, freq='W'):
            wkEnd = wkDate + pd.to_timedelta(6, unit='days')
            groupAcc = dfAccel.loc[(dfAccel['date'] >= wkDate) & (dfAccel['date'] <= wkEnd)]

            hc = dfKP['healthCode'].iloc[1]
            if len(groupAcc) < 2:
                continue

            cluster_features = accelClusterFeatures(groupAcc)
            n_clust = cluster_features[0]
            tot_clust_dist = cluster_features[1][1]
            avg_num_clust_perSess = cluster_features[2][1]
            avg_num_trans_perSess = cluster_features[3][1]
            n_clust_trans = cluster_features[4]

            medX_new, medY_new, medZ_new = accelMetrics_v2(groupAcc)
            Xmot_sum_new, Xmot_sd_new = accMotion_v2(groupAcc, 'cluster_center_x')
            Ymot_sum_new, Ymot_sd_new = accMotion_v2(groupAcc, 'cluster_center_y')
            Zmot_sum_new, Zmot_sd_new = accMotion_v2(groupAcc, 'cluster_center_z')
            chord_sum_new, arc_sum_new = rotMotion_v2(groupAcc)

            listOut.append((hc, wkDate, n_clust, tot_clust_dist, avg_num_clust_perSess, avg_num_trans_perSess,
                            n_clust_trans, medX_new, medY_new, medZ_new, Xmot_sum_new, Xmot_sd_new, Ymot_sum_new,
                            Ymot_sd_new, Zmot_sum_new, Zmot_sd_new, chord_sum_new, arc_sum_new))

            wkDate = wkDate + pd.to_timedelta(7, unit='days')

    dfOut = pd.DataFrame(listOut, columns=['healthCode', 'Date', 'n_clusters', 'total_distance_between_clusters', 'avg_n_clusters_perSession',
                                           'avg_n_transitions_perSession', 'n_cluster_transitions',
                                           'medianX_new', 'medianY_new', 'medianZ_new', 'Xmotion_sum_new', 'Xmotion_sd_new',
                                           'Ymotion_sum_new', 'Ymotion_sd_new', 'Zmotion_sum_new', 'Zmotion_sd_new',
                                           'chord_sum_new', 'arc_sum_new'])
    dfOut.to_parquet(out_path(PATH_OUTPUT_DATA, 'cluster_variables.parquet'), index=False)
    print('finish')


if __name__ == '__main__':
    main()
