#%%
import pandas as pd
import re
import glob
import numpy as np

pathIn = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/accel_with_clusters/'
filePHQ = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/PHQ/allUsers_PHQdata.csv'
fileDiag = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/diagnosis/allUsers_diagnosisData.csv'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/accel_with_clusters/processed/'

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

def num_cluster_transitions(cluster_col):
    n_transitions = np.where(np.diff(cluster_col) != 0, 1, 0).sum()
    return(n_transitions)

def num_binary_orientation_transitions(binary_col):
    binary = np.where(binary_col == 'Upright', 0, 1)
    n_transitions = np.where(np.diff(binary) != 0, 1, 0).sum()
    return(n_transitions)

def num_Z_orientation_transitions(Z_orientations):
    conditions = [Z_orientations == 'phone_face_up', Z_orientations == 'phone_face_vertical', Z_orientations == 'phone_face_down']
    choices = [0,1,2]
    Z_factors = np.select(conditions, choices, default=np.nan)
    n_transitions = np.where(np.diff(Z_factors) != 0, 1, 0).sum()
    return(n_transitions)

#######################################################################################################

dfPHQ = pd.read_csv(filePHQ, index_col=False)
dfDiag = pd.read_csv(fileDiag, index_col=False)

dfPHQ['date'] = pd.to_datetime(dfPHQ['timestampLocal'], format='%Y-%m-%d %H:%M:%S.%f').dt.date


# determine how much of data belongs to cluster per user
all_files = sorted(glob.glob(pathIn + "*_v3.csv"), key = numericalSort)
for file in all_files:
    df = pd.read_csv(file, index_col=False)
    print(df['userID'].iloc[0])

    # find median XYZ per session
    dictMedian = dict()
    dfSession = df.groupby(['recordId'])
    for session, group in dfSession:
        medianX = np.nanmedian(group['x'])
        medianY = np.nanmedian(group['y'])
        medianZ = np.nanmedian(group['z'])
        dictMedian.update({session: {'medianX': medianX, 'medianY': medianY, 'medianZ': medianZ}})
    df['medianX'] = df.apply(lambda x: dictMedian[x['recordId']]['medianX'], axis=1)
    df['medianY'] = df.apply(lambda x: dictMedian[x['recordId']]['medianY'], axis=1)
    df['medianZ'] = df.apply(lambda x: dictMedian[x['recordId']]['medianZ'], axis=1)
    
    # label sessions by 2 & 3 choice orientations
    df['binary_orientation'] = np.where((df['medianZ'] <= 0.1) & (df['medianX'] >= -0.2) & (df['medianX'] <= 0.2), 'Upright', 'Laying')
    conditions = [df['medianZ']<=-0.5, (df['medianZ']>-0.5) & (df['medianZ']<0.5), df['medianZ']>=0.5]
    choices = ['phone_face_up','phone_face_vertical','phone_face_down']
    df['Zorientation'] = np.select(conditions, choices, default=np.nan)

    # transitions per day
    dictTransitions = dict()
    dfWeek = df.groupby(['dayNumber'])
    for day, grp in dfWeek:
        n_clust_trans = num_cluster_transitions(grp['cluster'])
        n_binary_trans = num_binary_orientation_transitions(grp['binary_orientation'])
        n_Z_trans = num_Z_orientation_transitions(grp['Zorientation'])
        dictTransitions.update({day: {'n_clust_trans': n_clust_trans, 'n_binary_trans': n_binary_trans, 'n_Z_trans': n_Z_trans}})
    df['n_cluster_transitions'] = df.apply(lambda x: dictTransitions[x['dayNumber']]['n_clust_trans'], axis=1)
    df['n_binary_transitions'] = df.apply(lambda x: dictTransitions[x['dayNumber']]['n_binary_trans'], axis=1)
    df['n_Z_transitions'] = df.apply(lambda x: dictTransitions[x['dayNumber']]['n_Z_trans'], axis=1)

############################################################################################################
    # PHQ
    userPHQ = dfPHQ.loc[dfPHQ['healthCode'] == df['userID'].iloc[0]]
    phqScores = userPHQ.loc[(userPHQ['date'] >= wkDate) & (userPHQ['date'] <= wkEnd)]
    phq = phqScores['total'].max()


    #%% PHQ propagation
    dfOutUsers = dfOut.groupby(['healthCode'], sort=False, dropna=False)    
    scores = []
    for u, g in dfOutUsers:
        fwdprop = g['PHQ'].fillna(method = 'ffill', limit = 12)
        i=0
        for i in range(0, len(fwdprop)):
            scores.append(fwdprop.iloc[i])
    dfOut['PHQ_fwdProp'] = scores


    break
    df.to_csv(pathOut + file[81:-4] + '_processed.csv', index=False)
    
print('finish')

# %%
