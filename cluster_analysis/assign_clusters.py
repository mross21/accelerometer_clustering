#%%
import pandas as pd
import numpy as np
import re
import glob
from pyarrow import parquet
from numpy.linalg import norm

cluster_file = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/cluster_means.csv'
phq_file = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/PHQ/allUsers_PHQdata.csv'
diag_file = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/diagnosis/allUsers_diagnosisData.csv'
pathAccel = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/accel_with_clusters/'

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
    dfOut = xyz.loc[(xyz['r'] >= 0.9) & (xyz['r'] <= 1.1)]
    return(dfOut)

def addSpherCoords(xyz): # from spherical_kde function
    x = pd.to_numeric(xyz['z'])
    y = pd.to_numeric(xyz['x'])
    z = pd.to_numeric(xyz['y'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    xyz['phi'] = round(np.mod(np.arctan2(y, x), np.pi*2),2) 
    xyz['theta'] = round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2) 
    return(xyz)

def cosine_sim(coord, cmean):
    A = pd.to_numeric(np.squeeze(np.asarray(coord)))
    B = pd.to_numeric(np.squeeze(np.asarray(cmean)))
    cos = np.dot(A,B)/(norm(A)*norm(B))
    return(cos)

def find_clust_label(coord, dfCMean, tol):
    # cos = 0 means vectors overlap (very similar)
    if (dfCMean.iloc[0]['x'] == 0) & (dfCMean.iloc[0]['y'] == 0) & (dfCMean.iloc[0]['z'] == 0):
        cID = 0
    dfCMean['cos'] = dfCMean.apply(lambda row: cosine_sim(coord, row[['x','y','z']]), axis=1)
    if min(dfCMean['cos']) > tol: # if min cos > tol, then coord outside all clusters
        cID = 0 # no cluster found
    else:
        cID = dfCMean.loc[dfCMean['cos'] == min(dfCMean['cos'])]['clustID'].iloc[0] # ID based on min cos
    return(cID)

#%%
dfMeans = pd.read_csv(cluster_file, index_col=False)
dfPHQ = pd.read_csv(phq_file, index_col=False)
dfDiag = pd.read_csv(diag_file, index_col=False)

all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)

clust_list = []

for file in all_files:
    # dfAccel = pd.read_parquet(file, engine='pyarrow')
    dfAccel = pd.read_parquet(file, engine='pyarrow')
    # filter accel points to be on unit sphere:
    accel_filter(dfAccel)
    # convert cartesian coordinates to spherical
    #addSpherCoords(dfAccel)

    dfByUser = dfAccel.groupby(['userID', 'weekNumber'])
    for userAndWk, group in dfByUser:
        for i in range(len(group)): # loop through coordinates in user's week
            print('user: ' + str(userAndWk[0])) # user
            print('week: ' + str(userAndWk[1]))  # week number for that user
            print('index: ' + str(i) + '/' + str(len(group)))
            print('---------')
            xyz = group[['x','y','z']].iloc[i]
            clustGrp = dfMeans.loc[(dfMeans['userID'] == userAndWk[0]) & (dfMeans['weekNumber'] == userAndWk[1])]
            cID = find_clust_label(xyz, clustGrp, 0.5) # what should the tolerance be?
            clust_list.append(cID)

    dfAccel['cluster'] = clust_list

    dfAccel.to_csv(pathOut + 'user_' + str(userAndWk[0]) + '_accelData_clusters.csv', index=False)

# %%
