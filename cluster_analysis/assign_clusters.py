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

def cosine_similarity(coord, cmean):
    A = np.array(coord)
    B = np.array(cmean)
    cos = np.dot(A,B)/(norm(A)*norm(B))
    return(cos)

def find_clust_label(coord, dfCMean, tol):
    # cos = 0 means vectors overlap (very similar)
    dfCMean['cos'] = dfCMean.apply(lambda row: cosine_similarity(coord, row[['x','y','z']], axis=1))
    if min(dfCMean['cos']) > tol: # if min cos > tol, then coord outside all clusters
        cID = float('Nan') # no cluster found
    else:
        cID = dfCMean.loc[min(dfCMean['cos'])]['clustID'] # ID based on min cos
    return(cID)

#%%
dfMeans = pd.read_csv(cluster_file, index_col=False)
dfPHQ = pd.read_csv(phq_file, index_col=False)
dfDiag = pd.read_csv(diag_file, index_col=False)

all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)

# dictClustID = {}
# dictClustID['user'] = {}

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
        print('user: ' + str(userAndWk[0])) # user
        print('week: ' + str(userAndWk[1]))  # week number for that user
        for i in len(group): # loop through coordinates in user's week
            xyz = group[['x','y','z']][i]
            clustGrp = dfMeans.loc[dfMeans['userID'] == userAndWk[0] & dfMeans['weekNumber'] == userAndWk[1]]
            cID = find_clust_label(xyz, clustGrp, 0.2) # what should the tolerance be?
            #dictClustID[userAndWk[0]][userAndWk[1]] = cID
            clust_list = clust_list.append((userAndWk[0],userAndWk[1],xyz['x'],xyz['y'],xyz['z'],cID)) #user,wk,x,y,z,cID

dfClust = pd.DataFrame(clust_list, columns = ['userID','weekNumber','x','y','z','cluster'])











#df['avgDispSession'] = df.apply(lambda x: sessionDict[x['SubjectID']][x['recordId']], axis=1)

                













