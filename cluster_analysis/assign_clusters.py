#%%
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
import math
from numpy.linalg import norm
import re
import glob
from scipy import spatial

treeFile = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/optimize_k/tree_nodes_v4.csv'
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
    dfOut = xyz.loc[(xyz['r'] >= 0.95) & (xyz['r'] <= 1.05)]
    return(dfOut)

def addSpherCoords(xyz): # from spherical_kde function
    x = pd.to_numeric(xyz['z'])
    y = pd.to_numeric(xyz['x'])
    z = pd.to_numeric(xyz['y'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    xyz['phi'] = round(np.mod(np.arctan2(y, x), np.pi*2),2) 
    xyz['theta'] = round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2) 
    return(xyz)

# taken from haversine package (removed conversion to radians)
# https://github.com/mapado/haversine/blob/main/haversine/haversine.py
def haversine_dist(pt1,pt2): # theta, phi
    lat1, lng1 = pt1
    lat2, lng2 = pt2
    # convert theta range from 0 to pi to -pi/2 to pi/2 | LATITUDE (-90 to 90)
    lat1 = lat1 - (np.pi/2)
    lat2 = lat2 - (np.pi/2)
    # convert phi range from 0 to 2pi to -pi to pi | LONGITUDE (-180 to 180)
    lng1 = lng1 - np.pi
    lng2 = lng2 - np.pi
    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = (math.sin(lat * 0.5)**2) + (math.cos(lat1) * math.cos(lat2) * (math.sin(lng * 0.5)**2))
    return 2  * math.asin(math.sqrt(d))

def cosine_sim(pt1, pt2): # need points in XYZ
    A = pd.to_numeric(np.squeeze(np.asarray(pt1)))
    B = pd.to_numeric(np.squeeze(np.asarray(pt2)))
    cos = np.dot(A,B)/(norm(A)*norm(B))
    return(cos)   

def nearest_neighbour(points_a, points_b):
    tree = spatial.cKDTree(points_b) # indexes points to be compared
    return tree.query(points_a)[1] # get index of closest point above to coordinate

def flatten(l):
    return [item for sublist in l for item in sublist]

#%%
treeNodesCSV = pd.read_csv(treeFile, index_col=False)
treeNodes = pd.DataFrame(treeNodesCSV)

# column to split user data by
grouping = 'weekNumber'

all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)
for file in all_files:
    # dfAccel = pd.read_parquet(file, engine='pyarrow')
    dfAccel = pd.read_parquet(file, engine='pyarrow')

    if dfAccel['userID'].iloc[0] < 40:
        continue
    
    # if dfAccel['userID'].iloc[0] < 8:
    #     continue

    # filter accel points to be on unit sphere:
    df = accel_filter(dfAccel)

    # convert cartesian coordinates to spherical
    addSpherCoords(df)

    # clust_list = []
    dfOut = []
    dfByUser = df.groupby(['userID', grouping])
    for userAndGrp, group in dfByUser:
        group = group.reset_index()
        print('user: ' + str(userAndGrp[0])) # user
        print('grouping: ' + str(userAndGrp[1]))  # time grouping for that user

        # locate cluster center coordinates for user/group pair
        usrGrp = ';'.join([str(float(userAndGrp[0])),str(float(userAndGrp[1]))])
        grpClustCtrs = treeNodes.loc[treeNodes['userGroup'] == usrGrp]
        if len(grpClustCtrs) == 0:
            # clust_list.append([0]*len(group))
            group['cluster'] = [0]*len(group)
            continue
        # find nearest cluster center to coordinate
        # index is cluster coordinates reindexed 0-len(grpClustCtrs)
        group['neighborIdx'] = nearest_neighbour(group[['x','y','z']],grpClustCtrs[['x','y','z']])

        # calculate cosine similarity between coordinate and nearest cluster center
        group['xN'] = grpClustCtrs['x'].iloc[group['neighborIdx']].reset_index(drop=True) # get X value of cluster center closest to group['x']
        group['yN'] = grpClustCtrs['y'].iloc[group['neighborIdx']].reset_index(drop=True) # get Y value of cluster center closest to group['y']
        group['zN'] = grpClustCtrs['z'].iloc[group['neighborIdx']].reset_index(drop=True) # get Z value of cluster center closest to group['z']
        group['cosine_similarity'] = group.apply(lambda row: cosine_sim(row[['x','y','z']], row[['xN','yN','zN']]), axis=1)
        
        # if coordinate is close to cluster center (~10 nearest neighbors, 25 degrees) then label cluster by #, else 0
        # 0 means no cluster
        group['cluster'] = np.where(group['cosine_similarity'] >= 0.975, group['neighborIdx']+1, 0)
        # clust_list.append(group['cluster'])
        dfOut.append(group)
    dfOut = pd.concat(dfOut,axis=0,ignore_index=True)

    # df.to_csv(pathOut + 'user_' + str(userAndGrp[0]) + '_accelData_clusters.csv', index=False)
    dfOut.to_csv(pathOut + 'user_' + str(userAndGrp[0]) + '_accelData_clusters.csv', index=False)

print('finish')

# %%
group[['index','healthCode','recordId','weekNumber','x','y','z','xN','yN','zN','neighborIdx','cosine_similarity','cluster']].to_csv('/home/mindy/Desktop/test.csv', index=False)
# %%
