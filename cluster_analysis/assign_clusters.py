#%%
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
import math
from numpy.linalg import norm
import re
import glob


treeFile = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/optimize_k/tree_nodes.csv'
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

def cosine_sim(pt1, pt2):
    A = pt1 #pd.to_numeric(np.squeeze(np.asarray(pt1)))
    B = pt2 #pd.to_numeric(np.squeeze(np.asarray(pt2)))
    cos = np.dot(A,B)/(norm(A)*norm(B))
    return(cos)

# def regular_on_sphere_points(r,num):
#     points = []
#     #Break out if zero points
#     if num==0:
#         return points
#     a = 4.0 * math.pi*(r**2.0 / num)
#     d = math.sqrt(a)
#     m_theta = int(round(math.pi / d))
#     d_theta = math.pi / m_theta
#     d_phi = a / d_theta

#     for m in range(0,m_theta):
#         theta = math.pi * (m + 0.5) / m_theta
#         m_phi = int(round(2.0 * math.pi * math.sin(theta) / d_phi))
#         for n in range(0,m_phi):
#             phi = 2.0 * math.pi * n / m_phi
#             x = r * math.sin(theta) * math.cos(phi)
#             y = r * math.sin(theta) * math.sin(phi)
#             z = r * math.cos(theta)
#             points.append([x,y,z])
#     return points

#%%

treeNodesCSV = pd.read_csv(treeFile, index_col=False)
treeNodes = pd.DataFrame(treeNodesCSV, columns = ['userGroup','localMaxIndices'])

#%%
# column to split user data by
grouping = 'weekNumber'

all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)
for file in all_files:
    # dfAccel = pd.read_parquet(file, engine='pyarrow')
    dfAccel = pd.read_parquet(file, engine='pyarrow')
    # filter accel points to be on unit sphere:
    accel_filter(dfAccel)
    # convert cartesian coordinates to spherical
    addSpherCoords(dfAccel)

    dfByUser = dfAccel.groupby(['userID', grouping])
    for userAndGrp, group in dfByUser:
        print('user: ' + str(userAndGrp[0])) # user
        print('grouping: ' + str(userAndGrp[1]))  # time grouping for that user
        # if group['userID'].iloc[0] == 2:
        #     break

        # locate cluster center nodes for user/group pair
        clustCenters = treeNodes.loc[treeNodes['userGroup'] == ]

# %%
