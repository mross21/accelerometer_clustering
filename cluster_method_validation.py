"""
@author: Mindy Ross
python version 3.7.4
pandas version: 1.3.5
numpy version: 1.19.2
"""
# Validate clustering method using labeled test data

#%%
# import packages
import pandas as pd
from pyarrow import parquet
import numpy as np
import re
import spherical_kde
from math import cos, sin, asin, sqrt
from scipy import spatial
from scipy.spatial.distance import squareform, pdist
import networkx as nx
from matplotlib import pyplot as plt
# gets rid of the warnings for setting var to loc
pd.options.mode.chained_assignment = None

# FUNCTIONS
# sort files numerically
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

# filter accelerometer data by magnitude
def accel_filter(xyz):
    x = pd.to_numeric(xyz['x'])
    y = pd.to_numeric(xyz['y'])
    z = pd.to_numeric(xyz['z'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    dfOut = xyz.loc[(xyz['r'] >= 0.95) & (xyz['r'] <= 1.05)]
    return(dfOut)

# add spherical coordinates
def addSpherCoords(xyz): # from spherical_kde function
    x = pd.to_numeric(xyz['z'])
    y = pd.to_numeric(xyz['x'])
    z = pd.to_numeric(xyz['y'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    xyz['phi'] = round(np.mod(np.arctan2(y, x), np.pi*2),2) 
    xyz['theta'] = round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2) 
    return(xyz)

# make equidistant points on a sphere
def regular_on_sphere_points(r,num):
    points = []
    #Break out if zero points
    if num==0:
        return points
    a = 4.0 * np.pi*(r**2.0 / num)
    d = sqrt(a)
    m_theta = int(round(np.pi / d))
    d_theta = np.pi / m_theta
    d_phi = a / d_theta
    for m in range(0,m_theta):
        theta = np.pi * (m + 0.5) / m_theta
        m_phi = int(round(2.0 * np.pi * sin(theta) / d_phi))
        for n in range(0,m_phi):
            phi = 2.0 * np.pi * n / m_phi
            x = r * sin(theta) * cos(phi)
            y = r * sin(theta) * sin(phi)
            z = r * cos(theta)
            points.append([x,y,z])
    return points

# find haversine distance taken from haversine package (removed conversion to radians)
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
    d = (sin(lat * 0.5)**2) + (cos(lat1) * cos(lat2) * (sin(lng * 0.5)**2))
    return 2  * asin(sqrt(d))

# find number of clusters and corresponding sphere point indicies
def get_cluster_centers(distance_matrix,density_list,nNeighbors,threshold):
    num_clusters = 0
    idx_list = []
    for i in range(len(distance_matrix)):
        # sort distances by ascending order
        dmSort = distance_matrix[i].sort_values()
        # get list of indices of idx point and 10 closest points
        idxClosePts = dmSort[0:nNeighbors].index 
        # get corresponding densities of those points
        densities = density_list.iloc[idxClosePts]
        # if idx point has largest density [and density > frac max density], add cluster
        add_clust = np.where((max(densities) == densities.iloc[0]) & (densities.iloc[0] >= threshold), 1, 0)
        num_clusters = num_clusters + add_clust
        if add_clust == 1:
            idx_list.append(i)
    return(num_clusters, idx_list)

# get nearest neighbor point from list of points
def nearest_neighbour(points_a, points_b):
    tree = spatial.cKDTree(points_b) # indexes points to be compared
    return tree.query(points_a)[1] # get index of closest point above to coordinate


### ######################################################################################################
# FILE PATHS
# folder of accelerometer data to be clustered
accelFile = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/validation/all_positions_labeled.csv'
# folder where figures should be saved
pathFig = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/validation/'
# folder where output accelerometer data should be saved
pathAccOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/validation/'

# MAKE EQUIDISTANT POINTS TO SAMPLE KDE AT
radius = 1
num = 1000 # get this many sampled points
regular_surf_points = regular_on_sphere_points(radius,num)
pts_xyz=np.array(regular_surf_points)
x=pts_xyz[:,0]
y=pts_xyz[:,1]
z=pts_xyz[:,2]
equi_phi = np.ndarray.round(np.mod(np.arctan2(y, x), np.pi*2),2) 
equi_theta = np.ndarray.round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2) 
# dataframe of theta and phi coordinates of sampled points
spherePts = pd.DataFrame(np.column_stack((equi_theta,equi_phi)), columns = ['theta','phi'])

# MAKE ADJACENCY MATRIX TO BUILD NETWORK GRAPH
# distance matrix for equidistant sphere points
dm = pd.DataFrame(squareform(pdist(spherePts, metric=haversine_dist)), index=spherePts.index, columns=spherePts.index)
# max distance between neighboring points based on 1000 sampled points found experimentally 
# change if not using 1000 points
d = 0.19 
# adjacency matrix for equidistant sphere points
# if distance is less than d, flag as adjacent point
adjMatrix = np.where(dm < d, 1, 0)
# fill diagonal of matrix with 0 since points are not adjacent to themselves
np.fill_diagonal(adjMatrix,0)

### ######################################################################################################
# GET SPHERICAL KDE (VMF) FOR USER/WEEK

# read in user's accelerometer file
dfAllAccel = pd.read_csv(accelFile, index_col=False)

# filter accel points to be on unit sphere:
dfAccel = accel_filter(dfAllAccel)

# convert cartesian coordinates to spherical
addSpherCoords(dfAccel)


# calculate spherical KDE
sKDE = spherical_kde.SphericalKDE(dfAccel['phi'], dfAccel['theta'], weights=None, bandwidth=0.1)
# sample KDE at equidistant points to get densities
sKDE_densities = np.exp(sKDE(equi_phi, equi_theta))
# dataframe of points and densities
KDEdensities = np.column_stack([x,y,z,equi_phi,equi_theta,sKDE_densities])
grpKDE = pd.DataFrame(KDEdensities, columns = ['z', 'x', 'y', 'phi', 'theta', 'density'])

### ######################################################################################################
# find cluster centers

# number of neighbors
n = 9 # 8 neighbors plus the center point
# determine density to set threshold for local max for clusters
b,bins,patches = plt.hist(x=grpKDE['density'], bins=50)
localMaxThreshold = bins[1]

# get clusters for grouping
clusters = get_cluster_centers(dm, grpKDE['density'], n, localMaxThreshold)
k = clusters[0]
cluster_idx = clusters[1]
cluster_densities = list(grpKDE['density'].iloc[cluster_idx])
cluster_theta = list(grpKDE['theta'].iloc[cluster_idx])
cluster_phi = list(grpKDE['phi'].iloc[cluster_idx])
cluster_x = list(grpKDE['x'].iloc[cluster_idx])
cluster_y = list(grpKDE['y'].iloc[cluster_idx])
cluster_z = list(grpKDE['z'].iloc[cluster_idx])

### ######################################################################################################
# assign cluster IDs to raw accelerometer data

# matrix of all edge weights
# edge weights based on density- higher density between the two nodes, the lower the weight (invert density)
edge_weights = pd.DataFrame(squareform(pdist(grpKDE[['density']], lambda u,v: (np.exp(-((u+v)/2))))),
                                            index=grpKDE.index, columns=grpKDE.index)
# matrix of adjacent weights
adj_weights = edge_weights*adjMatrix
# make graph
G = nx.from_numpy_matrix(np.array(adj_weights), parallel_edges=False, create_using=nx.Graph())

# dictionary per user/week
dictClust = {}
dictClustIdx = {}
dictClustX = {}
dictClustY = {}
dictClustZ = {}
dictID = dict(zip(cluster_idx,range(1,len(cluster_idx)+1)))

# find all paths for each index to the possible clusters then identify closest cluster center
for i in range(0,len(adj_weights)):
    path_list = []
    for j in cluster_idx:
        # get path by indices
        path = nx.single_source_dijkstra(G,i,j, weight='weight')
        # get number of elements in path
        num_elements = len(path[1])
        path_list.append((i,j,num_elements,path[0],path[1]))
    dfPaths = pd.DataFrame(path_list, columns = ['point_idx','cluster_idx','n_elements','weight','path'])

    # figure out which cluster the source point should belong to
    min_path = dfPaths.loc[dfPaths['weight'] == min(dfPaths['weight'])]
    if len(min_path) > 1:
        print('min path lengths equal')

    closest_cluster_idx = int(min_path['cluster_idx'])
    # append every point on sphere and assigned cluster center to dictionary
    # match the sphere index with the cluster center/XYZ point
    dictClust[i] = dictID[closest_cluster_idx]
    dictClustIdx[i] = closest_cluster_idx
    dictClustX[i] = grpKDE.loc[grpKDE.index == closest_cluster_idx]['x'].iloc[0]
    dictClustY[i] = grpKDE.loc[grpKDE.index == closest_cluster_idx]['y'].iloc[0]
    dictClustZ[i] = grpKDE.loc[grpKDE.index == closest_cluster_idx]['z'].iloc[0]

######################################################################################################
# map cluster centers to raw accelerometer data 
# set XYZ to float
dfAccel['x'] = dfAccel['x'].astype(float)
dfAccel['y'] = dfAccel['y'].astype(float)
dfAccel['z'] = dfAccel['z'].astype(float)
# find nearest equidistant sphere point for each raw accel point
dfAccel['nodeIdx'] = nearest_neighbour(dfAccel[['x','y','z']], grpKDE[['x','y','z']])

# map cluster center to each raw accel point using the matched sphere point
dfAccel['cluster_center'] = dfAccel['nodeIdx'].map(dictClust)
dfAccel['cluster_center_idx'] = dfAccel['nodeIdx'].map(dictClustIdx)
dfAccel['cluster_center_x'] = dfAccel['nodeIdx'].map(dictClustX)
dfAccel['cluster_center_y'] = dfAccel['nodeIdx'].map(dictClustY)
dfAccel['cluster_center_z'] = dfAccel['nodeIdx'].map(dictClustZ)

## ######################################################################################################
# XZ plot of labeled accelerometer data
plt.rcParams.update({'font.size': 39})
fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
ax = fig.add_subplot()
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.scatter(dfAccel['x'], dfAccel['z'], c=dfAccel['cluster_center'], cmap='Set2_r', s=100)
plt.xlim([-1.2,1.2])
plt.ylim([-1.2,1.2])
plt.savefig(pathFig+'allPositions_XZ.png')

# XY plot
fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
ax = fig.add_subplot()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.scatter(dfAccel['x'], dfAccel['y'], c=dfAccel['cluster_center'], cmap='Set2_r')
plt.xlim([-1.2,1.2])
plt.ylim([-1.2,1.2])
# plt.show()
plt.savefig(pathFig+'allPositions_XY.png')

# YZ plot
fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
ax = fig.add_subplot()
ax.set_xlabel('Y')
ax.set_ylabel('Z')
ax.scatter(dfAccel['y'], dfAccel['z'], c=dfAccel['cluster_center'], cmap='Set2_r')
plt.xlim([-1.2,1.2])
plt.ylim([-1.2,1.2])
# plt.show()
plt.savefig(pathFig+'allPositions_YZ.png')

plt.close('all')

#%%
# plot raw accel points colored by position
plt.rcParams.update({'font.size': 39})
# XZ
fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
ax = fig.add_subplot()
positions = dfAccel['position'].unique()
colors = {positions[0]:'tab:green', 
          positions[1]:'tab:purple', 
          positions[2]:'tab:blue', 
          positions[3]:'tab:orange',
          positions[4]:'tab:pink',
          positions[5]:'tab:red'}
ax.set_xlabel('X')
ax.set_ylabel('Z')
plt.xlim([-1.2,1.2])
plt.ylim([-1.2,1.2])
ax.scatter(dfAccel['x'], dfAccel['z'], c=dfAccel['position'].map(colors), alpha=1, s=100)
# plt.show()
plt.savefig(pathFig+'rawData_allPositions_XZ.png')

#%%
#XY
fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
ax = fig.add_subplot()
positions = dfAccel['position'].unique()
colors = {positions[0]:'tab:green', 
          positions[1]:'tab:purple', 
          positions[2]:'tab:blue', 
          positions[3]:'tab:orange',
          positions[4]:'tab:pink',
          positions[5]:'tab:red'}
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.xlim([-1.2,1.2])
plt.ylim([-1.2,1.2])
ax.scatter(dfAccel['x'], dfAccel['y'], c=dfAccel['position'].map(colors), alpha=1)
# plt.show()
plt.savefig(pathFig+'rawData_allPositions_XY.png')

#YZ
fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
ax = fig.add_subplot()
positions = dfAccel['position'].unique()
colors = {positions[0]:'tab:green', 
          positions[1]:'tab:purple', 
          positions[2]:'tab:blue', 
          positions[3]:'tab:orange',
          positions[4]:'tab:pink',
          positions[5]:'tab:red'}
ax.set_xlabel('Y')
ax.set_ylabel('Z')
plt.xlim([-1.2,1.2])
plt.ylim([-1.2,1.2])
ax.scatter(dfAccel['y'], dfAccel['z'], c=dfAccel['position'].map(colors), alpha=1)
# plt.show()
plt.savefig(pathFig+'rawData_allPositions_YZ.png')

# %%
