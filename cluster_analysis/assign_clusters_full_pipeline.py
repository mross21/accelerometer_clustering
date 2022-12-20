#%%
# import packages
import pandas as pd
from pyarrow import parquet
import numpy as np
import re
import glob
import spherical_kde
from math import cos, sin, asin, sqrt
from scipy import spatial
from scipy.spatial.distance import squareform, pdist
import networkx as nx
from matplotlib import pyplot as plt
from itertools import count
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
pathAccel = '/home/mindy/Desktop/BiAffect-iOS/data_processing/processed_outputs/accel/'
# folder where figures should be saved
pathFig = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/graph_matrices/figures/open_science/'
# folder where output accelerometer data should be saved
pathAccOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/accel_with_clusters/open_science/'

# list of user accel files
all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)

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

# make file for user/week/number of clusters
k_list = []
dfOut = []

for file in all_files:
    # read in user's accelerometer file
    dfAllAccel = pd.read_parquet(file, engine='pyarrow')
    user = dfAllAccel['userID'].iloc[0]

    # filter accel points to be on unit sphere:
    dfAccel = accel_filter(dfAllAccel)
    # skip past android data (r > 1 for android data)
    if len(dfAccel) < 2: 
        continue
    # convert cartesian coordinates to spherical
    addSpherCoords(dfAccel)

    # for saving updated accelerometer file
    accOut = []
    # create KDE per user and week
    dfAccByWk = dfAccel.groupby(['userID', 'weekNumber'])
    for userAndWk, accGrp in dfAccByWk:
        wk = userAndWk[1]
        print('user: ' + str(user)) # user
        print('week: ' + str(wk))  # week number

        # sometimes my kernal would crash so if group size too large, remove every 4th row
        # while len(accGrp) > 700000: # change value to be as large as possible
        #     print('group size too large')
        #     print(len(accGrp))
        #     accGrp = accGrp[np.mod(np.arange(accGrp.index.size),4)!=0]
        print('accel group length: ' + str(len(accGrp)))

        # if group has no data, skip group
        if len(accGrp) < 2:
            continue

        # calculate spherical KDE
        try:
            sKDE = spherical_kde.SphericalKDE(accGrp['phi'], accGrp['theta'], weights=None, bandwidth=0.1)
        except ValueError:
            continue
        # sample KDE at equidistant points to get densities
        sKDE_densities = np.exp(sKDE(equi_phi, equi_theta))
        # dataframe of points and densities
        KDEdensities = np.column_stack([[userAndWk[0]]*len(equi_phi), [userAndWk[1]]*len(equi_phi),
                                    x,y,z,equi_phi,equi_theta,sKDE_densities])
        grpKDE = pd.DataFrame(KDEdensities, columns = ['userID','weekNumber','z', 'x', 'y', 'phi', 'theta', 'density'])
        
        # for csv file of all KDE points
        if len(accGrp) > 10000:
            dfOut.append(grpKDE)

        # print('finished making KDE')

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

        # append user/week/number of clusters
        k_list.append((user, wk, k))

        # print('finished finding cluster centers')
    
### ######################################################################################################
# assign cluster IDs to raw accelerometer data

        # # sigma alters weight function
        # sigma = 1
        # matrix of all edge weights
        # edge weights based on density- higher density between the two nodes, the lower the weight (invert density)
        edge_weights = pd.DataFrame(squareform(pdist(grpKDE[['density']], lambda u,v: (np.exp(-((u+v)/2))))), #1/((u+v)/2))**sigma
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
                # tie_break_min_path += 1
            

            # count += 1
            # min_elements = min(dfPaths['n_elements'])
            # min_path = dfPaths.loc[dfPaths['n_elements'] == min_elements]
            # # if multiple with same path length, choose cluster index closest to pt index
            # if len(min_path) > 1:
            #     tie_break_min_path += 1
            #     min_path['cluster_idx_dist'] = abs(min_path['cluster_idx'] - i)
            #     min_path = min_path.loc[min_path['cluster_idx_dist'] == min(min_path['cluster_idx_dist'])]
            # # if multiple with same index distance, choose cluster index with smaller corresponding weight
            # if len(min_path) > 1:
            #     tie_break_min_path2 += 1
            #     min_path = min_path.loc[min_path['weight'] == min(min_path['weight'])]


            closest_cluster_idx = int(min_path['cluster_idx'])
            # append every point on sphere and assigned cluster center to dictionary
            # match the sphere index with the cluster center/XYZ point
            dictClust[i] = dictID[closest_cluster_idx]
            dictClustIdx[i] = closest_cluster_idx
            dictClustX[i] = grpKDE.loc[grpKDE.index == closest_cluster_idx]['x'].iloc[0]
            dictClustY[i] = grpKDE.loc[grpKDE.index == closest_cluster_idx]['y'].iloc[0]
            dictClustZ[i] = grpKDE.loc[grpKDE.index == closest_cluster_idx]['z'].iloc[0]
        # print('finished making dictionary of equidistant sphere points and linked cluster center')
################################################################
        # # plot network
        # fig = plt.figure(1, figsize=(20, 20))
        # nx.draw_networkx(G, node_size=10, with_labels=False, node_color = colors)
        # plt.show()

        # # set nodes attributes
        # nx.set_node_attributes(G, dictClustIdx,"cluster")
        # # get unique groups
        # groups = set(nx.get_node_attributes(G,'cluster').values())
        # mapping = dict(zip(sorted(groups),count()))
        # nodes = G.nodes()
        # colors = [mapping[G.nodes[n]['cluster']] for n in nodes]

        # # plot graph
        # fig = plt.figure(1, figsize=(20, 20))
        # nx.draw_networkx(G, node_size=10, with_labels=False, node_color = colors)
        # plt.show()
### ######################################################################################################
# map cluster centers to raw accelerometer data 
        # set XYZ to float
        accGrp['x'] = accGrp['x'].astype(float)
        accGrp['y'] = accGrp['y'].astype(float)
        accGrp['z'] = accGrp['z'].astype(float)
        # find nearest equidistant sphere point for each raw accel point
        accGrp['nodeIdx'] = nearest_neighbour(accGrp[['x','y','z']], grpKDE[['x','y','z']])


        # # if KDE point has too small of density, delete point and raw accel points near it
        # KDE_threshold = max(grpKDE['density'])/100 
        # grpKDE_filter = grpKDE.loc[grpKDE['density'] > KDE_threshold]
        # accGrp_filter = accGrp[accGrp['nodeIdx'].isin(grpKDE_filter.index)]

        accGrp_filter = accGrp

        # map cluster center to each raw accel point using the matched sphere point
        accGrp_filter['cluster_center'] = accGrp_filter['nodeIdx'].map(dictClust)
        accGrp_filter['cluster_center_idx'] = accGrp_filter['nodeIdx'].map(dictClustIdx)
        accGrp_filter['cluster_center_x'] = accGrp_filter['nodeIdx'].map(dictClustX)
        accGrp_filter['cluster_center_y'] = accGrp_filter['nodeIdx'].map(dictClustY)
        accGrp_filter['cluster_center_z'] = accGrp_filter['nodeIdx'].map(dictClustZ)
        # append updated accel group
        accOut.append(accGrp_filter)
        # print('finished adding cluster labels to accelerometer data')

## ######################################################################################################
# XZ plot of labeled accelerometer data
        plt.rcParams.update({'font.size': 32})
        groupAcc_filter=accGrp_filter
        fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
        ax = fig.add_subplot()
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.scatter(groupAcc_filter['x'], groupAcc_filter['z'], c=groupAcc_filter['cluster_center'], cmap='Set2')
        # ax.scatter(cluster_x,cluster_z, c='red')
        # for i in range(len(cluster_idx)):
        #     plt.text(cluster_x[i],cluster_z[i],str(list(dictID.values())[i]), color="red", fontsize=16)
        plt.xlim([-1.2,1.2])
        plt.ylim([-1.2,1.2])
        plt.show()
        # plt.savefig(pathFig+'user_' + str(int(user)) + '_week_' + str(int(wk)) + '_plot_by_graphDistance-xz-v2.png')

        # XY plot
        fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
        ax = fig.add_subplot()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.scatter(groupAcc_filter['x'], groupAcc_filter['y'], c=groupAcc_filter['cluster_center'], cmap='Set2')
        # ax.scatter(cluster_x,cluster_y, c='red')
        # for i in range(len(cluster_idx)):
        #     plt.text(cluster_x[i],cluster_y[i],str(list(dictID.values())[i]), color="red", fontsize=16)
        plt.xlim([-1.2,1.2])
        plt.ylim([-1.2,1.2])
        plt.show()
        # plt.savefig(pathFig+'user_' + str(int(user)) + '_week_' + str(int(wk)) + '_plot_by_graphDistance-xy-v2.png')

        # YZ plot
        fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
        ax = fig.add_subplot()
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.scatter(groupAcc_filter['y'], groupAcc_filter['z'], c=groupAcc_filter['cluster_center'], cmap='Set2')
        # ax.scatter(cluster_y,cluster_z, c='red')
        # for i in range(len(cluster_idx)):
        #     plt.text(cluster_y[i],cluster_z[i],str(list(dictID.values())[i]), color="red", fontsize=16)
        plt.xlim([-1.2,1.2])
        plt.ylim([-1.2,1.2])
        plt.show()
        # plt.savefig(pathFig+'user_' + str(int(user)) + '_week_' + str(int(wk)) + '_plot_by_graphDistance-yz-v2.png')

#         plt.close('all')

### ######################################################################################################
# save KDE points
    # if len(dfOut) < 1:
    #     continue
    # dfOut2 = pd.concat(dfOut,axis=0,ignore_index=True)
    # dfOut2.to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/KL/openScience_coords_KDEdensities_bw01-'+str(num)+'pts-v2.csv', index=False)

# # save updated accel data to csv
#     if len(accOut) < 1:
#         continue
#     dfAccOut = pd.concat(accOut,axis=0,ignore_index=True)
#     # dfAccOut.to_csv(pathAccOut + 'user_'+str(int(user))+'_accel_withClusters.csv', index=False)
#     dfAccOut.to_parquet(pathAccOut + 'user_'+str(int(user))+'_accel_withClusters-v2.parquet')
#     # print('finished saving updated accelerometer data')
#     # print('=========================================================================================')

#     # save file for number of clusters for every user/week
#     dfK = pd.DataFrame(k_list, columns = ['userID','weekNumber','k'])
#     dfK.to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/kmeans/k_list-v2.csv', index=False)

print('finished with everything')

#%%