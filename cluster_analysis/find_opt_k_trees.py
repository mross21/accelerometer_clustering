#%% 
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
from math import cos, sin, asin, sqrt

pathIn = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/kde_sampled_points/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/optimize_k/'
file1000 = 'coords_with_KDEdensities_bw01-1000pts.csv'

df = pd.read_csv(pathIn + file1000, index_col=False)

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
    d = (sin(lat * 0.5)**2) + (cos(lat1) * cos(lat2) * (sin(lng * 0.5)**2))
    return 2  * asin(sqrt(d))

def find_optK(distance_matrix,density_list,nNeighbors):
    densityThresh = max(density_list)/10 # 0.3
    num_clusters = 0
    for i in range(len(distance_matrix)):
        # sort distances by ascending order
        dmSort = distance_matrix[i].sort_values()
        # get list of indices of idx point and 10 closest points
        idxClosePts = dmSort[0:nNeighbors].index 
        # get corresponding densities of those points
        densities = density_list.iloc[idxClosePts]
        # if idx point has largest density [and density > 1/10 max density], add cluster
        add_clust = np.where((max(densities) == densities.iloc[0]) & (densities.iloc[0] >= densityThresh), 1, 0)
        num_clusters = num_clusters + add_clust
    return(num_clusters)

def density_tree(distance_matrix,density_list,nNeighbors):
    tree_nodes = {}
    tree_densities = {}
    for i in range(len(distance_matrix)):
        # print(i)
        iIter = i
        node_list = []
        # sort distances by ascending order
        dmSort = distance_matrix[iIter].sort_values()
        # get list of indices of idx point and n closest points
        idxClosePts = dmSort[0:nNeighbors].index 
        # get corresponding densities of those points
        densities = density_list.iloc[idxClosePts]
        node_list.append(iIter)
        edges = 0
        # while max density in subset is not equal to point i
        while max(densities) != density_list.iloc[iIter]:
            # get max density of point and neighbors
            maxD = max(densities)
            # if no densities, return empty dictionaries
            if np.isnan(maxD) == True:
                # print(iIter)
                # print(density_list.iloc[idxClosePts])
                return(tree_nodes, tree_densities)
            # print(maxD)
            # change index to that of max density
            iIter = list(density_list).index(maxD)
            # print(iIter)
            # append index of node with highest density
            node_list.append(iIter)
            # print(node_list)
            # sort density values of row iIter
            dmSort2 = distance_matrix[iIter].sort_values()
            # find neighbors to point iIter
            idxClosePts2 = dmSort2[0:nNeighbors].index 
            # find densities of neighbors to point iIter
            densities = density_list.iloc[idxClosePts2]
            # add edge
            edges += 1
            # print('edges: ' + str(edges))

        tree_nodes[i] = node_list
        tree_densities[i] = list(density_list.iloc[node_list])
    return(tree_nodes, tree_densities)

def flatten(l):
    return [item for sublist in l for item in sublist]

###############################################################################
# variable to group user's data
grouping = 'weekNumber'
# number of nearest neighbors to compare densities to
n = 9
treeNodes = {}
treeDensity = {}
dfByGroup = df.groupby(['userID', grouping])
for userGrp, grp in dfByGroup:
    # reset indexing
    grp = grp.reset_index()
    user = userGrp[0]
    print('user: ' + str(user))
    groupedBy = userGrp[1]
    print(str(grouping) + str(groupedBy))
    userGrp = ';'.join([str(user),str(groupedBy)])

    # if user > 4:
    #     break

    # get distance matrix of haversine distances between points
    dm = pd.DataFrame(squareform(pdist(grp[['theta','phi']], metric=haversine_dist)), index=grp.index, columns=grp.index)
    # make tree for user/grouping group 
    # dictionary key is 'user;grouping' pair
    treeNodes[userGrp], treeDensity[userGrp] = density_tree(dm, grp['density'],n)
    print('=====')
#%%
# find number of clusters (get local max indices only)
kList = []
for i in treeNodes: # iterate through the keys (i is key)
    allK = []
    print(i)
    for j in treeNodes[i].items(): # iterate through lists for each point j
        print(j)
        allK.append(j[1][-1]) # append last point in list (index of local max density)
    uniqueK = np.unique(np.array(allK)) # find unique indices of local max densities
    kList.append((i,uniqueK)) # indices of local maxima for each user/group pair
#%%
# make dataframe of local maximums for each user/group pairing
dictK = dict(kList)
dfK = pd.DataFrame([(k, y) for k, v in dictK.items() for y in v], columns = ['userGroup','localMaxIndex'])
xList = []
yList = []
zList = []
thetaList = []
phiList = []
densityList = []
# create column to match user/group pairing
df['userGroup'] = df[['userID','weekNumber']].astype(str).agg(';'.join, axis=1)
# loop through the user/group pairs
for usrGrp in dfK['userGroup'].unique():
    # select rows for user/group pair
    grp2 = df.loc[df['userGroup'] == usrGrp].reset_index(drop=True)
    grpK = dfK.loc[dfK['userGroup'] == usrGrp].reset_index(drop=True)
    # get values
    xList.append(grp2['x'].iloc[grpK['localMaxIndex']].reset_index(drop=True))
    yList.append(grp2['y'].iloc[grpK['localMaxIndex']].reset_index(drop=True))
    zList.append(grp2['z'].iloc[grpK['localMaxIndex']].reset_index(drop=True))
    thetaList.append(grp2['theta'].iloc[grpK['localMaxIndex']].reset_index(drop=True))
    phiList.append(grp2['phi'].iloc[grpK['localMaxIndex']].reset_index(drop=True))
    densityList.append(grp2['density'].iloc[grpK['localMaxIndex']].reset_index(drop=True))

dfK['x'] = flatten(xList)
dfK['y'] = flatten(yList)
dfK['z'] = flatten(zList)
dfK['theta'] = flatten(thetaList)
dfK['phi'] = flatten(phiList)
dfK['density'] = flatten(densityList)

# filter out peaks below density 0.2
dfKOut = dfK.loc[dfK['density'] >= 0.2]
# csv file of local max densities
dfKOut.to_csv(pathOut+'tree_nodes_v4.csv', index=False)

print('finish')





#%%
################## 
## TO PLOT


grp2 = df.loc[(df['userID'] == 1) & (df['weekNumber'] == 4)].reset_index()
usrGrp = ';'.join([str(1.0),str(4.0)])
dfK2 = dfKOut.loc[dfKOut['userGroup'] == usrGrp].reset_index()

# flag colors to highlight
grp2['color']=0
for r in grp2.index:
    if r in list(dfK2['localMaxIndex']): #idxClosePts: #tree_nodes[i]: 
        print(r)
        grp2['color'].iloc[r] = 1
print(grp2.loc[grp2['color'] != 0][['phi','theta','density']])

# plot
import matplotlib.pyplot as plt
ax = plt.axes(projection='3d')
ax.scatter(grp2.x, grp2.y, grp2.z, c=grp2.color)
# fig = plt.figure()
# plt.scatter(grp2.y,grp2.z,c=grp2.color)
# plt.show()


#%%

### TESTING THE NEAREST NEIGHBORS

# grp2 = df.loc[(df['userID'] == 1) & (df['weekNumber'] == 4)].reset_index()
# dm2 = pd.DataFrame(squareform(pdist(grp2[['theta','phi']], metric=haversine_dist)), index=grp2.index, columns=grp2.index)

# n = 9
# nNeighbors= n 
# # numK = find_optKtree(dm, grp['density'],n)
# distance_matrix = dm2
# density_list=grp2['density']
# # densityThresh = 0.3 #max(density_list)/10
# densityThresh=0 # no threshold for now
# num_clusters = 0


# ### might not be able to do for loop? or add in another wandering search? or google python code for trees?

# # https://www.delftstack.com/howto/python/trees-in-python/

# tree_nodes = {}
# tree_densities = {}
# for i in range(len(distance_matrix)):
#     print(i)
#     iIter = i
#     node_list = []
#     # sort distances by ascending order
#     dmSort = distance_matrix[iIter].sort_values()
#     # get list of indices of idx point and n closest points
#     idxClosePts = dmSort[0:nNeighbors].index 
#     # get corresponding densities of those points
#     densities = density_list.iloc[idxClosePts]
#     node_list.append(iIter)
#     edges = 0
#     # while max density in subset is not equal to point i
#     while max(densities) != density_list.iloc[iIter]:
#         maxD = max(densities)
#         # print(maxD)
#         iIter = list(density_list).index(maxD)
#         # print(iIter)
#         node_list.append(iIter)
#         # print(node_list)
#         dmSort2 = distance_matrix[iIter].sort_values()
#         idxClosePts2 = dmSort2[0:nNeighbors].index 
#         densities = density_list.iloc[idxClosePts2]
#         edges += 1
#         # print('edges: ' + str(edges))

#     tree_nodes[i] = node_list
#     tree_densities[i] = list(density_list.iloc[node_list])
#     # return(tree_nodes, tree_densities)

    


#     # if idx point has largest density [and density > 1/10 max density], add cluster
#     # add_clust = np.where((max(densities) == densities.iloc[0]) & (densities.iloc[0] >= densityThresh), 1, 0)
#     # num_clusters = num_clusters + add_clust

# print(tree_nodes)
# print(tree_densities)

#%%
# grp2 = df.loc[(df['userID'] == 1) & (df['weekNumber'] == 1)].reset_index()
# dm2 = pd.DataFrame(squareform(pdist(grp2[['theta','phi']], metric=haversine_dist)), index=grp2.index, columns=grp2.index)

# i = 990
# n = 9
# # make list of indices of nearest neighbors to index i
# print(dm2[i].sort_values()[0:n])
# l=list(dm2[i].sort_values()[0:n].index)

# # flag colors to highlight
# grp2['color']=0
# for r in grp2.index:
#     if r in l: 
#         grp2['color'].iloc[r] = r/10
# print(grp2.loc[grp2['color'] != 0][['phi','theta','density']])

# # plot
# import matplotlib.pyplot as plt
# # ax = plt.axes(projection='3d')
# # ax.scatter(grp2.x, grp2.y, grp2.z, c=grp2.color)
# plt.scatter(grp2.x,grp2.y,c=grp2.color)
# plt.show()













#%%

# find and compare neighboring points
# num_clusters = 0
# for idx in range(len(dm)):
    # # sort distances by ascending order
    # dmSort = dm[idx].sort_values()
    # # get list of indices of idx point and 10 closest points
    # idxClosePts = dmSort[0:75].index 
    # # get corresponding densities of those points
    # densities = df['density'].iloc[idxClosePts]
    # # if idx point has largest density, add cluster
    # add_clust = np.where(max(densities) == densities.iloc[0], 1, 0)
    # num_clusters = num_clusters + add_clust

###############################################################################################
# NOT COMPLETE OR NOT WORKING


# def great_circle_distance(r, coord1, coord2):
#     # phi ~ longitude
#     # theta ~ latitude


#     return r * acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2))

# [great_circle_distance(*combo) for combo in combinations_with_replacement(list_of_coords,2)]

# # create combinations of indices of the coordinates
# combos = list(combinations(df.index, 2))
# for idx in len(combos):
#     # df row index of points to compare
#     iPt1 = combos[idx][0]
#     iPt2 = combos[idx][1]
#     # get phi/theta point for each index
#     pt1 = tuple(df[['phi','theta']].iloc[iPt1]) # as (phi, theta)
#     pt2 = tuple(df[['phi','theta']].iloc[iPt2]) # as (phi, theta)
#     # find distance between points
#     gcdist = great_circle_distance(1,pt1,pt2)




# # compare point 1 to all other points in list
# from itertools import repeat

# pts = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12)]

# [distance(*pair) for pair in zip(repeat(pts[0]),pts[1:])]

# %%
