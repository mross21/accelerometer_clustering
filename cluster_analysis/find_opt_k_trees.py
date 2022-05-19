#%% 
import pandas as pd
# from itertools import combinations
# from sklearn.metrics.pairwise import haversine_distances
import numpy as np
from scipy.spatial.distance import squareform, pdist
# from haversine import haversine
from math import cos, sin, asin, sqrt


pathIn = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/kde_sampled_points/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/optimize_k/'
# file2500 = 'coords_with_KDEdensities-2500pts.csv'
file1000 = 'coords_with_KDEdensities-1000pts.csv'
# file500 = 'coords_with_KDEdensities-500pts.csv'
# file1000_v2 = 'coords_with_KDEdensities_bw02-1000pts.csv'
# file1000_v3 = 'coords_with_KDEdensities_bw015-1000pts.csv'

df = pd.read_csv(pathIn + file1000, index_col=False)

# taken from haversine package (removed conversion to radians)
# https://github.com/mapado/haversine/blob/main/haversine/haversine.py
def haversine_dist(pt1,pt2): # theta, phi
    lat1, lng1 = pt1
    lat2, lng2 = pt2
    # convert theta range from 0 to 2pi to -pi to pi
    lat1 = lat1 - np.pi
    lat2 = lat2 - np.pi
    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = (sin(lat * 0.5)**2) + (cos(lat1) * cos(lat2) * (sin(lng * 0.5)**2))
    return 2  * asin(sqrt(d))

def find_optK(distance_matrix,density_list,nNeighbors):
    densityThresh = 0.3 #max(density_list)/10
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

#%%
# # subset to user 1 week 1
# df = df.loc[(df['userID'] == 1) & (df['weekNumber'] == 1)]

# # variable to group user's data
# grouping = 'weekNumber'
# # number of nearest neighbors to compare densities to
# nPts = [20,30,40,50,60,70,80,90,100,110,120,130]
# kList = []
# dfByGroup = df.groupby(['userID', grouping])
# for userGrp, grp in dfByGroup:
#         # reset indexing
#     grp = grp.reset_index()
#     user = userGrp[0]
#     print('user: ' + str(user))
#     groupedBy = userGrp[1]
#     print(str(grouping) + str(groupedBy))
#     if user > 1:
#         break
#     for n in nPts:
#         print('n: ' + str(n))
#         # get distance matrix of haversine distances between points
#         dm = pd.DataFrame(squareform(pdist(grp[['theta','phi']], metric=haversine_dist)), index=grp.index, columns=grp.index)
#         # get number of clusters for grouping
#         numK = find_optK(dm, grp['density'],n)
#         kList.append((user,groupedBy,n,numK))
#         print('clusters: ' + str(numK))
#     print('=====')

# dfK = pd.DataFrame(kList, columns = ['userID', grouping, 'n_neighbors','k'])
# dfK.to_csv(pathOut + 'test_parameters_for_optK_1000pts_KDEbw01_sample03DensityThresh.csv', index=False)

# print('finish')
#%%

### TESTING THE NEAREST NEIGHBORS

grp2 = df.loc[(df['userID'] == 1) & (df['weekNumber'] == 4)].reset_index()
dm2 = pd.DataFrame(squareform(pdist(grp2[['theta','phi']], metric=haversine_dist)), index=grp2.index, columns=grp2.index)

#%%
i = 990
n = 9
# make list of indices of nearest neighbors to index i
print(dm2[i].sort_values()[0:n])
l=list(dm2[i].sort_values()[0:n].index)

# flag colors to highlight
grp2['color']=0
for r in grp2.index:
    if r in l: 
        grp2['color'].iloc[r] = r/10
print(grp2.loc[grp2['color'] != 0][['phi','theta','density']])

# plot
import matplotlib.pyplot as plt
# ax = plt.axes(projection='3d')
# ax.scatter(grp2.x, grp2.y, grp2.z, c=grp2.color)
plt.scatter(grp2.x,grp2.y,c=grp2.color)
plt.show()













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
