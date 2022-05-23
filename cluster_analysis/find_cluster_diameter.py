#%%

import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
import math
from numpy.linalg import norm



pathIn = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/kde_sampled_points/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/optimize_k/'
file1000 = 'coords_with_KDEdensities-1000pts.csv'

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
    d = (math.sin(lat * 0.5)**2) + (math.cos(lat1) * math.cos(lat2) * (math.sin(lng * 0.5)**2))
    return 2  * math.asin(math.sqrt(d))

def cosine_sim(pt1, pt2):
    A = pt1 #pd.to_numeric(np.squeeze(np.asarray(pt1)))
    B = pt2 #pd.to_numeric(np.squeeze(np.asarray(pt2)))
    cos = np.dot(A,B)/(norm(A)*norm(B))
    return(cos)

#%%
grp = df.loc[(df['userID'] == 6) & (df['weekNumber'] == 2)].reset_index()

dm = pd.DataFrame(squareform(pdist(grp[['theta','phi']], metric=haversine_dist)), index=grp.index, columns=grp.index)

idx=600
distance_matrix=dm
nNeighbors = 10

dmSort = distance_matrix[idx].sort_values()
# get list of indices of idx point and n closest points
idxClosePts = dmSort[0:nNeighbors].index 

nearPts = grp[['x','y','z']].iloc[idxClosePts]
print(nearPts.iloc[0])
print(nearPts.iloc[-1])
cosSim = cosine_sim(nearPts.iloc[0],nearPts.iloc[-1]) # cosine similarity needs XYZ
print(cosSim)
ang = math.degrees(math.acos(cosSim))
print(ang)

#%%

idxClosePts = [idxClosePts[0],idxClosePts[-1]]

# flag colors to highlight
grp['color']=0
for r in grp.index:
    if r in idxClosePts: #tree_nodes[i]: 
        grp['color'].iloc[r] = 1
print(grp.loc[grp['color'] != 0][['phi','theta','density']])

# plot
import matplotlib.pyplot as plt
ax = plt.axes(projection='3d')
ax.scatter(grp.x, grp.y, grp.z, c=grp.color)
# fig = plt.figure()
# plt.scatter(grp2.x,grp2.y,c=grp2.color)
# plt.show()


# %%
