#%%
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
from math import cos, sin, asin, sqrt

pathIn = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/kde_sampled_points/'
# pathOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/optimize_k/'
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


# variable to group user's data
grouping = 'weekNumber'
# number of nearest neighbors to compare densities to
# n = 9

dfByGroup = df.groupby(['userID', grouping])
for userGrp, grp in dfByGroup:
    # reset indexing
    grp = grp.reset_index()
    user = userGrp[0]
    print('user: ' + str(user))
    groupedBy = userGrp[1]
    print(str(grouping) + str(groupedBy))
    userGrp = ';'.join([str(user),str(groupedBy)])

    # get distance matrix of haversine distances between points
    dm = pd.DataFrame(squareform(pdist(grp[['theta','phi']], metric=haversine_dist)), index=grp.index, columns=grp.index)
    
    # if distance is less than 0.16, flag as adjacent point
    adjMatrix = np.where(dm < 0.16, 1, 0)

    break










# %%
