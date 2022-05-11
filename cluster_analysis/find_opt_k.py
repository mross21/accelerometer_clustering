#%% 
import pandas as pd
from itertools import combinations
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
from scipy.spatial.distance import squareform, pdist
from haversine import haversine


pathIn = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/kde_sampled_points/'
file2500 = 'coords_with_KDEdensities-2500pts.csv'
file500 = 'coords_with_KDEdensities-500pts.csv'

df = pd.read_csv(pathIn + file500, index_col=False)

#%%
# subset to user 1 week 1
df = df.loc[(df['userID'] == 1) & (df['weekNumber'] == 1)]

# create combinations of indices of the coordinates
combos = list(combinations(df.index, 2))
for idx in len(combos):
    # df row index of points to compare
    iPt1 = combos[idx][0]
    iPt2 = combos[idx][1]
    # get phi/theta point for each index
    pt1 = tuple(df[['phi','theta']].iloc[iPt1]) # as (phi, theta)
    pt2 = tuple(df[['phi','theta']].iloc[iPt2]) # as (phi, theta)
    # find distance between points
    gcdist = great_circle_distance(1,pt1,pt2)




dm = pd.DataFrame(squareform(pdist(df, metric=haversine)), index=df.index, columns=df.index)



#%%
def great_circle_distance(r, coord1, coord2):
    # phi ~ longitude
    # theta ~ latitude


    return r * acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2))

[great_circle_distance(*combo) for combo in combinations_with_replacement(list_of_coords,2)]






# # compare point 1 to all other points in list
# from itertools import repeat

# pts = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12)]

# [distance(*pair) for pair in zip(repeat(pts[0]),pts[1:])]




























# %%
