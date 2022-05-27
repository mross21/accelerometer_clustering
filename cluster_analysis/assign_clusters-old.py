#%%
import pandas as pd
import numpy as np
import re
import glob
from pyarrow import parquet
# from numpy.linalg import norm
from scipy import spatial
import math

cluster_file = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/coords_KDE_clusters.csv'
cluster_means_file = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/cluster_means.csv'
phq_file = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/PHQ/allUsers_PHQdata.csv'
diag_file = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/diagnosis/allUsers_diagnosisData.csv'
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

# def cosine_sim(coord, cmean):
#     A = pd.to_numeric(np.squeeze(np.asarray(coord)))
#     B = pd.to_numeric(np.squeeze(np.asarray(cmean)))
#     cos = np.dot(A,B)/(norm(A)*norm(B))
#     return(cos)

# def find_clust_label(coord, dfCMean, tol):
#     # cos = 1 means vectors overlap (very similar)
#     if (dfCMean.iloc[0]['x'] == 0) & (dfCMean.iloc[0]['y'] == 0) & (dfCMean.iloc[0]['z'] == 0):
#         cID = 0
#         return(cID)
#     dfCMean['cos'] = dfCMean.apply(lambda row: cosine_sim(coord, row[['x','y','z']]), axis=1)
#     if max(dfCMean['cos']) < tol: # if max cos < tol, then coord outside all clusters
#         cID = 0 # no cluster found
#     else:
#         cID = dfCMean.loc[dfCMean['cos'] == max(dfCMean['cos'])]['clustID'].iloc[0] # ID based on max cos
#     return(cID)

def nearest_neighbour(points_a, points_b):
    tree = spatial.cKDTree(points_b)
    return tree.query(points_a)[1]

def flatten(l):
    return [item for sublist in l for item in sublist]

def regular_on_sphere_points(r,num):
    points = []
    #Break out if zero points
    if num==0:
        return points
    a = 4.0 * math.pi*(r**2.0 / num)
    d = math.sqrt(a)
    m_theta = int(round(math.pi / d))
    d_theta = math.pi / m_theta
    d_phi = a / d_theta

    for m in range(0,m_theta):
        theta = math.pi * (m + 0.5) / m_theta
        m_phi = int(round(2.0 * math.pi * math.sin(theta) / d_phi))
        for n in range(0,m_phi):
            phi = 2.0 * math.pi * n / m_phi
            x = r * math.sin(theta) * math.cos(phi)
            y = r * math.sin(theta) * math.sin(phi)
            z = r * math.cos(theta)
            points.append([x,y,z])
    return points


dfClusters = pd.read_csv(cluster_file, index_col=False)
dfMeans = pd.read_csv(cluster_means_file, index_col=False)
dfPHQ = pd.read_csv(phq_file, index_col=False)
dfDiag = pd.read_csv(diag_file, index_col=False)

# make equidistant points on sphere to sample
radius = 1
num = 2500
regular_surf_points = regular_on_sphere_points(radius,num)
pts_xyz=np.array(regular_surf_points)
x=pts_xyz[:,0]
y=pts_xyz[:,1]
z=pts_xyz[:,2]
equi_phi = np.ndarray.round(np.mod(np.arctan2(y, x), np.pi*2),2) 
equi_theta = np.ndarray.round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2) 
equi_points = pd.DataFrame(np.stack((equi_phi, equi_theta), axis=-1), columns = ['phi','theta'])

all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)

for file in all_files:
    # dfAccel = pd.read_parquet(file, engine='pyarrow')
    dfAccel = pd.read_parquet(file, engine='pyarrow')
    # filter accel points to be on unit sphere:
    dfAccel = accel_filter(dfAccel)
    # convert cartesian coordinates to spherical
    addSpherCoords(dfAccel)

    clust_list = []
    dfByUser = dfAccel.groupby(['userID', 'weekNumber'])
    for userAndWk, group in dfByUser:
        print('user: ' + str(userAndWk[0])) # user
        print('week: ' + str(userAndWk[1]))  # week number for that user
        # if group['userID'].iloc[0] == 2:
        #     break

        clustGrp = dfClusters.loc[(dfClusters['userID'] == userAndWk[0]) & (dfClusters['weekNumber'] == userAndWk[1])]
        clustIDs = pd.merge(equi_points, clustGrp[['phi','theta','cluster']], on= ['phi','theta'], how='outer').fillna(0)
        nn = nearest_neighbour(group[['phi','theta']],clustIDs[['phi','theta']])
        clust_labs = clustIDs['cluster'].iloc[nn-1]
        clust_list.append(clust_labs)


#################
# the kde coords list doesnt have all of the phi theta, only the ones w/ density > 0.2
# but looks like the nearest neighbor function works
#################



        # for i in range(len(group)): # loop through coordinates in user's week
        #     print('user: ' + str(userAndWk[0])) # user
        #     print('week: ' + str(userAndWk[1]))  # week number for that user
        #     print('index: ' + str(i) + '/' + str(len(group)))
        #     print('---------')
        #     xyz = group[['x','y','z']].iloc[i]
        #     clustGrp = dfMeans.loc[(dfMeans['userID'] == userAndWk[0]) & (dfMeans['weekNumber'] == userAndWk[1])]
        #     if len(clustGrp) == 0:
        #         print('len cluster means is 0')
        #         print('user: ' + str(userAndWk[0])) # user
        #         print('week: ' + str(userAndWk[1]))
        #         cID = 0
        #         continue
        #     cID = find_clust_label(xyz, clustGrp, 0.8) # what should the tolerance be?
        #     clust_list.append(cID)


    dfAccel['cluster'] = flatten(clust_list)
    dfAccel.to_csv(pathOut + 'user_' + str(userAndWk[0]) + '_accelData_clusters.csv', index=False)


print('finish')
# %%
