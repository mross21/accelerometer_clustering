#%%
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
import math
from numpy.linalg import norm
import re
import glob
from scipy import spatial
import spherical_kde

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

#%%
treeNodesCSV = pd.read_csv(treeFile, index_col=False)
treeNodes = pd.DataFrame(treeNodesCSV)

# make equidistant points on sphere to sample
radius = 1
num = 5000
regular_surf_points = regular_on_sphere_points(radius,num)
pts_xyz=np.array(regular_surf_points)
x=pts_xyz[:,0]
y=pts_xyz[:,1]
z=pts_xyz[:,2]
equi_phi = np.ndarray.round(np.mod(np.arctan2(y, x), np.pi*2),2) 
equi_theta = np.ndarray.round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2) 

# column to split user data by
grouping = 'weekNumber'

all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)
for file in all_files:
    # dfAccel = pd.read_parquet(file, engine='pyarrow')
    dfAccel = pd.read_parquet(file, engine='pyarrow')
    
    if dfAccel['userID'].iloc[0] < 11:
        continue

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
        print(len(group))


        # if group size too large, remove every 4th row
        while len(group) > 150000:
            print('group size above 185000')
            print(len(group))
            group = group[np.mod(np.arange(group.index.size),4)!=0]
        print('length group: ' + str(len(group)))


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
        

        theta_samples = group['theta']
        phi_samples = group['phi']
        sKDE = spherical_kde.SphericalKDE(phi_samples, theta_samples, weights=None, bandwidth=0.1, density=50)           
        equiPts = pd.DataFrame(np.column_stack((equi_phi,equi_theta)), columns=['phi','theta'])
        equiPts['density'] = np.exp(sKDE(equiPts['phi'], equiPts['theta']))
        neighborEquiPtIdx = nearest_neighbour(group[['phi','theta']],equiPts[['phi','theta']])
        group['density'] = equiPts['density'].iloc[neighborEquiPtIdx].reset_index(drop=True) # get phi value of cluster center closest to group['phi']


        # if coordinate is close to cluster center (~10 nearest neighbors, 25 degrees) then label cluster by #, else 0
        # 0 means no cluster
        # group['cluster'] = np.where(group['cosine_similarity'] >= 0.975, group['neighborIdx']+1, 0)

        group['cluster'] = np.where((group['density'] >= 0.25) | (group['cosine_similarity'] >= 0.975), group['neighborIdx']+1, 0)



        ##################################
        # add in something about if cosine_similarity too low, then unlabel cluster
        # cosine similarity of 0.866 -> 30 degrees
        group.loc[group['cosine_similarity'] < 0.866, ['cluster']] = 0




        ##################################


        # clust_list.append(group['cluster'])
        dfOut.append(group)
        
    
    dfOut = pd.concat(dfOut,axis=0,ignore_index=True)

    # df.to_csv(pathOut + 'user_' + str(userAndGrp[0]) + '_accelData_clusters.csv', index=False)
    dfOut.to_csv(pathOut + 'user_' + str(userAndGrp[0]) + '_accelData_clusters_v3.csv', index=False)

print('finish')

# %%
group[['index','healthCode','recordId','weekNumber','x','y','z','xN','yN','zN','neighborIdx','cosine_similarity','cluster']].to_csv('/home/mindy/Desktop/test.csv', index=False)
# %%
group[['index','healthCode','recordId','weekNumber','x','y','z','neighborIdx','density','cluster']].to_csv('/home/mindy/Desktop/test.csv', index=False)

# %%
