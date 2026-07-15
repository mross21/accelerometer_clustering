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
from config import PATH_ACCELEROMETER_PARQUET, PATH_FIGURES
from io_utils import out_path
pd.options.mode.chained_assignment = None

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

def addSpherCoords(xyz):
    x = pd.to_numeric(xyz['z'])
    y = pd.to_numeric(xyz['x'])
    z = pd.to_numeric(xyz['y'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    xyz['phi'] = round(np.mod(np.arctan2(y, x), np.pi*2),2) 
    xyz['theta'] = round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2) 
    return(xyz)

def regular_on_sphere_points(r,num):
    points = []
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

def haversine_dist(pt1,pt2):
    lat1, lng1 = pt1
    lat2, lng2 = pt2
    lat1 = lat1 - (np.pi/2)
    lat2 = lat2 - (np.pi/2)
    lng1 = lng1 - np.pi
    lng2 = lng2 - np.pi
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = (sin(lat * 0.5)**2) + (cos(lat1) * cos(lat2) * (sin(lng * 0.5)**2))
    return 2  * asin(sqrt(d))

def get_cluster_centers(distance_matrix,density_list,nNeighbors,threshold):
    num_clusters = 0
    idx_list = []
    for i in range(len(distance_matrix)):
        dmSort = distance_matrix[i].sort_values()
        idxClosePts = dmSort[0:nNeighbors].index 
        densities = density_list.iloc[idxClosePts]
        add_clust = np.where((max(densities) == densities.iloc[0]) & (densities.iloc[0] >= threshold), 1, 0)
        num_clusters = num_clusters + add_clust
        if add_clust == 1:
            idx_list.append(i)
    return(num_clusters, idx_list)

def nearest_neighbour(points_a, points_b):
    tree = spatial.cKDTree(points_b)
    return tree.query(points_a)[1]


def main():
    all_files = sorted(glob.glob(PATH_ACCELEROMETER_PARQUET + "*.parquet"), key=numericalSort)
    radius = 1
    num = 1000
    regular_surf_points = regular_on_sphere_points(radius, num)
    pts_xyz = np.array(regular_surf_points)
    x = pts_xyz[:, 0]
    y = pts_xyz[:, 1]
    z = pts_xyz[:, 2]
    equi_phi = np.ndarray.round(np.mod(np.arctan2(y, x), np.pi * 2), 2)
    equi_theta = np.ndarray.round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)), 2)
    spherePts = pd.DataFrame(np.column_stack((equi_theta, equi_phi)), columns=['theta', 'phi'])

    dm = pd.DataFrame(squareform(pdist(spherePts, metric=haversine_dist)), index=spherePts.index, columns=spherePts.index)
    d = 0.19
    adjMatrix = np.where(dm < d, 1, 0)
    np.fill_diagonal(adjMatrix, 0)

    for file in all_files:
        dfAllAccel = pd.read_parquet(file, engine='pyarrow')
        dfAccel = accel_filter(dfAllAccel)
        addSpherCoords(dfAccel)

        sKDE = spherical_kde.SphericalKDE(dfAccel['phi'], dfAccel['theta'], weights=None, bandwidth=0.1)
        sKDE_densities = np.exp(sKDE(equi_phi, equi_theta))
        KDEdensities = np.column_stack([x, y, z, equi_phi, equi_theta, sKDE_densities])
        grpKDE = pd.DataFrame(KDEdensities, columns=['z', 'x', 'y', 'phi', 'theta', 'density'])

        n = 9
        b, bins, patches = plt.hist(x=grpKDE['density'], bins=50)
        localMaxThreshold = bins[1]

        clusters = get_cluster_centers(dm, grpKDE['density'], n, localMaxThreshold)
        cluster_idx = clusters[1]

        edge_weights = pd.DataFrame(squareform(pdist(grpKDE[['density']], lambda u, v: (np.exp(-((u + v) / 2))))),
                                    index=grpKDE.index, columns=grpKDE.index)
        adj_weights = edge_weights * adjMatrix
        G = nx.from_numpy_matrix(np.array(adj_weights), parallel_edges=False, create_using=nx.Graph())

        dictClust = {}
        dictClustIdx = {}
        dictClustX = {}
        dictClustY = {}
        dictClustZ = {}
        dictID = dict(zip(cluster_idx, range(1, len(cluster_idx) + 1)))

        for i in range(0, len(adj_weights)):
            path_list = []
            for j in cluster_idx:
                path = nx.single_source_dijkstra(G, i, j, weight='weight')
                num_elements = len(path[1])
                path_list.append((i, j, num_elements, path[0], path[1]))
            dfPaths = pd.DataFrame(path_list, columns=['point_idx', 'cluster_idx', 'n_elements', 'weight', 'path'])

            min_path = dfPaths.loc[dfPaths['weight'] == min(dfPaths['weight'])]
            if len(min_path) > 1:
                print('min path lengths equal')

            closest_cluster_idx = int(min_path['cluster_idx'])
            dictClust[i] = dictID[closest_cluster_idx]
            dictClustIdx[i] = closest_cluster_idx
            dictClustX[i] = grpKDE.loc[grpKDE.index == closest_cluster_idx]['x'].iloc[0]
            dictClustY[i] = grpKDE.loc[grpKDE.index == closest_cluster_idx]['y'].iloc[0]
            dictClustZ[i] = grpKDE.loc[grpKDE.index == closest_cluster_idx]['z'].iloc[0]

        dfAccel['x'] = dfAccel['x'].astype(float)
        dfAccel['y'] = dfAccel['y'].astype(float)
        dfAccel['z'] = dfAccel['z'].astype(float)
        dfAccel['nodeIdx'] = nearest_neighbour(dfAccel[['x', 'y', 'z']], grpKDE[['x', 'y', 'z']])

        dfAccel['cluster_center'] = dfAccel['nodeIdx'].map(dictClust)
        dfAccel['cluster_center_idx'] = dfAccel['nodeIdx'].map(dictClustIdx)
        dfAccel['cluster_center_x'] = dfAccel['nodeIdx'].map(dictClustX)
        dfAccel['cluster_center_y'] = dfAccel['nodeIdx'].map(dictClustY)
        dfAccel['cluster_center_z'] = dfAccel['nodeIdx'].map(dictClustZ)

        plt.rcParams.update({'font.size': 39})
        fig = plt.figure(figsize=(16, 16), facecolor=(1, 1, 1))
        ax = fig.add_subplot()
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.scatter(dfAccel['x'], dfAccel['z'], c=dfAccel['cluster_center'], cmap='Set2_r', s=100)
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.2, 1.2])
        plt.savefig(out_path(PATH_FIGURES, 'allPositions_XZ.png'))

        fig = plt.figure(figsize=(16, 16), facecolor=(1, 1, 1))
        ax = fig.add_subplot()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.scatter(dfAccel['x'], dfAccel['y'], c=dfAccel['cluster_center'], cmap='Set2_r')
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.2, 1.2])
        plt.savefig(out_path(PATH_FIGURES, 'allPositions_XY.png'))

        fig = plt.figure(figsize=(16, 16), facecolor=(1, 1, 1))
        ax = fig.add_subplot()
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.scatter(dfAccel['y'], dfAccel['z'], c=dfAccel['cluster_center'], cmap='Set2_r')
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.2, 1.2])
        plt.savefig(out_path(PATH_FIGURES, 'allPositions_YZ.png'))

        plt.close('all')


if __name__ == '__main__':
    main()
ax.scatter(dfAccel['x'], dfAccel['z'], c=dfAccel['position'].map(colors), alpha=1, s=100)
plt.savefig(pathFig+'rawData_allPositions_XZ.png')

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
plt.savefig(pathFig+'rawData_allPositions_XY.png')

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
plt.savefig(pathFig+'rawData_allPositions_YZ.png')
