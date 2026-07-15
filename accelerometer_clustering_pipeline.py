import pandas as pd
import numpy as np
import glob
import spherical_kde
from scipy import spatial
from scipy.spatial.distance import squareform, pdist
import networkx as nx
from matplotlib import pyplot as plt
from accelerometer_utils import accel_filter, add_spher_coords, haversine_dist, numerical_sort, regular_on_sphere_points
from config import PATH_ACCELEROMETER_PARQUET, PATH_CLUSTERS, PATH_FIGURES
from io_utils import out_path
pd.options.mode.chained_assignment = None


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
    all_files = sorted(glob.glob(PATH_ACCELEROMETER_PARQUET + "*.parquet"), key = numerical_sort)

    radius = 1
    num = 1000
    regular_surf_points = regular_on_sphere_points(radius,num)
    pts_xyz=np.array(regular_surf_points)
    x=pts_xyz[:,0]
    y=pts_xyz[:,1]
    z=pts_xyz[:,2]
    equi_phi = np.ndarray.round(np.mod(np.arctan2(y, x), np.pi*2),2)
    equi_theta = np.ndarray.round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2)
    spherePts = pd.DataFrame(np.column_stack((equi_theta,equi_phi)), columns = ['theta','phi'])
    dm = pd.DataFrame(squareform(pdist(spherePts, metric=haversine_dist)), index=spherePts.index, columns=spherePts.index)
    d = 0.19
    adjMatrix = np.where(dm < d, 1, 0)
    np.fill_diagonal(adjMatrix,0)

    k_list = []
    for file in all_files:
        dfAllAccel = pd.read_parquet(file, engine='pyarrow')
        user = dfAllAccel['userID'].iloc[0]

        dfAccel = accel_filter(dfAllAccel)
        if len(dfAccel) < 2:
            continue
        addSpherCoords(dfAccel)

        accOut = []
        dfAccByWk = dfAccel.groupby(['userID', 'weekNumber'])
        for userAndWk, accGrp in dfAccByWk:
            wk = userAndWk[1]
            print('user: ' + str(user))
            print('week: ' + str(wk))

            if len(accGrp) < 2:
                continue

            try:
                sKDE = spherical_kde.SphericalKDE(accGrp['phi'], accGrp['theta'], weights=None, bandwidth=0.1)
            except ValueError:
                continue
            sKDE_densities = np.exp(sKDE(equi_phi, equi_theta))
            KDEdensities = np.column_stack([[userAndWk[0]]*len(equi_phi), [userAndWk[1]]*len(equi_phi),
                                        x,y,z,equi_phi,equi_theta,sKDE_densities])
            grpKDE = pd.DataFrame(KDEdensities, columns = ['userID','weekNumber','z', 'x', 'y', 'phi', 'theta', 'density'])

            print('finished making KDE')


            n = 9
            b,bins,patches = plt.hist(x=grpKDE['density'], bins=50)
            localMaxThreshold = bins[1]

            clusters = get_cluster_centers(dm, grpKDE['density'], n, localMaxThreshold)
            k = clusters[0]
            cluster_idx = clusters[1]

            k_list.append((user, wk, k))
            print('finished finding cluster centers')

            edge_weights = pd.DataFrame(squareform(pdist(grpKDE[['density']], lambda u,v: (np.exp(-((u+v)/2))))),
                                                        index=grpKDE.index, columns=grpKDE.index)
            adj_weights = edge_weights*adjMatrix
            G = nx.from_numpy_matrix(np.array(adj_weights), parallel_edges=False, create_using=nx.Graph())

            dictClust = {}
            dictClustIdx = {}
            dictClustX = {}
            dictClustY = {}
            dictClustZ = {}
            dictID = dict(zip(cluster_idx,range(1,len(cluster_idx)+1)))

            for i in range(0,len(adj_weights)):
                path_list = []
                for j in cluster_idx:
                    path = nx.single_source_dijkstra(G,i,j, weight='weight')
                    path_list.append((i,j,len(path[1]),path[0],path[1]))
                dfPaths = pd.DataFrame(path_list, columns = ['point_idx','cluster_idx','n_elements','weight','path'])
                min_path = dfPaths.loc[dfPaths['weight'] == min(dfPaths['weight'])]
                if len(min_path) > 1:
                    print('min path lengths equal')
                    break
                closest_cluster_idx = int(min_path['cluster_idx'])
                dictClust[i] = dictID[closest_cluster_idx]
                dictClustIdx[i] = closest_cluster_idx
                dictClustX[i] = grpKDE.loc[grpKDE.index == closest_cluster_idx]['x'].iloc[0]
                dictClustY[i] = grpKDE.loc[grpKDE.index == closest_cluster_idx]['y'].iloc[0]
                dictClustZ[i] = grpKDE.loc[grpKDE.index == closest_cluster_idx]['z'].iloc[0]

            print('finished making dictionary of equidistant sphere points and linked cluster center')

            accGrp['x'] = accGrp['x'].astype(float)
            accGrp['y'] = accGrp['y'].astype(float)
            accGrp['z'] = accGrp['z'].astype(float)
            accGrp['nodeIdx'] = nearest_neighbour(accGrp[['x','y','z']], grpKDE[['x','y','z']])
            accGrp['cluster_center'] = accGrp['nodeIdx'].map(dictClust)
            accGrp['cluster_center_idx'] = accGrp['nodeIdx'].map(dictClustIdx)
            accGrp['cluster_center_x'] = accGrp['nodeIdx'].map(dictClustX)
            accGrp['cluster_center_y'] = accGrp['nodeIdx'].map(dictClustY)
            accGrp['cluster_center_z'] = accGrp['nodeIdx'].map(dictClustZ)
            accOut.append(accGrp)

            print('finished adding cluster labels to accelerometer data')

            plt.rcParams.update({'font.size': 32})
            fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
            ax = fig.add_subplot()
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.scatter(accGrp['x'], accGrp['z'], c=accGrp['cluster_center'], cmap='Set2')
            plt.xlim([-1.2,1.2])
            plt.ylim([-1.2,1.2])
            plt.savefig(out_path(PATH_FIGURES, 'user_' + str(int(user)) + '_week_' + str(int(wk)) + '_plot_by_graphDistance-xz.png'))

            fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
            ax = fig.add_subplot()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.scatter(accGrp['x'], accGrp['y'], c=accGrp['cluster_center'], cmap='Set2')
            plt.xlim([-1.2,1.2])
            plt.ylim([-1.2,1.2])
            plt.savefig(out_path(PATH_FIGURES, 'user_' + str(int(user)) + '_week_' + str(int(wk)) + '_plot_by_graphDistance-xy.png'))

            fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
            ax = fig.add_subplot()
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            ax.scatter(accGrp['y'], accGrp['z'], c=accGrp['cluster_center'], cmap='Set2')
            plt.xlim([-1.2,1.2])
            plt.ylim([-1.2,1.2])
            plt.savefig(out_path(PATH_FIGURES, 'user_' + str(int(user)) + '_week_' + str(int(wk)) + '_plot_by_graphDistance-yz.png'))
            plt.close('all')

        if len(accOut) < 1:
            continue
        dfAccOut = pd.concat(accOut,axis=0,ignore_index=True)
        dfAccOut.to_parquet(out_path(PATH_CLUSTERS, 'user_'+str(int(user))+'_accel_withClusters.parquet'), index=False)
        print('finished saving updated accelerometer data')
        print('===========================================')

        dfK = pd.DataFrame(k_list, columns = ['userID','weekNumber','k'])
        dfK.to_parquet(out_path(PATH_CLUSTERS, 'k_list.parquet'), index=False)

    print('DONE')


if __name__ == '__main__':
    main()
