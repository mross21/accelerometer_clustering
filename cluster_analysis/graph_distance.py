#%%
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
from math import cos, sin, asin, sqrt, exp
import networkx as nx
import re
import glob
from scipy import spatial


pathIn = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/kde_sampled_points/'
pathAccel = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/graph_matrices/'
file1000 = 'coords_with_KDEdensities_bw01-1000pts.csv'

df = pd.read_csv(pathIn + file1000, index_col=False)

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

def nearest_neighbour(points_a, points_b):
    tree = spatial.cKDTree(points_b) # indexes points to be compared
    return tree.query(points_a)[1] # get index of closest point above to coordinate

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

# # find neighbors
# def find_neighbors(ptRow, allPts):
#   idxNeighbors = []
#   i = ptRow['index']
#   return(idxNeighbors)

# variable to group user's data
grouping = 'weekNumber'
# max distance between neighbors
d = 0.17

sigma = 1

grp1 = df.loc[(df['userID'] == 1) & (df[grouping] == 1)]
 # get distance matrix of haversine distances between points
dm = pd.DataFrame(squareform(pdist(grp1[['theta','phi']], metric=haversine_dist)), index=grp1.index, columns=grp1.index)

# if distance is less than d, flag as adjacent point
adjMatrix = np.where(dm < d, 1, 0)

# matrix of all edge weights for user 1 week 1
# edge weights based on density- higher density between the two nodes, the higher the weight
edge_weights = pd.DataFrame(squareform(pdist(grp1[['density']], lambda u,v: sigma*1/2*(u+v))), index=grp1.index, columns=grp1.index)

# matrix of adjacent weights
adj_weights = edge_weights*adjMatrix

# np.savetxt(pathOut +"distance_matrix.csv", dm, delimiter=",")
# np.savetxt(pathOut +"adjacency_matrix.csv", adjMatrix, delimiter=",")
# np.savetxt(pathOut +"all_edge_weights_matrix.csv", edge_weights, delimiter=",")
# np.savetxt(pathOut +"adjacent_edge_weights_matrix.csv", adj_weights, delimiter=",")


#%%
G = nx.from_numpy_matrix(np.array(adj_weights), parallel_edges=False, create_using=nx.Graph())
# # check weights of edges
# [G.edges[i]['weight'] for i in list(G.edges)]

# from 'find_opt_k.py', found indices of cluster centers for user 1 week 1
# used find_optK() function

centers = [471, 527, 582, 637, 690, 788, 985] # targets
dictClust = {}
# find all paths for each index to the possible clusters
for i in range(0,len(adj_weights)):
  path_list = []
  # pt = adj_weights[i]
  for j in centers:
    # get path by indices
    path = nx.single_source_dijkstra(G,i,j, weight='weight')
    # get number of elements in path
    num_elements = len(path[1])
    path_list.append((i,j,num_elements,path[0],path[1]))
    dfPaths = pd.DataFrame(path_list, columns = ['point_idx','center_idx','n_elements','weight','path'])
  # figure out which cluster the source point should belong to
  min_elem = min(dfPaths['n_elements'])
  min_clust_path = dfPaths.loc[dfPaths['n_elements'] == min_elem]
  min_clust_path['min_clust_dist'] = abs(min_clust_path['center_idx'] - i)
  clust_ctr = min_clust_path.loc[min_clust_path['min_clust_dist'] == min(min_clust_path['min_clust_dist'])]['center_idx']

  # need to get which cluster is associated with smallest number difference
  # clust_idx = clust.loc[clust == clust_ctr].index
  # but the closest number might not be on same side of sphere?
  dictClust[i] = clust_ctr.iloc[0]


# # dataframe of points, the center, the number of elements to get to the center, and the path
# dfPaths = pd.DataFrame(path_list, columns = ['point_idx','center_idx','n_elements','weight','path'])

# next, identify which cluster the point should belong to based on the options
# figure out what to do if many paths have same number of elements
# then do it for all points

#%%
# get actual points for user 1 week 1
# link them to nearest sphere point
# plot where it's colored by the cluster center idx


all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)
for file in all_files:
    dfAccel = pd.read_parquet(file, engine='pyarrow')

    print(dfAccel['userID'].iloc[0])
    if dfAccel['userID'].iloc[0] > 1:
        break

    # filter accel points to be on unit sphere:
    df = accel_filter(dfAccel)

    if len(df) == 0:
        continue

    # convert cartesian coordinates to spherical
    addSpherCoords(df)

    # clust_list = []
    dfOut = []
    dfByUser = df.groupby(['userID', grouping])
    for userAndGrp, group in dfByUser:
      group = group.reset_index()
      print('user: ' + str(userAndGrp[0])) # user
      print('grouping: ' + str(userAndGrp[1]))  # time grouping for that user
      # print(len(group))

      group['nodeIdx'] = nearest_neighbour(group[['x','y','z']],grp1[['x','y','z']])

      group['clust_ctr'] = group['nodeIdx'].map(dictClust)
      print(group[['nodeIdx','clust_ctr']])

      break

    break

#%%
# plot accel data colored by new cluster IDs
# 2D density plot
from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot()
#XZ
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.scatter(group['x'], group['z'], c=group['clust_ctr'], s=50)
plt.show()


#%%
# # visualize static graph
# not great way to visualize without adjusting node/edge sizes etc
# import numpy as np
# from matplotlib import pyplot as plt

# # set seed for reproducibility
# np.random.seed(123)
# fig = nx.draw(G)
#%%
# visualize interactive graph 
from pyvis.network import Network

# create pyvis Network object

for i in G.nodes:
  G.nodes[i]['color'] = "red" if i in clust['path'][0] else "lightblue"


net = Network(height = "800px", width = "800px", notebook = True)
# np.random.seed(111)
# import graph
net.from_nx(G)
# save interactive plot
net.show(pathOut + 'weighted_graph2.html')

#%%


# grid = np.meshgrid(list_num,list_num, sparse=True)

# dfByGroup = df.groupby(['userID', grouping])
# for userGrp, grp in dfByGroup:
#     # reset indexing
#     grp = grp.reset_index()
#     user = userGrp[0]
#     print('user: ' + str(user))
#     groupedBy = userGrp[1]
#     print(str(grouping) + str(groupedBy))
#     userGrp = ';'.join([str(user),str(groupedBy)])

   

#     # index 0-997 (998 points)

#     # get a matrix of weights for each adjacent node (all cells w/ value 1)
#     # w = exp(sigma*1/2(KDE_i + KDE_j))
#     # wMatrix = adjMatrix*w
    
#     # G = nx.from_numpy_matrix(wMatrix)



#     break

#%%
# check distance matrix
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,4))
plt.imshow(dm[500:510], interpolation='nearest')
# plt.savefig('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/plots/distance_matrix_heatmap-500-510.jpg')


#%%

# other way to find neighbors
neighbors_list = []
for i in range(len(adjMatrix)):
  neighbors = np.where(adjMatrix[i] == 1)
  if np.shape(neighbors)[1] > 9:
    print(i)
    print('too many neighbors')
  neighbors_list.append((i,list(neighbors)))

dfNeighbors = pd.DataFrame(neighbors_list, columns = ['point_index','neighbors_indices'])
dfNeighbors.to_csv(pathOut + 'neighbors-d17.csv',index=False)


#%%
# ######################################################################
# # example for how to create network graph in python

# import pandas as pd
# import networkx as nx

# # create edgelist as dict
# gwork_edgelist = dict(
#   David = ["Zubin", "Suraya", "Jane"],
#   Jane = ["Zubin", "Suraya"]
# )

# # create graph dict
# gwork = nx.Graph(gwork_edgelist)

# # see vertices as list
# list(gwork.nodes)

# gwork_edgelist=dict(
#   source=["David", "David", "David", "Jane", "Jane"],
#   target=["Zubin", "Suraya", "Jane", "Zubin", "Suraya"]
# )

# #create edgelist as Pandas DataFrame
# gwork_edgelist = pd.DataFrame(gwork_edgelist)

# # create graph from Pandas DataFrame
# gwork = nx.from_pandas_edgelist(gwork_edgelist)

# gmanage_edgelist=dict(
#   David=["Zubin", "Jane"],
#   Suraya=["David"]
# )

# # create directed graph
# gmanage=nx.DiGraph(gmanage_edgelist)

# # check edges
# list(gmanage.edges)

# gwork.is_multigraph()
# gwork.is_directed()
# #%%

# import numpy as np

# # create adjacency matrix
# adj_flights = np.reshape((0,4,4,5,0,1,2,0,0), (3,3))

# # generate directed multigraph 
# multiflights = nx.from_numpy_matrix(adj_flights, parallel_edges=False, create_using=nx.MultiDiGraph())

# # name nodes
# label_mapping = {0: "SFO", 1: "PHL", 2: "TUS"}
# multiflights = nx.relabel_nodes(multiflights, label_mapping)

# # check some edges
# list(multiflights.edges)[0:3]

# # check weights of edges
# [multiflights.edges[i]['weight'] for i in list(multiflights.edges)]









# # %%
