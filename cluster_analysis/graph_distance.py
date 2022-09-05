#%%
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
from math import cos, sin, asin, sqrt, exp
import networkx as nx

pathIn = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/kde_sampled_points/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/graph_matrices/'
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
edge_weights = pd.DataFrame(squareform(pdist(grp1[['density']], lambda u,v: sigma*1/2*(u+v))), index=grp1.index, columns=grp1.index)

# matrix of adjacent weights
adj_weights = edge_weights*adjMatrix

# np.savetxt(pathOut +"distance_matrix.csv", dm, delimiter=",")
# np.savetxt(pathOut +"adjacency_matrix.csv", adjMatrix, delimiter=",")
# np.savetxt(pathOut +"all_edge_weights_matrix.csv", edge_weights, delimiter=",")
# np.savetxt(pathOut +"adjacent_edge_weights_matrix.csv", adj_weights, delimiter=",")


G = nx.from_numpy_matrix(np.array(adj_weights), parallel_edges=False, create_using=nx.Graph())


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
net = Network(height = "800px", width = "800px", notebook = True)
# np.random.seed(111)
# import graph
net.from_nx(G)
# save interactive plot
net.show(pathOut + 'weighted_graph.html')

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
