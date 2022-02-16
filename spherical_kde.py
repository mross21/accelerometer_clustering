#%%
# 2022-01-31
import pandas as pd
from pyarrow import parquet
import numpy as np
import re
import glob
from sklearn.utils.extmath import density
import spherical_kde
#import cartopy.crs as ccrs
import spherical_kde.utils
from sklearn import manifold
import matplotlib.pyplot as plt
# gets rid of the warnings for setting var to loc or something
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
    dfOut = xyz.loc[(xyz['r'] >= 0.9) & (xyz['r'] <= 1.1)]
    return(dfOut)

def addSpherCoords(xyz): # from spherical_kde function
    x = pd.to_numeric(xyz['z'])
    y = pd.to_numeric(xyz['x'])
    z = pd.to_numeric(xyz['y'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    xyz['phi'] = round(np.mod(np.arctan2(y, x), np.pi*2),2) 
    xyz['theta'] = round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2) 
    return(xyz)

def regular_on_sphere_points(r,num):
    points = []
    #Break out if zero points
    if num==0:
        return points
    a = 4.0 * np.pi*(r**2.0 / num)
    d = np.sqrt(a)
    m_theta = int(round(np.pi / d))
    d_theta = np.pi / m_theta
    d_phi = a / d_theta

    for m in range(0,m_theta):
        theta = np.pi * (m + 0.5) / m_theta
        m_phi = int(round(2.0 * np.pi * np.sin(theta) / d_phi))
        for n in range(0,m_phi):
            phi = 2.0 * np.pi * n / m_phi
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            points.append([x,y,z])
    return points

#######################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
pathTestData = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/test_data/'
pathAccel = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/'# /test2/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/'
plotPath = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/plots/replicate_KDE_plots/'
# uprightFile = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/test_data/upright.csv'

sKDEList = []
sKDEList_noTest = []
m1 = np.array([])
m2 = np.array([])
pi = np.pi
# dfUpright = pd.read_csv(uprightFile, index_col=False)

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

# make array to label users on plots
colors = np.array([])
colors2 = np.array([])
orient_list = []

# test data
all_files_test = sorted(glob.glob(pathTestData + "*.csv"), key = numericalSort)
for test_file in all_files_test:
    dfTest = pd.read_csv(test_file, index_col=False)
    orientation = test_file.split('/')[8].split('.')[0]
    orient_list.append(orientation)
    print(orientation)
    accel_filter(dfTest)
    # convert cartesian coordinates to spherical
    addSpherCoords(dfTest)
    if len(dfTest) < 2:
        continue

    if dfTest['theta'].min() == 0:
        thetaMean = 0
    else:
        thetaMean = np.mean(dfTest['theta'])
    if dfTest['phi'].min() == 0:
        phiMean = 0
    else:
        phiMean = np.mean(dfTest['phi'])

    # print('theta')
    # print('min: ' + str(dfTest['theta'].min()))
    # print('max: ' + str(dfTest['theta'].max()))
    # print('mean: ' + str(np.mean(dfTest['theta'])))
    # print('thetaMean: ' + str(thetaMean))
    # print('phi')
    # print('min: ' + str(dfTest['phi'].min()))
    # print('max: ' + str(dfTest['phi'].max()))
    # print('mean: ' + str(np.mean(dfTest['phi'])))
    # print('phiMean: ' + str(phiMean))

    theta_samples_test = np.random.normal(thetaMean, .1, 25*dfTest['theta'].size)
    phi_samples_test = np.random.normal(phiMean, .1, 25*dfTest['phi'].size)


# this was for when the isomap wasn't working. but not working bc of computational power-not issues with test data (1/30/2022)
    # if test_file != uprightFile:
    #     theta_samples_test.append(dfUpright[''])
        # add percentage of other orientations' spher coords to dfupright


    sKDE_test = spherical_kde.SphericalKDE(phi_samples_test, theta_samples_test, weights=None, bandwidth=0.2, density=50)
    sKDEList.append((orientation, 1, sKDE_test))

    density_vector_test = np.exp(sKDE_test(equi_phi, equi_theta))

    #density_vector = density_matrix.reshape(len(equi_phi))
    if np.isnan(np.sum(density_vector_test)) == True:
        print('detected nan values in array')
        continue
    m1 = np.append(m1, density_vector_test)
    colors = np.append(colors, orientation)

# user data
all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)
for file in all_files:
    dfAccel = pd.read_parquet(file, engine='pyarrow')

    # if dfAccel['userID'].iloc[0] > 34:
    #     break

    # filter accel points to be on unit sphere:
    accel_filter(dfAccel)

    # convert cartesian coordinates to spherical
    addSpherCoords(dfAccel)

    # create KDE per user and day
    dfByUser = dfAccel.groupby(['userID', 'dayNumber'])
    for userAndDay, group in dfByUser:
        print('user: ' + str(userAndDay[0])) # user
        print('day: ' + str(userAndDay[1]))  # day number for that user

        if len(group) < 2:
            continue
        theta_samples = group['theta']
        phi_samples = group['phi']

        sKDE = spherical_kde.SphericalKDE(phi_samples, theta_samples, weights=None, bandwidth=0.1, density=50)
        sKDEList.append((userAndDay[0], userAndDay[1], sKDE))
        sKDEList_noTest.append((userAndDay[0], userAndDay[1], sKDE))

        # points_arr = np.meshgrid(theta_points, phi_points) # np.vstack([theta_points, phi_points])
        density_vector = np.exp(sKDE(equi_phi, equi_theta))

        #density_vector = density_matrix.reshape(len(equi_phi))
        if np.isnan(np.sum(density_vector)) == True:
            print('detected nan values in array')
            continue
        m1 = np.append(m1, density_vector)
        m2 = np.append(m2, density_vector)

        colors = np.append(colors, userAndDay[0])
        colors2 = np.append(colors2, userAndDay[0])

pd.DataFrame(colors, columns = ['user']).to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/colors.csv', index=False)
pd.DataFrame(colors2, columns = ['user']).to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/colors2.csv', index=False)
pd.DataFrame(orient_list, columns = ['orientation']).to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/orient_list.csv', index=False)

ndays = len(m1) / len(equi_phi)
ndays2 = len(m2) / len(equi_phi)
M = m1.reshape((int(ndays), len(equi_phi)))
M2 = m2.reshape((int(ndays2), len(equi_phi)))
# max_user = len(np.unique(colors)) + 1

pd.DataFrame(M).to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/M.csv', index=False)
pd.DataFrame(M2).to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/M2.csv', index=False)

print('finish matrix')

#%%
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

import pandas as pd
import numpy as np
from sklearn import manifold

matrix_file = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/M.csv', index_col = False)
matrix_file2 = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/M2.csv', index_col = False)
color_file = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/colors.csv', index_col=False)
color_file2 = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/colors2.csv', index_col=False)
orient_file = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/orient_list.csv', index_col=False)

orient_color_list = ['green', 'magenta', 'cyan', 'blue', 'yellow', 'lightgreen']
dfColor = pd.DataFrame({'orientation': orient_list, 'colors': orient_color_list})
#dfColor.to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/colors.csv', index=False)

M = np.array(matrix_file)
M2 = np.array(matrix_file2)

print('matrix loaded')

#%%
#ISOMAP
# nearest neighbor = 5% total datapoints
nNeighbors = int(0.05 * len(M2)) # len(M) = rows, len(M[0] = cols)

iso = manifold.Isomap(n_neighbors=nNeighbors, n_components=2)
iso.fit(M2)

###########
# save embedding dataframe for future use***
###########

isomap = iso.transform(M2)
dfIso = pd.DataFrame(isomap, columns = ['component1', 'component2'])
dfIso_c = pd.merge(dfIso, color_file, left_index=True,right_index=True) # pd.DataFrame(colors, columns = ['userID']), left_index=True,right_index=True)
dfIso_c.to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/dfIso_embedding.csv', index=False)


import matplotlib.pyplot as plt
for u in dfIso_c['user'].unique():
    print('user: ' + str(u))
    dfIso_c['colors'] = np.where(dfIso_c['user'] == u, 'red', 'black')
    dfIso_c.loc[dfIso_c['user'].isin(orient_list), "colors"] = orient_color_list

    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(111)
    ax.scatter(dfIso_c['component1'], dfIso_c['component2'], c = dfIso_c['colors'], cmap = 'hsv',
                                label=dfIso_c['user'], marker='o',alpha=1, s=50, edgecolor='black')
    ax.set_title('Isomap \n User ' + str(u))
    # plt.show() 
    plt.savefig('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/manifold_plots/iso/' + str(u) + '_iso.png')

#%%
# TSNE
tsne = manifold.TSNE(n_components=2, perplexity = 30.0)
tsne_emb = tsne.fit_transform(M)
dfTSNE = pd.DataFrame(tsne_emb, columns = ['component1', 'component2'])
dfTSNE_c = pd.merge(dfTSNE, color_file, left_index=True,right_index=True) #pd.DataFrame(colors, columns = ['userID']), left_index=True,right_index=True)
print(dfColor)
dfTSNE_c.to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/dfTSNE_embedding.csv', index=False)

#%%
###########
# save embedding dataframe for future use***
###########

for user in dfTSNE_c['user'].unique():
    print('user: ' + str(user))
    dfTSNE_c['colors'] = np.where(dfTSNE_c['user'] == user, 'red', 'black')  
    dfTSNE_c.loc[dfTSNE_c['user'].isin(orient_list), "colors"] = orient_color_list

    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(111)
    ax.scatter(dfTSNE_c['component1'], dfTSNE_c['component2'], c = dfTSNE_c['colors'], 
            cmap = 'hsv', label=dfTSNE_c['user'], marker='o',alpha=1, s=50, edgecolor='black')
    ax.set_title('tSNE \n User ' + str(user))
    plt.show()
    # plt.savefig('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/manifold_plots/tsne/' + str(user) + '_tsne-v2.png')













#%%
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

# # plot spherical sampling points
# import matplotlib.pyplot as plt
# import numpy as np
# plt.rcParams["figure.figsize"] = [10.00, 10.00]
# plt.rcParams["figure.autolayout"] = True
# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')
# r = 1
# u, v = points_arr
# x = np.cos(u) * np.sin(v)
# y = np.sin(u) * np.sin(v)
# z = np.cos(v)
# ax.plot_surface(x, y, z, cmap = plt.cm.YlGnBu_r)
# ax.scatter(x, y, z, color = "k", s = 10)
# ax.view_init(0, 0)
# plt.show()

#%%
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################


# SPECTRAL EMBEDDING

import sklearn

#adjacency matrix
spec_emb = sklearn.manifold.spectral_embedding(m3, n_components=2, eigen_solver=None, 
                        random_state=None, eigen_tol=0.0, norm_laplacian=True, drop_first=True)
spec_Emb2d = pd.DataFrame(spec_emb, columns = ['component1', 'component2'])


import matplotlib.pyplot as plt

fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_subplot(111)
ax.scatter(spec_Emb2d['component1'], spec_Emb2d['component2'], marker='.',alpha=1)
plt.show()


#%%


        # for p, t in zip(sKDE.phi, sKDE.theta):
        #     m[i][j] = tuple((p, t))
        #     if j < len(sKDE.phi):
        #         j += 1
        #     else: 
        #         continue
        # i += 1

        # if userAndDay[1] == 3:
        #     break

# pd.DataFrame(m).to_csv(pathOut + 'sKDEmatrix.csv')

#%%
# # make tuple of user, day, and sKDE
# sKDETup = tuple(sKDEList)
# # create combinations of tuples
# comb = list(combinations(sKDETup, 2))
# #%%
# # create matrix of zeros with dimensions of number of users/days
# # works for one user/all days but not multiple users- memory issues
# m = np.zeros((len(sKDETup), len(sKDETup)))
# # m.shape

# #%%
# # calculate KL divergences
# time0 = time.time()
# KL_list = []
# i = 0
# j = 1
# for idx in range(0, len(comb)):
#     time1 = time.time() 
#     # set matrix indices
#     i = i
#     j = j

#     # select combination
#     cmp = comb[idx]
#     p = cmp[0][2]
#     q = cmp[1][2]

#     # calculate (P || Q)
#     kl_pq = spherical_kde.utils.spherical_kullback_liebler(p, q)
#     print('User: ' + str(cmp[0][0]) + ' day: ' + str(cmp[0][1]) + ' & User: ' + str(cmp[1][0]) + ' day: ' + str(cmp[1][1]))
#     print('KL(P || Q): %.3f bits' % kl_pq)
#     # calculate (Q || P)
#     kl_qp = spherical_kde.utils.spherical_kullback_liebler(q,p)
#     print('KL(Q || P): %.3f bits' % kl_qp)
#     # calculate symmetric KL
#     symKL = kl_pq + kl_qp
#     print('symmetric KL: %.3f bits' % symKL)
#     KL_list.append((cmp[0][0], cmp[0][1], cmp[1][0], cmp[1][1], kl_pq, kl_qp, symKL))


#     # update matrix with symmetric KL value
#     print('i: ' + str(i))
#     print('j: ' + str(j))
#     m[i][j] = symKL
#     m[j][i] = symKL

#     # update matrix indices
#     if j == (len(sKDETup)-1):
#         i += 1
#         j = i + 1
#     else:
#         j += 1

#     # time calculations
#     elapsed = time.time() - time1
#     total_time = time.time() - time0
#     print('calculation took ' + str(round(elapsed/60, 2)) + ' minutes')
#     print('cumulative time: ' + str(round(total_time/60, 2)) + ' minutes')

#     print(str(idx+1) + ' of ' + str(len(comb)) + ' combinations')
#     print('==============================')

#     # update loop index
#     idx += 1

# # #%%
#     # create csv every iteration - for now
#     dfKL = pd.DataFrame(KL_list, columns = ['first_user', 'first_user_day', 'second_user', 'second_user_day', 'KL_pq', 'KL_qp', 'symmetric_KL'])
#     dfKL.to_csv(pathOut + 'dfKL_sample_v6.csv', index=False)
#     pd.DataFrame(m).to_csv(pathOut + 'matrix3.csv')



# #%%

# # # add other half of matrix after the fact

# f = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix2.csv'
# dfM = pd.read_csv(f)
# dfM=dfM.drop(dfM.columns[0], axis=1)
# M = np.array(dfM)
# # newM = M + M.T
# # pd.DataFrame(newM).to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix2.csv')

# #%%
# # for spectral embedding
# m2 = M + .0001
# m3 = 1/m2
# m4 = np.fill_diagonal(m3, 0)

# n = m3.shape[0]
# m3[range(n), range(n)] = 0

# pd.DataFrame(m3).to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix3.csv')


# # add in the matrix transpose addition to the above code

# #%%
# #ISOMAP
# from sklearn import manifold
# iso = manifold.Isomap(n_neighbors=8, n_components=2)
# iso.fit(M)
# manifold_2Da = iso.transform(M)
# manifold_2D = pd.DataFrame(manifold_2Da, columns = ['component1', 'component2'])

# print(manifold_2D)


# import matplotlib.pyplot as plt

# fig = plt.figure()
# fig.set_size_inches(10, 10)
# ax = fig.add_subplot(111)
# ax.scatter(manifold_2D['component1'], manifold_2D['component2'], marker='.',alpha=1)
# plt.show()

# #%%

# # SPECTRAL EMBEDDING

# import sklearn
# spec_emb = sklearn.manifold.spectral_embedding(m3, n_components=2, eigen_solver=None, 
#                         random_state=None, eigen_tol=0.0, norm_laplacian=True, drop_first=True)
# spec_Emb2d = pd.DataFrame(spec_emb, columns = ['component1', 'component2'])


# import matplotlib.pyplot as plt

# fig = plt.figure()
# fig.set_size_inches(10, 10)
# ax = fig.add_subplot(111)
# ax.scatter(spec_Emb2d['component1'], spec_Emb2d['component2'], marker='.',alpha=1)
# plt.show()


# #%%

# # TSNE
# import sklearn
# tsne = sklearn.manifold.TSNE(n_components=2, perplexity = 15.0)
# tsne_emb = tsne.fit_transform(M)
# dfTSNE = pd.DataFrame(tsne_emb, columns = ['component1', 'component2'])


# import matplotlib.pyplot as plt

# fig = plt.figure()
# fig.set_size_inches(10, 10)
# ax = fig.add_subplot(111)
# ax.scatter(dfTSNE['component1'], dfTSNE['component2'], marker='.',alpha=1)
# plt.show()














#%%

#########################################################################################################
# # 2021-10-22

# # dfSpher includes the spherical coordinates of the accel data


# f1 = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/test/User_3_accelData.parquet'
# f2 = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/test/User_4_accelData.parquet'
# f3 = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/User_7_accelData.parquet'
# f4 = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/User_28_accelData.parquet'



# df1 = pd.read_parquet(f1, engine='pyarrow')
# df2 = pd.read_parquet(f2, engine='pyarrow')
# df3 = pd.read_parquet(f3, engine='pyarrow')
# df4 = pd.read_parquet(f4, engine='pyarrow')
# dfSpher1 = addSpherCoords(df1)
# dfSpher2 = addSpherCoords(df2)
# dfSpher3 = addSpherCoords(df3)
# dfSpher4 = addSpherCoords(df4)


# #%%

# # MIGHT BE TOO LARGE TO RUN AS RAW DATA. FIND MEDIAN PER SESSION?


# vMFdist1 = spherical_kde.SphericalKDE(dfSpher1['phi'][0:100], dfSpher1['theta'][0:100], weights=None, bandwidth=None, density=100)
# vMFdist2 = spherical_kde.SphericalKDE(dfSpher2['phi'][0:100], dfSpher2['theta'][0:100], weights=None, bandwidth=None, density=100)
# vMFdist3 = spherical_kde.SphericalKDE(dfSpher3['phi'][0:100], dfSpher3['theta'][0:100], weights=None, bandwidth=None, density=100)
# vMFdist4 = spherical_kde.SphericalKDE(dfSpher4['phi'][0:100], dfSpher4['theta'][0:100], weights=None, bandwidth=None, density=100)

# #spherical_kde.distributions.VonMisesFisher_distribution(dfSpher2['phi'], dfSpher2['theta'], 0, 0, len(dfSpher2))
# # vMFmean = spherical_kde.distributions.VonMises_mean(dfSpher['phi'], dfSpher['theta'])
# # vMFsd = spherical_kde.distributions.VonMises_std(dfSpher['phi'], dfSpher['theta'])
# kl12 = spherical_kde.utils.spherical_kullback_liebler(vMFdist1, vMFdist2)
# print('done 12')
# kl23 = spherical_kde.utils.spherical_kullback_liebler(vMFdist2, vMFdist3)
# print('done 23')
# kl34 = spherical_kde.utils.spherical_kullback_liebler(vMFdist3, vMFdist4)
# print('done 34')
# kl13 = spherical_kde.utils.spherical_kullback_liebler(vMFdist1, vMFdist3)
# print('done 13')
# kl14 = spherical_kde.utils.spherical_kullback_liebler(vMFdist1, vMFdist4)
# print('done 14')
# kl24 = spherical_kde.utils.spherical_kullback_liebler(vMFdist2, vMFdist4)
# print('done 24')

#%%
