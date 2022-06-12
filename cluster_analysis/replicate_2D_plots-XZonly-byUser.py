#%%
import pandas as pd
from pyarrow import parquet
import numpy as np
import re
import glob
import spherical_kde
import cartopy.crs as ccrs
import spherical_kde.utils
from matplotlib import pyplot as plt
from itertools import combinations
import matplotlib
from matplotlib.gridspec import GridSpec
import math
from pprint import pprint
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

#######################################################################################################

pathAccel = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/'
pathTest = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/test_data/'
plotPath = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/plots/XZbyUser/'

all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)
sKDEList = []
pi = np.pi

# make equidistant points on sphere to sample
radius = 1
num = 500
regular_surf_points = regular_on_sphere_points(radius,num)
pts_xyz=np.array(regular_surf_points)
x=pts_xyz[:,0]
y=pts_xyz[:,1]
z=pts_xyz[:,2]
equi_phi = np.ndarray.round(np.mod(np.arctan2(y, x), np.pi*2),2) 
equi_theta = np.ndarray.round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2) 

for file in all_files:
    # dfAccel = pd.read_parquet(file, engine='pyarrow')
    dfAccel = pd.read_parquet(file, engine='pyarrow')
    # filter accel points to be on unit sphere:
    dfAccel = accel_filter(dfAccel)
    # convert cartesian coordinates to spherical
    addSpherCoords(dfAccel)

    # create KDE per user and day
    dfByUser = dfAccel.groupby(['userID'])
    for user, group in dfByUser:
        print('user: ' + str(user)) # user

        # if userAndWk[1] < 32:
        #     continue

        # if group size too large, remove every 4th row
        while len(group) > 70000:
            print('group size above 70000')
            print(len(group))
            group = group[np.mod(np.arange(group.index.size),4)!=0]

        if len(group) < 2:
            continue
        theta_samples = group['theta']
        phi_samples = group['phi']

        sKDE = spherical_kde.SphericalKDE(phi_samples, theta_samples, weights=None, bandwidth=0.1, density=50)
        sKDEList.append((user, sKDE))

        density_vector = np.exp(sKDE(equi_phi, equi_theta))

        # dataframe of points and densities
        arrDensities = np.vstack([x,y,z,equi_phi, equi_theta, density_vector])
        arrDensities_t = arrDensities.transpose()

# redefine the axes again to fit the axis labels from the iOS data (z coming out of page)
        dfDensities = pd.DataFrame(arrDensities_t, columns = ['z', 'x', 'y', 'phi', 'theta', 'density'])

        # 2D density plot
        fig = plt.figure()
        ax = fig.add_subplot()
        d = dfDensities['density']

        #XZ
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        p = ax.scatter(dfDensities['x'], dfDensities['z'], c=d, s=50, cmap = 'viridis_r')
        plt.colorbar(p)
        # plt.show()
        plt.savefig(plotPath + '2D_plotXZ_500pts_user_' + str(user) + '.png')
        plt.close()
        plt.clf()

#############################################################################
# test data
# all_test_files = sorted(glob.glob(pathAccel + "*.csv"), key = numericalSort)
# sKDEList_test = []

# for test_file in all_test_files:
#     dfTest = pd.read_csv(test_file, index_col=False)
#     # filter accel points to be on unit sphere:
#     accel_filter(dfTest)
#     # convert cartesian coordinates to spherical
#     addSpherCoords(dfTest)

# # plot test orientation data:
#     orientation = test_file.split('/')[8].split('.')[0]
#     print(orientation)

#     if len(dfTest) < 2:
#         continue
#     if dfTest['theta'].min() == 0:
#         thetaMean_test = 0
#     else:
#         thetaMean_test = np.mean(dfTest['theta'])
#     if dfTest['phi'].min() == 0:
#         phiMean_test = 0
#     else:
#         phiMean_test = np.mean(dfTest['phi'])
#     # theta_samples_test = np.random.normal(thetaMean_test, .1, 25*dfTest['theta'].size)
#     # phi_samples_test = np.random.normal(phiMean_test, .1, 25*dfTest['phi'].size)

#     theta_samples_test = dfTest['theta']
#     phi_samples_test = dfTest['phi']

#     sKDE_test = spherical_kde.SphericalKDE(phi_samples_test, theta_samples_test, weights=None, bandwidth=0.2, density=50)
#     sKDEList_test.append((orientation, sKDE_test))
#     density_vector_test = np.exp(sKDE_test(equi_phi, equi_theta))

#     # dataframe of points and densities
#     arrDensities_test = np.vstack([x,y,z,equi_phi, equi_theta, density_vector_test])
#     arrDensities_t_test = arrDensities_test.transpose()
# # redefine the axes again to fit the axis labels from the iOS data (z coming out of page)
#     dfDensities_test = pd.DataFrame(arrDensities_t_test, columns = ['z', 'x', 'y', 'phi', 'theta', 'density'])

#     # 2D density plot
#     fig = plt.figure()
#     ax = fig.add_subplot()
#     d_test = dfDensities_test['density']

#     #XZ
#     ax.set_xlabel('X')
#     ax.set_ylabel('Z')
#     ax.scatter(dfDensities_test['x'], dfDensities_test['z'], c=d_test, s=50, cmap = 'viridis_r')
#     # plt.show()
#     plt.savefig(plotPath + '2D_plotXZ_' + str(orientation) + '.png')
#     plt.close()
#     plt.clf()

print('finish')

# %%
