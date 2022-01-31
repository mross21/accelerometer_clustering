#%%
# 2022-01-31
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

#pathAccel = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/'# /test2/'
pathAccel = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/test_data/'
plotPath = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/plots/replicate_KDE_plots/XZ/'

all_files = sorted(glob.glob(pathAccel + "*.csv"), key = numericalSort)
# all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)
sKDEList = []
pi = np.pi

# make equidistant points on sphere to sample
radius = 1
num = 10000
regular_surf_points = regular_on_sphere_points(radius,num)
pts_xyz=np.array(regular_surf_points)
x=pts_xyz[:,0]
y=pts_xyz[:,1]
z=pts_xyz[:,2]
equi_phi = np.ndarray.round(np.mod(np.arctan2(y, x), np.pi*2),2) 
equi_theta = np.ndarray.round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2) 

for file in all_files:
    # dfAccel = pd.read_parquet(file, engine='pyarrow')
    dfAccel = pd.read_csv(file, index_col=False)

    # filter accel points to be on unit sphere:
    accel_filter(dfAccel)

    # convert cartesian coordinates to spherical
    addSpherCoords(dfAccel)

    # create KDE per user and day (using KDE of median theta & phi)
#     dfByUser = dfAccel.groupby(['userID', 'dayNumber'])
#     for userAndDay, group in dfByUser:
#         print('user: ' + str(userAndDay[0])) # user
#         print('day: ' + str(userAndDay[1]))  # day number for that user

#         if len(group) < 2:
#             continue
#         theta_samples = group['theta']
#         phi_samples = group['phi']

#         sKDE = spherical_kde.SphericalKDE(phi_samples, theta_samples, weights=None, bandwidth=0.1, density=50)
#         sKDEList.append((userAndDay[0], userAndDay[1], sKDE))

#         # points_arr = np.meshgrid(theta_points, phi_points) # np.vstack([theta_points, phi_points])
#         density_vector = np.exp(sKDE(equi_phi, equi_theta))

#         # dataframe of points and densities
#         arrDensities = np.vstack([x,y,z,equi_phi, equi_theta, density_vector])
#         arrDensities_t = arrDensities.transpose()
# # redefine the axes again to fit the axis labels from the iOS data (z coming out of page)
#         dfDensities = pd.DataFrame(arrDensities_t, columns = ['z', 'x', 'y', 'phi', 'theta', 'density'])

#         # 2D density plot
#         fig = plt.figure()
#         ax = fig.add_subplot()
#         d = dfDensities['density']

#         #XZ
#         ax.set_xlabel('X')
#         ax.set_ylabel('Z')
#         ax.scatter(dfDensities['x'], dfDensities['z'], c=d, s=50, cmap = 'viridis_r')
#         plt.show()
#         #plt.savefig(plotPath + '2D_plotXZ_user_' + str(userAndDay[0]) + '_day_' + str(userAndDay[1]) + '.png')


    orientation = file.split('/')[8].split('.')[0]

    print(orientation)

    if len(dfAccel) < 2:
        continue
    if dfAccel['theta'].min() == 0:
        thetaMean = 0
    else:
        thetaMean = np.mean(dfAccel['theta'])
    if dfAccel['phi'].min() == 0:
        phiMean = 0
    else:
        phiMean = np.mean(dfAccel['phi'])
    # theta_samples_test = np.random.normal(thetaMean, .1, 25*dfAccel['theta'].size)
    # phi_samples_test = np.random.normal(phiMean, .1, 25*dfAccel['phi'].size)

    theta_samples = dfAccel['theta']
    phi_samples = dfAccel['phi']

    sKDE = spherical_kde.SphericalKDE(phi_samples, theta_samples, weights=None, bandwidth=0.2, density=50)
    sKDEList.append((orientation, sKDE))

    # points_arr = np.meshgrid(theta_points, phi_points) # np.vstack([theta_points, phi_points])
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
    ax.scatter(dfDensities['x'], dfDensities['z'], c=d, s=50, cmap = 'viridis_r')
    plt.show()
    # plt.savefig(plotPath + '2D_plotXZ_user_' + str(userAndDay[0]) + '_day_' + str(userAndDay[1]) + '.png')
    #plt.savefig(plotPath + '2D_plotXZ_' + str(orientation) + '.png')


print('finish')

# %%
