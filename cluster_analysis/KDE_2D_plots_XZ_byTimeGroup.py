#%%
import pandas as pd
# from pyarrow import parquet
import numpy as np
import re
import glob
import spherical_kde
from matplotlib import pyplot as plt
import math
# gets rid of the warnings for setting var to loc or something
pd.options.mode.chained_assignment = None
                                 
## Functions
# sort accelerometer files numerically
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

# filter accelerometer data to within magnitude range 0.95-1.05
def accel_filter(xyz):
    # xyz: dataframe with x, y, z coordinates
    x = pd.to_numeric(xyz['x'])
    y = pd.to_numeric(xyz['y'])
    z = pd.to_numeric(xyz['z'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    dfOut = xyz.loc[(xyz['r'] >= 0.95) & (xyz['r'] <= 1.05)]
    return(dfOut)

# add theta and phi coordinates to accelerometer data
def addSpherCoords(xyz): # from spherical_kde function
    # xyz: dataframe with x, y, z coordinates
    x = pd.to_numeric(xyz['z'])
    y = pd.to_numeric(xyz['x'])
    z = pd.to_numeric(xyz['y'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    xyz['phi'] = round(np.mod(np.arctan2(y, x), np.pi*2),2) 
    xyz['theta'] = round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2) 
    return(xyz)

# create coordinates of evenly spaced points
def regular_on_sphere_points(r,num):
    # r: radius of sphere
    # num: number of coordinates to create (might be slightly less to have even spacing)
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
# folder path for accelerometer files
pathAccel = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/'
# folder to save plots
plotPath = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/plots/XZ_userAndWeek/'

# get list of sorted accelerometer filenames
all_files = sorted(glob.glob(pathAccel + "*.csv"), key = numericalSort)
# all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)
pi = np.pi

# make equidistant points on sphere to sample
radius = 1 # radius of sphere
num = 10000 # number of points to create on sphere
regular_surf_points = regular_on_sphere_points(radius,num)
pts_xyz=np.array(regular_surf_points)
x=pts_xyz[:,0]
y=pts_xyz[:,1]
z=pts_xyz[:,2]
# find theta and phi of equidistant points
equi_phi = np.ndarray.round(np.mod(np.arctan2(y, x), np.pi*2),2) 
equi_theta = np.ndarray.round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2) 

# loop through accelerometer files
for file in all_files:
<<<<<<< Updated upstream
    dfAccel = pd.read_csv(file, index_col=False)
    # dfAccel = pd.read_parquet(file, engine='pyarrow')
=======
    # dfAccel = pd.read_csv(file, index_col=False)
    dfAccel = pd.read_parquet(file, engine='pyarrow')

    if dfAccel['userID'].unique() != 4:
        continue
>>>>>>> Stashed changes

    # filter accel points to only include when phone is stationary (magnitude ~1)
    dfAccel = accel_filter(dfAccel)
    # convert cartesian coordinates to spherical coordinates
    addSpherCoords(dfAccel)

    # create KDE per user and time group
    grouping = 'weekNumber' 
    dfByUser = dfAccel.groupby(['userID', grouping])
    for userGrp, group in dfByUser:
        print('user: ' + str(userGrp[0])) # user
        print('time group: ' + str(userGrp[1]))  # time group

<<<<<<< Updated upstream
        # if group size too large, remove every 4th row
        while len(group) > 70000:
            print('group size above 70000')
            print(len(group))
            group = group[np.mod(np.arange(group.index.size),4)!=0]
=======
        if userGrp[1] != 15:
            continue

        # if userGrp[0] < 230:
        #     continue

        # if (userGrp[0] > 100) & (userGrp[0] < 180):
        #     continue

        # if (userGrp[0] > 180) & (userGrp[0] < 337):
        #     continue

        # if (userGrp[0] > 337) & (userGrp[0] < 350):
        #     continue

        # if userGrp[0] > 230:
        #     break


        # # if group size too large, remove every 4th row
        # while len(group) > 69500:
        #     print('group size too big')
        #     print(len(group))
        #     group = group[np.mod(np.arange(group.index.size),4)!=0]
        
        print(len(group))
>>>>>>> Stashed changes

        # if group length less than 2 rows, skip plot
        if len(group) < 2:
            continue

        # get theta and phi coordinates
        theta_samples = group['theta']
        phi_samples = group['phi']

        # make spherical KDE for user/group
        sKDE = spherical_kde.SphericalKDE(phi_samples, theta_samples, weights=None, bandwidth=0.1, density=50)
        # make vectors of density values at each point on sphere
        density_vector = np.exp(sKDE(equi_phi, equi_theta))
        # dataframe of points and densities
        arrDensities = np.vstack([x,y,z,equi_phi, equi_theta, density_vector])
        arrDensities_t = arrDensities.transpose()
        # redefine the axes again to fit the axis labels from the iOS data (z coming out of page)
        dfDensities = pd.DataFrame(arrDensities_t, columns = ['z', 'x', 'y', 'phi', 'theta', 'density'])

        # 2D density plot
<<<<<<< Updated upstream
        fig = plt.figure()
=======
        fig = plt.figure(facecolor=(1, 1, 1))
        plt.rcParams.update({'font.size': 18})
>>>>>>> Stashed changes
        ax = fig.add_subplot()
        d = dfDensities['density']
        #XZ
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        p = ax.scatter(dfDensities['x'], dfDensities['z'], c=d, s=50, cmap = 'viridis_r')
        plt.colorbar(p)
<<<<<<< Updated upstream
        # plt.show()
        plt.savefig(plotPath + '2D_plotXZ_user_' + str(userGrp[0]) + '_timeGroup_' + str(userGrp[1]) + '.png')
        plt.close()
        plt.clf()
=======
        plt.show()
        # plt.savefig(plotPath + '2D_plotXZ_user_' + str(int(userGrp[0])) + '_week_' + str(int(userGrp[1])) + '.png') # '_fixScale.png')
        # plt.close()
        # plt.clf()

        break
>>>>>>> Stashed changes

print('finish')

# %%
