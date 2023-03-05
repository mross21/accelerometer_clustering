#%%
# import packages
import pandas as pd
import numpy as np
from pyarrow import parquet
import re
import glob
from matplotlib import pyplot as plt

# FUNCTIONS
# sort files numerically
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

# filter accelerometer data by magnitude
def accel_filter(xyz):
    x = pd.to_numeric(xyz['x'])
    y = pd.to_numeric(xyz['y'])
    z = pd.to_numeric(xyz['z'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    dfOut = xyz.loc[(xyz['r'] >= 0.95) & (xyz['r'] <= 1.05)]
    return(dfOut)

# add spherical coordinates
def addSpherCoords(xyz): # from spherical_kde function
    x = pd.to_numeric(xyz['z'])
    y = pd.to_numeric(xyz['x'])
    z = pd.to_numeric(xyz['y'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    xyz['phi'] = round(np.mod(np.arctan2(y, x), np.pi*2),2) 
    xyz['theta'] = round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2) 
    return(xyz)


### ######################################################################################################
# FILE PATHS
# changed to UNMASCK
pathAccel = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/'
# folder where figures should be saved
# pathFig = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/graph_matrices/figures/open_science/'
# folder where output accelerometer data should be saved
# pathAccOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/accel_with_clusters/open_science/'

# list of user accel files
all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)
pathOut = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/accelerometer_3D_plots/'
### ######################################################################################################
# GET SPHERICAL KDE (VMF) FOR USER/WEEK

# loop through accelerometer files
for file in all_files:
    # read in user's accelerometer file
    dfAllAccel = pd.read_parquet(file, engine='pyarrow')
    user = dfAllAccel['userID'].iloc[0]
    print(user)

    # filter accel points to be on unit sphere:
    dfAccel = accel_filter(dfAllAccel)
    # skip past android data (r > 1 for android data)
    if len(dfAccel) < 2: 
        continue
    # convert cartesian coordinates to spherical
    addSpherCoords(dfAccel)

    # for saving updated accelerometer file
    accOut = []
    # create KDE per user and week
    dfAccByWk = dfAccel.groupby(['userID', 'weekNumber'])
    # loop through weeks
    for userAndWk, accGrp in dfAccByWk:
        wk = userAndWk[1]
        print('user: ' + str(user)) # user
        print('week: ' + str(wk))  # week number

        # if group has no data, skip group
        if len(accGrp) < 2:
            continue
        
        x=pd.to_numeric(accGrp['x'])
        y=pd.to_numeric(accGrp['y'])
        z=pd.to_numeric(accGrp['z'])


        fig = plt.figure(facecolor=(1, 1, 1), figsize = (11,11))
        plt.rcParams.update({'font.size': 16})

        ax = plt.axes(projection='3d')

        p=ax.scatter(x,y,z, alpha=0.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        ax.xaxis.labelpad = 30
        ax.yaxis.labelpad = 10
        ax.zaxis.labelpad = 20
        ax.tick_params(axis='x', which='major', pad=15, rotation=-20)
        ax.tick_params(axis='y', which='major', pad=0, rotation=0)
        ax.tick_params(axis='z', which='major', pad=12, rotation=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(9))
        ax.yaxis.set_major_locator(plt.MaxNLocator(9))
        ax.zaxis.set_major_locator(plt.MaxNLocator(9))

        ax.view_init(20,190)

        # plt.show()

        plt.savefig(pathOut + 'u'+str(user)+'_wk'+str(wk)+'.png')


    #     break
    # break


# %%
