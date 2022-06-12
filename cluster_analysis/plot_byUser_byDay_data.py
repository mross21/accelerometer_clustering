#%%
import numpy as np
from spherical_kde import SphericalKDE
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs
from matplotlib.gridspec import GridSpec
import pandas as pd
import re
import glob
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

##################################################################################################
pathIn = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/'
pathOutDay = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/plots/userAndDay/'
pathOutUser = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/plots/user/'

pi = np.pi

all_files = sorted(glob.glob(pathIn + "*.parquet"), key = numericalSort)
for file in all_files:
    df = pd.read_parquet(file, engine='pyarrow')
    uIdx = df['userID'].iloc[0]

    if uIdx < 20:
        continue
    
    dfMedCoords = df.groupby(['userID','dayNumber','sessionNumber'])[['x','y','z']].median()

    # filter accel points to be on unit sphere
    accel_filter(df)
    accel_filter(dfMedCoords)

    # convert cartesian coordinates to spherical
    addSpherCoords(df)
    addSpherCoords(dfMedCoords)

    # plot all data for user
    print('plot all data for user: ' + str(uIdx))
    figU = plt.figure(figsize=(10, 10))
    gs_vertU = GridSpec(nrows=4, ncols=1)

    figU.add_subplot(gs_vertU[0], projection=cartopy.crs.Mollweide())
    figU.add_subplot(gs_vertU[1], projection=cartopy.crs.Orthographic(0,0))
    figU.add_subplot(gs_vertU[2], projection=cartopy.crs.Orthographic(0,-90))
    figU.add_subplot(gs_vertU[3], projection=cartopy.crs.Orthographic(180,0))

    # Choose parameters for samples
    theta_samplesU = dfMedCoords['theta']
    phi_samplesU = dfMedCoords['phi']
    sKDEU = SphericalKDE(phi_samplesU, theta_samplesU, weights=None, bandwidth=0.1, density=50)

    try:
        for axU in figU.axes:
            axU.set_global()
            axU.gridlines()
            sKDEU.plot(axU)
            #sKDEU.plot_samples(ax)
    except ValueError:
        pass

    # Save to plot
    figU.tight_layout()
    figU.savefig(pathOutUser + 'user_' + str(uIdx) + '_medianPerSession-bw0' + str(sKDEU.bandwidth)[2:] + 'd' + str(sKDEU.density) + '.png')
    matplotlib.pyplot.close()


    # plot per user and day
    dfByUser = df.groupby(['userID', 'dayNumber'])
    for userAndDay, group in dfByUser:
        print('user: ' + str(userAndDay[0])) # user
        print('day: ' + str(userAndDay[1]))  # day number for that user

        if len(group) < 2:
            continue

        # reduce group to 1/2 total rows if too many rows
        while len(group) > 28000:
            group = group[::2]

        # plot
        fig = plt.figure(figsize=(10, 10))
        gs_vert = GridSpec(nrows=4, ncols=1)

        fig.add_subplot(gs_vert[0], projection=cartopy.crs.Mollweide())
        fig.add_subplot(gs_vert[1], projection=cartopy.crs.Orthographic(0,0))
        fig.add_subplot(gs_vert[2], projection=cartopy.crs.Orthographic(0,-90))
        fig.add_subplot(gs_vert[3], projection=cartopy.crs.Orthographic(180,0))

        # Choose parameters for samples
        theta_samples = group['theta']
        phi_samples = group['phi']
        sKDE = SphericalKDE(phi_samples, theta_samples, weights=None, bandwidth=0.1, density=50)

        try: 
            for ax in fig.axes:
                ax.set_global()
                ax.gridlines()
                sKDE.plot(ax, 'b', extend='both')
                #sKDE.plot_samples(ax)
        except ValueError:
            continue

        # Save to plot
        fig.tight_layout()
        fig.savefig(pathOutDay + 'user_' + str(userAndDay[0]) + '_day_' + str(userAndDay[1]) + '-bw0' + str(sKDE.bandwidth)[2:] + 'd' + str(sKDE.density) + '.png')
        matplotlib.pyplot.close()
    
        print('done with user ' + str(userAndDay[0]) + ' and day ' + str(userAndDay[1]))

# %%
