
import numpy as np
from spherical_kde import SphericalKDE
import matplotlib.pyplot as plt
import cartopy.crs
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import pandas as pd
import re
import glob

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

def addSpherCoords(xyz):
    x = pd.to_numeric(xyz['x'])
    y = pd.to_numeric(xyz['y'])
    z = pd.to_numeric(xyz['z'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    xyz['theta'] = np.arccos(z / (np.sqrt(x**2 + y**2 + z**2)))
    xyz['phi'] = np.arctan2(y, x) # arctan2 requires (y,x)
    return(xyz)

#%%
pathIn = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/plots/'

all_files = sorted(glob.glob(pathIn + "*.parquet"), key = numericalSort)
for file in all_files:
    df = pd.read_parquet(file, engine='pyarrow')
    uIdx = df.userID.iloc[0]
    
    # convert cartesian coordinates to spherical
    df['x'] = pd.to_numeric(df['x'])
    df['y'] = pd.to_numeric(df['y'])
    df['z'] = pd.to_numeric(df['z'])
    dfSpher = addSpherCoords(df[0:4000])

    fig = plt.figure(figsize=(10, 10))
    gs_vert = GridSpec(2, 1)
    gs_lower = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_vert[1])

    fig.add_subplot(gs_vert[0], projection=cartopy.crs.Mollweide())
    fig.add_subplot(gs_vert[1], projection=cartopy.crs.Orthographic(-90, -50))

    # Choose parameters for samples
    pi = np.pi

    # Generate some samples centered on (1,1) +/- 0.3 radians
    theta_samples = dfSpher['theta']
    phi_samples = dfSpher['phi']
    kde_green = SphericalKDE(phi_samples, theta_samples)

    for ax in fig.axes:
        ax.set_global()
        ax.gridlines()
        kde_green.plot(ax, 'b')
        kde_green.plot_samples(ax)


    # Save to plot
    fig.tight_layout()
    fig.savefig(pathOut + 'user_' + str(uIdx) + '.png')
    
    print('done with user: ' + str(uIdx))