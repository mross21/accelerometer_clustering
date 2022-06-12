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

###################################################################################################
pathIn = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/test_data/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/test_data/for_plots/'

pi = np.pi
theta_samples = np.array([])
phi_samples = np.array([])

all_files = sorted(glob.glob(pathIn + "*.csv"), key = numericalSort)
for file in all_files:
    df = pd.read_csv(file, index_col=False)    

    # filter accel points to be on unit sphere
    accel_filter(df)

    # convert cartesian coordinates to spherical
    addSpherCoords(df)

    # reduce group to 1/2 total rows if too many rows
    while len(df) > 28000:
        df = df[::2]

    theta = df['theta']
    if theta.min() == 0:
        theta_mean = 0
    else:
        theta_mean = np.mean(theta)
    theta_noise = np.random.normal(theta_mean, .1, theta.shape)
    theta_samples = np.append(theta_samples, theta_noise)

    phi = df['phi'] 
    if phi.min() == 0:
        phi_mean = 0
    else:
        phi_mean = np.mean(phi)
    phi_noise = np.random.normal(phi_mean, .1, phi.shape) # when phi = 0 and 6.28, mean=0
    phi_samples = np.append(phi_samples, phi_noise)

    sKDE = SphericalKDE(phi_samples, theta_samples, weights=None, bandwidth=0.1, density=50)

# plot
fig = plt.figure(figsize=(10, 10))
gs_vert = GridSpec(nrows=4, ncols=1)
# gs_vert = GridSpec(nrows=1, ncols=1)

fig.add_subplot(gs_vert[0], projection=cartopy.crs.Mollweide())
fig.add_subplot(gs_vert[1], projection=cartopy.crs.Orthographic(0,0))
fig.add_subplot(gs_vert[2], projection=cartopy.crs.Orthographic(0,-90))
fig.add_subplot(gs_vert[3], projection=cartopy.crs.Orthographic(180,0))

for ax in fig.axes:
    ax.set_global()
    ax.gridlines()
    # ax.text(-10, 15, 'Facing \n Down', transform=cartopy.crs.Geodetic())
    # ax.text(-175, 15, 'Facing \n Up', transform=cartopy.crs.Geodetic())
    # ax.text(160, 15, 'Facing \n Up', transform=cartopy.crs.Geodetic())
    # ax.text(80, 15, 'Horizontal \n Right', transform=cartopy.crs.Geodetic())
    # ax.text(-105, 15, 'Horizontal \n Left', transform=cartopy.crs.Geodetic())
    # ax.text(-20, -70, 'Upright', transform=cartopy.crs.Geodetic())

    sKDE.plot(ax, 'b')
    #sKDE.plot_samples(ax)

    # Save to plot
fig.tight_layout()
plt.show()
#fig.savefig(pathOut + 'sim_allOrientations_labeled-bw01d50.png')
matplotlib.pyplot.close()

print('finish')


# %%

