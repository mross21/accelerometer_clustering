"""
@author: Mindy Ross
python version 3.7.4
pandas version: 1.3.5
numpy version: 1.19.2
"""
# Make dataframe containing equidistant coordinates and corresponding sampled vMF density for each subject

#%%
import pandas as pd
from pyarrow import parquet
import numpy as np
import re
import glob
import spherical_kde
import spherical_kde.utils
from matplotlib.gridspec import GridSpec
import math
# gets rid of the warnings for setting var to loc
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

pathAccel = ''
plotPath = ''
pathOut = ''

all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)
sKDEList = []
pi = np.pi

# make equidistant points on sphere to sample
radius = 1
num = 1000
regular_surf_points = regular_on_sphere_points(radius,num)
pts_xyz=np.array(regular_surf_points)
x=pts_xyz[:,0]
y=pts_xyz[:,1]
z=pts_xyz[:,2]
equi_phi = np.ndarray.round(np.mod(np.arctan2(y, x), np.pi*2),2) 
equi_theta = np.ndarray.round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2) 

dfOut = pd.DataFrame([],columns = ['userID','weekNumber','z', 'x', 'y', 'phi', 'theta', 'density'])

# loop through all user files
for file in all_files:
    dfAccel = pd.read_parquet(file, engine='pyarrow')
    # filter accel points to be on unit sphere:
    df = accel_filter(dfAccel)
    # convert cartesian coordinates to spherical
    addSpherCoords(df)

    # create spherical KDE per user and week
    dfByUser = df.groupby(['userID', 'weekNumber'])
    for userAndWk, group in dfByUser:
        print('user: ' + str(userAndWk[0])) # user
        print('week: ' + str(userAndWk[1]))  # week number for that user
        
        # if group size too large, remove every 4th row
        while len(group) > 250000:
            print('group size above 250000')
            print(len(group))
            group = group[np.mod(np.arange(group.index.size),4)!=0]
        print('length group: ' + str(len(group)))

        if len(group) < 2:
            continue
        theta_samples = group['theta']
        phi_samples = group['phi']

        sKDE = spherical_kde.SphericalKDE(phi_samples, theta_samples, weights=None, bandwidth=0.1, density=50)
        sKDEList.append((userAndWk[0], userAndWk[1], sKDE))
        density_vector = np.exp(sKDE(equi_phi, equi_theta))

        # dataframe of points and densities
        arrDensities = np.vstack([[userAndWk[0]]*len(equi_phi), [userAndWk[1]]*len(equi_phi),x,y,z,equi_phi, equi_theta, density_vector])
        arrDensities_t = arrDensities.transpose()
        dfDensities = pd.DataFrame(arrDensities_t, columns = ['userID','weekNumber','z', 'x', 'y', 'phi', 'theta', 'density'])
        dfOut = dfOut.append(dfDensities)

dfOut.to_csv('*.csv', index=False)
print('finish')
