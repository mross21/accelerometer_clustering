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

# # label morning, afternoon, evening, night for each timestamp
# def timeOfDay(dataframe): 
# #    ts = pd.to_datetime(dataframe['sessionTimestampLocal'])
#     l = []
#     for t in dataframe['sessionTimestampLocal']:
#         if (t.hour >= 6) & (t.hour < 12):
#             l.append('morning')
#         elif (t.hour >=12) & (t.hour < 18):
#             l.append('afternoon')
#         elif (t.hour >= 18):
#             l.append('evening')
#         elif (t.hour < 6):
#             l.append('night')
#         else:
#             l.append('')
#     dataframe['timeOfDay'] = pd.DataFrame(l)
#     return(dataframe)

#######################################################################################################

pathAccel = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/'

all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)
sKDEList = []
pi = np.pi

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

dfOut = pd.DataFrame([],columns = ['userID','weekNumber','z', 'x', 'y', 'phi', 'theta', 'density'])

for file in all_files:
    # dfAccel = pd.read_parquet(file, engine='pyarrow')
    dfAccel = pd.read_parquet(file, engine='pyarrow')
    # filter accel points to be on unit sphere:
    df = accel_filter(dfAccel)
    # convert cartesian coordinates to spherical
    addSpherCoords(df)
    # # add time of day
    # timeOfDay(df)

    # print('user: ' + str(df['userID'].iloc[0]))
    # if df['userID'].iloc[0] < 19:
    #     continue


    # create KDE per user and day
    dfByUser = df.groupby(['userID', 'weekNumber', 'timeOfDay'])
    for filters, group in dfByUser:
        print('user: ' + str(filters[0])) # user
        print('week: ' + str(filters[1]))  # week number for that user
        print('timeOfDay: ' + str(filters[2]))
        
        # if group size too large, remove every 4th row
        while len(group) > 250000:
            # print('group size above 250000')
            # print(len(group))
            group = group[np.mod(np.arange(group.index.size),4)!=0]
        # print('length group: ' + str(len(group)))

        if len(group) < 2:
            continue
        theta_samples = group['theta']
        phi_samples = group['phi']

        try:
            sKDE = spherical_kde.SphericalKDE(phi_samples, theta_samples, weights=None, bandwidth=0.1, density=50) #changed bandwidth from 0.1
        except ValueError:
            print('ValueError: skipping KDE')
            continue
        sKDEList.append((filters[0], filters[1], filters[2], len(group), sKDE))

        density_vector = np.exp(sKDE(equi_phi, equi_theta))

        # dataframe of points and densities
        arrDensities = np.vstack([[filters[0]]*len(equi_phi),[filters[1]]*len(equi_phi),[filters[2]]*len(equi_phi),[len(group)]*len(equi_phi),x,y,z,equi_phi, equi_theta, density_vector])
        arrDensities_t = arrDensities.transpose()

        dfDensities = pd.DataFrame(arrDensities_t, columns = ['userID','weekNumber','timeOfDay','n_accelReadings','z', 'x', 'y', 'phi', 'theta', 'density'])

        dfOut = dfOut.append(dfDensities)

dfOut.to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/KDE_identification/sampledKDEdensities_byTOD_bw01_'+str(num)+'pts.csv', index=False)

print('finish')

# %%