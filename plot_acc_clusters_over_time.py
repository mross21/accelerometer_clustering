"""
@author: Mindy Ross
python version 3.7.4
pandas version: 1.3.5
numpy version: 1.19.2
"""
# Plot changes to phone orientation preference (accelerometer cluster label) over time

#%%
import pandas as pd
import numpy as np
from pyarrow import parquet
import re
import glob
from matplotlib import pyplot as plt
from math import cos, sin, asin, sqrt

pathAccel = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/accel_with_clusters/open_science/'
pathFig = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/cluster_plots_over_time/open_science/'

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

# taken from haversine package (removed conversion to radians)
# https://github.com/mapado/haversine/blob/main/haversine/haversine.py
def haversine_dist(pt1,pt2): # theta, phi
    lat1, lng1 = pt1
    lat2, lng2 = pt2
    # convert theta range from 0 to pi to -pi/2 to pi/2 | LATITUDE (-90 to 90)
    lat1 = lat1 - (np.pi/2)
    lat2 = lat2 - (np.pi/2)
    # convert phi range from 0 to 2pi to -pi to pi | LONGITUDE (-180 to 180)
    lng1 = lng1 - np.pi
    lng2 = lng2 - np.pi
    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = (sin(lat * 0.5)**2) + (cos(lat1) * cos(lat2) * (sin(lng * 0.5)**2))
    return 2  * asin(sqrt(d))

def addSpherCoords(xyz): # from spherical_kde function
    x = pd.to_numeric(xyz['z'])
    y = pd.to_numeric(xyz['x'])
    z = pd.to_numeric(xyz['y'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    xyz['phi'] = round(np.mod(np.arctan2(y, x), np.pi*2),2).astype(float)
    xyz['theta'] = round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2).astype(float)
    return(xyz)

all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)

dayOfWk_list = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

# loop through all user files
for file in all_files:
    dfAcc = pd.read_parquet(file, engine='pyarrow')
    user=dfAcc['userID'].iloc[0]
    print(user)

    dfAcc['hour'] = pd.to_datetime(dfAcc['sessionTimestampLocal']).dt.hour
    dfAcc["dateHour"] = pd.to_datetime(dfAcc['sessionTimestampLocal']).dt.strftime('%Y-%m-%d %H')
    dfAcc['hourOfWeek'] = (dfAcc['dayOfWeek'] * 24) + (dfAcc['hour'])
    dfAcc['sessionNumByWk'] = dfAcc.groupby(['sessionNumber'], as_index=False).ngroup()

    grouping = 'weekNumber'
    grouping_list = list(dfAcc[grouping].unique())
    x_list = []
    y_list = []
    # loop through all weeks per user
    for t in grouping_list:
        grp = dfAcc.loc[dfAcc[grouping] == t]
        grp['sessionNumByWk'] = grp.groupby(['sessionNumber'], as_index=False).ngroup()

        groupingAgg = 'hourOfWeek'
        grpAgg = grp.groupby(groupingAgg, as_index=False)[['date','hour','dateHour','hourOfWeek','sessionNumByWk',
                                                'sessionNumber','cluster_center','cluster_center_idx','cluster_center_x',
                                                'cluster_center_y','cluster_center_z']].agg(lambda x:x.value_counts().index[0])
        
        # rename x,y,z columns
        grpAgg = grpAgg.rename(columns={'cluster_center_x': 'x', 'cluster_center_y': 'y', 'cluster_center_z': 'z'})
        
        # add spherical coordinates
        grpAgg = addSpherCoords(grpAgg)

        # distance from previous
        coords = list(zip(grpAgg['theta'],grpAgg['phi']))
        succ_dist = [haversine_dist(coords[r], coords[r-1]) for r in range(1,len(grpAgg))]
        succ_dist.insert(0,float('NaN'))
        grpAgg['dist_from_previous'] = succ_dist
        grpAgg['date'] = pd.to_datetime(grpAgg['date'])
        grpAgg['dateHour'] = pd.to_datetime(grpAgg['dateHour'])

        grpAgg['dist_from_previous'] = np.where((grpAgg['hourOfWeek'] - grpAgg['hourOfWeek'].shift(1)) < 24, 
                                                grpAgg['dist_from_previous'], float('NaN'))
##########################################################################################
# stacked line plots
    # https://stackoverflow.com/questions/71848923/adding-vertically-stacked-3-row-subplots-to-matplotlib-in-for-loop
        x_list.append(grpAgg[groupingAgg])
        y_list.append(grpAgg['dist_from_previous'])
    
    try:
        fig, ax = plt.subplots(nrows=len(grouping_list), ncols=1, sharex=True,facecolor=(1, 1, 1), figsize = (10,len(grouping_list))) 
        fig.supxlabel('Hour of Week (Monday - Sunday)')
        fig.supylabel('Distance From Previous Cluster')
        for i, (y_label,x,y) in enumerate(zip(grouping_list,x_list,y_list)):
            for t in [24,48,72,96,120,144]:
                ax[i].axvline(x=t,color='black',linewidth=0.5)
            ax[i].plot(x,y)
            ax[i].scatter(x,y, c='r', s=20)
            ax[i].set_ylabel('Week ' + str(y_label))
            ax[i].set_xlim([-0.25,168])
            ax[i].set_ylim([-0.25, 2.75])
        plt.tight_layout()
        # plt.show()
        plt.savefig(pathFig+'user_' + str(int(user)) + '_allDays.png')
        plt.close()
    except TypeError:
        continue

print('finish')
# %%
