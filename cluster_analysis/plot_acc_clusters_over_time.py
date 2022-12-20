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


# all_files = sorted(glob.glob(pathAccel + "*.csv"), key = numericalSort)
all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)

dayOfWk_list = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

for file in all_files:
    # dfAcc = pd.read_csv(file, index_col=False)
    dfAcc = pd.read_parquet(file, engine='pyarrow')
    user=dfAcc['userID'].iloc[0]
    print(user)
    # if user < 2:
    #     continue

    dfAcc['hour'] = pd.to_datetime(dfAcc['sessionTimestampLocal']).dt.hour
    dfAcc["dateHour"] = pd.to_datetime(dfAcc['sessionTimestampLocal']).dt.strftime('%Y-%m-%d %H')
    dfAcc['hourOfWeek'] = (dfAcc['dayOfWeek'] * 24) + (dfAcc['hour'])
    dfAcc['sessionNumByWk'] = dfAcc.groupby(['sessionNumber'], as_index=False).ngroup()

    # grouping = 'date'
    grouping = 'weekNumber'

    grouping_list = list(dfAcc[grouping].unique())
    x_list = []
    y_list = []
    for t in grouping_list:
        grp = dfAcc.loc[dfAcc[grouping] == t]
        grp['sessionNumByWk'] = grp.groupby(['sessionNumber'], as_index=False).ngroup()

        groupingAgg = 'hourOfWeek'
        # groupingAgg = 'hour'
        # groupingAgg = 'sessionNumByWk'
        # groupingAgg = 'sessionNumber'
        # grpAgg = grp.groupby(groupingAgg, as_index=False)[['date','hour','dayHour','sessionNumber','cluster_center','cluster_center_idx','cluster_center_x','cluster_center_y','cluster_center_z']].transforn(pd.Series.mode)
        grpAgg = grp.groupby(groupingAgg, as_index=False)[['date','hour','dateHour','hourOfWeek','sessionNumByWk',
                                                'sessionNumber','cluster_center','cluster_center_idx','cluster_center_x',
                                                'cluster_center_y','cluster_center_z']].agg(lambda x:x.value_counts().index[0])
        
        # # remove instances when two clusters equally present in grouping (2 modes)      
        # grpAgg['flag'] = grpAgg.apply(lambda row: np.where(len(np.shape(row['cluster_center']))==0,True,False), axis=1)
        # grpAgg = grpAgg.loc[grpAgg['flag'] == True]
        # rename x,y,z columns
        grpAgg = grpAgg.rename(columns={'cluster_center_x': 'x', 'cluster_center_y': 'y', 'cluster_center_z': 'z'})
        
        # add spherical coordinates
        grpAgg = addSpherCoords(grpAgg)


        # # set point to measure distances from
        # uprightPt = [3.14,0] # theta, phi for 0,-1,0
        # # get distance from set point
        # grpAgg['dist_from_upright'] = grpAgg[['theta','phi']].apply(lambda row: haversine_dist(row,uprightPt), axis=1)

        # distance from previous
        coords = list(zip(grpAgg['theta'],grpAgg['phi']))
        # dist_list = []
        succ_dist = [haversine_dist(coords[r], coords[r-1]) for r in range(1,len(grpAgg))]
        succ_dist.insert(0,float('NaN'))
        grpAgg['dist_from_previous'] = succ_dist
        grpAgg['date'] = pd.to_datetime(grpAgg['date'])
        grpAgg['dateHour'] = pd.to_datetime(grpAgg['dateHour'])

        grpAgg['dist_from_previous'] = np.where((grpAgg['hourOfWeek'] - grpAgg['hourOfWeek'].shift(1)) < 24, #pd.to_timedelta(24, unit = 'hours'), 
                                                grpAgg['dist_from_previous'], float('NaN'))
##########################################################################################
# single plots
        # # XZ plot
        # fig = plt.figure(figsize=(16,16))
        # ax = fig.add_subplot()
        # ax.set_xlabel('X')
        # ax.set_ylabel('Z')
        # # ax.scatter(grpAgg['clust_ctr_x'], grpAgg['clust_ctr_z'], c=grpAgg['clust_ctr'], cmap='Set2')
        # ax.plot(grpAgg['x'], grpAgg['z'], linestyle='-')
        # plt.xlim([-1.2,1.2])
        # plt.ylim([-1.2,1.2])
        # plt.show()
        # plt.savefig(pathFig+'user_' + str(int(user)) + '_week_' + str(int(groupedBy)) + '_plot_by_graphDistance-xz-2.png')

        # # plot by session number
        # fig = plt.figure(figsize=(20,10))
        # ax = fig.add_subplot()
        # ax.set_xlabel('session number')
        # ax.set_ylabel('distance from previous')
        # ax.plot(grpAgg['sessionNumber'], grpAgg['dist_from_previous'], linestyle='-')
        # ax.scatter(grpAgg['sessionNumber'], grpAgg['dist_from_previous'], c='r', s=10)
        # # plt.xlim([-1.2,1.2])
        # # plt.ylim([-1.2,1.2])
        # plt.show()
        # plt.savefig(pathFig+'user_' + str(int(user)) + '_week_' + str(int(groupedBy)) + '_plot_by_graphDistance-xz-2.png')
    
        # # plot by day/hour
        # fig = plt.figure(figsize=(20,10))
        # ax = fig.add_subplot()
        # ax.set_xlabel('day hour')
        # ax.set_ylabel('Distance from Previous')
        # ax.plot(grpAgg['dayHour'], grpAgg['dist_from_previous'], linestyle='-')
        # ax.scatter(grpAgg['dayHour'], grpAgg['dist_from_previous'], c='r')
        # # plt.xlim([-1.2,1.2])
        # # plt.ylim([-1.2,1.2])
        # plt.show()
        # plt.savefig(pathFig+'user_' + str(int(user)) + '_week_' + str(int(groupedBy)) + '_plot_by_graphDistance-xz-2.png')
    
##########################################################################################
# individual plot that when stacked are similar to Elena's
        # # plot by hour for each day
        # fig = plt.figure(figsize=(20,10),facecolor=(1, 1, 1))
        # plt.plot(grpAgg['hour'], grpAgg['dist_from_previous'], linestyle='-')
        # plt.scatter(grpAgg['hour'], grpAgg['dist_from_previous'], c='r')
        # plt.xlim([0,23])
        # plt.ylim([-0.25, 2.5])
        # plt.xlabel('Hour')
        # plt.ylabel('Distance from Previous')
        # # plt.show()
        # plt.savefig(pathFig+'user_' + str(int(user)) + '_day_' + str(int(t)) + '.png')
        # plt.close()

##########################################################################################
# # stacked line plots (similar to Elena's)
#     # https://stackoverflow.com/questions/71848923/adding-vertically-stacked-3-row-subplots-to-matplotlib-in-for-loop
#         x_list.append(grpAgg['hour'])
#         y_list.append(grpAgg['dist_from_previous'])

#     fig, ax = plt.subplots(nrows=len(grouping_list), ncols=1, sharex=True, figsize = (10,len(grouping_list)/1.5),facecolor=(1, 1, 1)) 
#     fig.supxlabel('Hour')
#     # fig.supylabel('Day')
#     for i, (d,x,y) in enumerate(zip(grouping_list,x_list,y_list)):
#         ax[i].plot(x,y)
#         ax[i].scatter(x,y, c='r', s=20)
#         ax[i].set_ylabel(d, rotation = 75)
#         ax[i].set_xlim([0,23])
#         ax[i].set_ylim([-0.25, 2.5])
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(pathFig+'user_' + str(int(user)) + '_allDays.png')
#     plt.close()

# # stacked line plots (similar to Elena's)
#     # https://stackoverflow.com/questions/71848923/adding-vertically-stacked-3-row-subplots-to-matplotlib-in-for-loop
#         x_list.append(grpAgg['sessionNumByWk'])
#         y_list.append(grpAgg['dist_from_previous'])
#     fig, ax = plt.subplots(nrows=len(grouping_list), ncols=1, sharex=True,facecolor=(1, 1, 1)) 
#     fig.supxlabel('Session Number of Week')
#     # fig.supylabel('Day')
#     for i, (d,x,y) in enumerate(zip(grouping_list,x_list,y_list)):
#         ax[i].plot(x,y)
#         ax[i].scatter(x,y, c='r', s=20)
#         ax[i].set_ylabel(d, rotation = 75)
#         # ax[i].set_xlim([0,23])
#         ax[i].set_ylim([-0.25, 2.5])
#     plt.tight_layout()
#     plt.show()
#     # plt.savefig(pathFig+'user_' + str(int(user)) + '_allDays.png')
#     # plt.close()

# stacked line plots (similar to Elena's)
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
