#%%
import pandas as pd
import numpy as np
import re
import glob
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

# gets rid of the warnings for setting var to loc
pd.options.mode.chained_assignment = None

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
# files and paths
pathAccel = '/home/mindy/Desktop/BiAffect-iOS/data_processing/processed_outputs/accel/'
pathFig = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/dbscan/python_plots/'
pathAccOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/accel_with_clusters/open_science/dbscan/'
# kFile = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/dbscan/k_list.csv'

# dfK = pd.read_csv(kFile, index_col=False)
# list of user accel files
all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)

for file in all_files:
    # read in user's accelerometer file
    df = pd.read_parquet(file, engine='pyarrow')
    user = df['userID'].iloc[0]
    print(user)

    if user != 4:
        continue

    # filter accel points to be on unit sphere:
    df = accel_filter(df)

    # add spherical coordinates
    addSpherCoords(df)

    # for saving updated accelerometer file
    accOut = []

    dfAccByWk = df.groupby(['userID', 'weekNumber'])
    for userAndWk, accGrp in dfAccByWk:
        wk = userAndWk[1]
        print('user: ' + str(user)) # user
        print('week: ' + str(wk))  # week number

        if wk != 2: 
            continue
        
        # print(len(accGrp))
        # # if group size too large, remove every 4th row
        # while len(accGrp) > 40000:
        #     print('group size too large')
        #     print(len(accGrp))
        #     accGrp = accGrp[np.mod(np.arange(accGrp.index.size),4)!=0]

        # # get number of clusters
        # try:
        #     k = int(dfK.loc[(dfK['userID']==user) & (dfK['weekNumber']==wk)]['k'])
        # except TypeError:
        #     continue

        accGrp['x'] = pd.to_numeric(accGrp['x'])
        accGrp['y'] = pd.to_numeric(accGrp['y'])
        accGrp['z'] = pd.to_numeric(accGrp['z'])

        # dbscan
        dbscan = DBSCAN(metric='cosine', n_jobs=-1)
        accGrp['cluster']=dbscan.fit_predict(accGrp[['x','y','z']])
        # accGrp['cluster'] = dbscan.predict(accGrp[['x','y','z']])+1

        plt.rcParams.update({'font.size': 32})
        # XZ Plot
        fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
        ax = fig.add_subplot()
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.scatter(accGrp['x'], accGrp['z'], c=accGrp['cluster'], cmap='Set2')
        plt.xlim([-1.2,1.2])
        plt.ylim([-1.2,1.2])
        plt.show()
        # plt.savefig(pathFig+'user_' + str(int(user)) + '_week_' + str(int(wk)) + '_plot_dbscan-xz.png')
        break
        # # XY Plot
        # fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
        # ax = fig.add_subplot()
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.scatter(accGrp['x'], accGrp['y'], c=accGrp['cluster'], cmap='Set2')
        # plt.xlim([-1.2,1.2])
        # plt.ylim([-1.2,1.2])
        # # plt.show()
        # plt.savefig(pathFig+'user_' + str(int(user)) + '_week_' + str(int(wk)) + '_plot_dbscan-xy.png')
        
        # # YZ plot
        # fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
        # ax = fig.add_subplot()
        # ax.set_xlabel('Y')
        # ax.set_ylabel('Z')
        # ax.scatter(accGrp['y'], accGrp['z'], c=accGrp['cluster'], cmap='Set2')
        # plt.xlim([-1.2,1.2])
        # plt.ylim([-1.2,1.2])
        # # plt.show()
        # plt.savefig(pathFig+'user_' + str(int(user)) + '_week_' + str(int(wk)) + '_plot_dbscan-yz.png')
        
        # plt.close('all')


#         # append updated accel group
#         accOut.append(accGrp)

# # save updated accel data to csv
#     if len(accOut) < 1:
#         continue
#     dfAccOut = pd.concat(accOut,axis=0,ignore_index=True)
#     # dfAccOut.to_csv(pathAccOut + 'user_'+str(int(user))+'_accel_withClusters.csv', index=False)
#     dfAccOut.to_parquet(pathAccOut + 'user_'+str(int(user))+'_accel_withClusters-dbscan.parquet')
#     # print('finished saving updated accelerometer data')
#     # print('=========================================================================================')
    
print('finish')

# %%