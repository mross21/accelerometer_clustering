#%%
import pandas as pd
from pyarrow import parquet
import numpy as np
import re
import glob
import scipy.optimize
import spherical_kde
import cartopy
import spherical_kde.utils
from matplotlib import pyplot as plt
from itertools import combinations
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

def addSpherCoords(xyz):
    x = pd.to_numeric(xyz['x'])
    y = pd.to_numeric(xyz['y'])
    z = pd.to_numeric(xyz['z'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    xyz['theta'] = np.arccos(z / (np.sqrt(x**2 + y**2 + z**2)))
    xyz['phi'] = np.mod(np.arctan2(y, x), np.pi*2)
    return(xyz)


#%%
pathAccel = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/test/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/'

all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)
sKDEList = []
for file in all_files:
    dfAccel = pd.read_parquet(file, engine='pyarrow')
    # convert cartesian coordinates to spherical
    dfSpher = addSpherCoords(dfAccel)
    dfSpher['dayNumber'] = dfSpher['dayNumber'].astype(float)

    # find median theta and phi per session
    dfMedCoords = dfSpher.groupby(['userID','dayNumber','sessionNumber'])[['theta', 'phi']].median()
    
    # filter accel points to be on unit sphere:
    # filters such that when the median point isn't on unit sphere, session removed
    # when using raw points, should filter above
    dfMedians_filter = accel_filter(dfMedCoords)

    # create KDE per user and day (using KDE of median theta & phi)
    dfByUser = dfMedians_filter.groupby(['userID', 'dayNumber'])
    for userAndDay, group in dfByUser:
        print('user: ' + str(userAndDay[0])) # user
        print('day: ' + str(userAndDay[1]))  # day number for that user
        # KDEsamples = spherical_kde.distributions.VonMisesFisher_sample(phi0, theta0, sigma0, size=None)
        sKDE = spherical_kde.SphericalKDE(group['phi'], group['theta'], weights=None, bandwidth=None, density=100)
        sKDEList.append((userAndDay[0], userAndDay[1], sKDE))

#%%
sKDETup = tuple(sKDEList)
comb = list(combinations(sKDETup, 2))
#%%
KL_list = []
for i in range(0, len(comb)):
    cmp = comb[i]
    p = cmp[0][2]
    q = cmp[1][2]
    # calculate (P || Q)
    kl_pq = spherical_kde.utils.spherical_kullback_liebler(p, q)
    print('User: ' + str(cmp[0][0]) + ' day: ' + str(cmp[0][1]) + ' & User: ' + str(cmp[1][0]) + ' day: ' + str(cmp[1][1]))
    print('KL(P || Q): %.3f bits' % kl_pq)
    # calculate (Q || P)
    kl_qp = spherical_kde.utils.spherical_kullback_liebler(q,p)
    print('KL(Q || P): %.3f bits' % kl_qp)
    # calculate symmetric KL
    symKL = kl_pq + kl_qp
    print('symmetric KL: %.3f bits' % symKL)
    print('==============================')

    KL_list.append((cmp[0][0], cmp[0][1], cmp[1][0], cmp[1][1], kl_pq, kl_qp, symKL))
    i += 1
#%%
dfKL = pd.DataFrame(KL_list, columns = ['first_user', 'first_user_day', 'second_user', 'second_user_day', 'KL_pq', 'KL_qp', 'symmetric_KL'])
dfKL.to_csv(pathOut + 'dfKL_sample.csv', index=False)



#%%

#########################################################################################################
# # 2021-10-22

# # dfSpher includes the spherical coordinates of the accel data


# f1 = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/test/User_3_accelData.parquet'
# f2 = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/test/User_4_accelData.parquet'
# f3 = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/User_7_accelData.parquet'
# f4 = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/User_28_accelData.parquet'



# df1 = pd.read_parquet(f1, engine='pyarrow')
# df2 = pd.read_parquet(f2, engine='pyarrow')
# df3 = pd.read_parquet(f3, engine='pyarrow')
# df4 = pd.read_parquet(f4, engine='pyarrow')
# dfSpher1 = addSpherCoords(df1)
# dfSpher2 = addSpherCoords(df2)
# dfSpher3 = addSpherCoords(df3)
# dfSpher4 = addSpherCoords(df4)


# #%%

# # MIGHT BE TOO LARGE TO RUN AS RAW DATA. FIND MEDIAN PER SESSION?


# vMFdist1 = spherical_kde.SphericalKDE(dfSpher1['phi'][0:100], dfSpher1['theta'][0:100], weights=None, bandwidth=None, density=100)
# vMFdist2 = spherical_kde.SphericalKDE(dfSpher2['phi'][0:100], dfSpher2['theta'][0:100], weights=None, bandwidth=None, density=100)
# vMFdist3 = spherical_kde.SphericalKDE(dfSpher3['phi'][0:100], dfSpher3['theta'][0:100], weights=None, bandwidth=None, density=100)
# vMFdist4 = spherical_kde.SphericalKDE(dfSpher4['phi'][0:100], dfSpher4['theta'][0:100], weights=None, bandwidth=None, density=100)

# #spherical_kde.distributions.VonMisesFisher_distribution(dfSpher2['phi'], dfSpher2['theta'], 0, 0, len(dfSpher2))
# # vMFmean = spherical_kde.distributions.VonMises_mean(dfSpher['phi'], dfSpher['theta'])
# # vMFsd = spherical_kde.distributions.VonMises_std(dfSpher['phi'], dfSpher['theta'])
# kl12 = spherical_kde.utils.spherical_kullback_liebler(vMFdist1, vMFdist2)
# print('done 12')
# kl23 = spherical_kde.utils.spherical_kullback_liebler(vMFdist2, vMFdist3)
# print('done 23')
# kl34 = spherical_kde.utils.spherical_kullback_liebler(vMFdist3, vMFdist4)
# print('done 34')
# kl13 = spherical_kde.utils.spherical_kullback_liebler(vMFdist1, vMFdist3)
# print('done 13')
# kl14 = spherical_kde.utils.spherical_kullback_liebler(vMFdist1, vMFdist4)
# print('done 14')
# kl24 = spherical_kde.utils.spherical_kullback_liebler(vMFdist2, vMFdist4)
# print('done 24')

#%%