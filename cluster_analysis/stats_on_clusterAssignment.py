#%%
import pandas as pd
import numpy as np
import re
import glob

pathIn = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/accel_with_clusters/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/accel_with_clusters/stats/'

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

# determine how much of data belongs to cluster per user
all_files = sorted(glob.glob(pathIn + "*_v2.csv"), key = numericalSort)
stats = []
for file in all_files:
        df = pd.read_csv(file, index_col=False)
        print(df['userID'].iloc[0])

        labeled = df.loc[df['cluster'] != 0]

        fracLabeled = len(labeled)/len(df) 
        stats.append((df['userID'].iloc[0], fracLabeled))
dfStats = pd.DataFrame(stats, columns = ['userID', 'fraction_labeled'])

#%%

frac_mean = np.mean(dfStats['fraction_labeled'])
frac_median = np.median(dfStats['fraction_labeled'])
frac_min = min(dfStats['fraction_labeled'])
frac_max = max(dfStats['fraction_labeled'])
usersLow = dfStats.loc[dfStats['fraction_labeled'] < 0.85]

#%%
dfStats.to_csv(pathOut + 'labeled_data_stats_v2.csv', index=False)

# %%
