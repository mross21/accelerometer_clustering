#%%
import pandas as pd
from pyarrow import parquet
import re
import glob

pathIn = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/keypress/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/'

#%% FUNCTIONS

# sort files by numerical order
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

#%% sort all csv files in pathIn 
all_files = sorted(glob.glob(pathIn + "*.parquet"), key = numericalSort)

session_list = []
for filename in all_files:
    # read parquet into dataframe
    df = pd.read_parquet(filename, engine='pyarrow')
    if len(df) < 2:
        continue
    user = df['userID'].iloc[0]
    total_session = df['sessionNumber'].max()
    session_list.append((user, total_session))


dfOut = pd.DataFrame(session_list, columns = ['userID', 'total_sessions'])

print(dfOut['total_sessions'].median())
print(dfOut['total_sessions'].mean())

dfOut.to_csv(pathOut + 'participant_stats.csv', index=False)

print('finish')







# %%
